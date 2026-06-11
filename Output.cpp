// =============================================================================
// Output.cpp — HFAQE Clean Public API
// =============================================================================
#ifndef HFAQE_OUTPUT_CPP
#define HFAQE_OUTPUT_CPP
//   (a) Other C++ translation units / downstream transformer stages
//   (b) Python via pybind11 (PyTorch-compatible numpy/tensor interface)
//
// Spec coverage:
//   §2.2  Forward pass  — embed_tokens(), embed_text()
//   §2.4  LM head       — lm_head_logits()
//   §2.4  Weight tying  — HFAQEOutput::set_tied_lm_head()
//   §3.3  mmap          — load_from_files() for cold-tier out-of-core
//   §7    ARC capability — dynamic vocabulary expansion (add_cold_token())
//
// Public C++ surface (no pybind11 required):
//   HFAQEOutput                — main API object
//     ::embed_tokens(ids)       → EmbedResult  (fp32 matrix + metadata)
//     ::embed_text(str)         → EmbedResult
//     ::lm_head_logits(h)       → LogitResult  (fp32[V])
//     ::add_cold_token(id, vec) → extend vocabulary at runtime (§7 ARC)
//     ::memory_report()         → MemoryBudget
//     ::set_tied_lm_head()      → ensure pointer equality (§5.4)
//
// pybind11 module (compiled when HFAQE_WITH_PYBIND11 is defined):
//   import hfaqe
//   model = hfaqe.HFAQE(config_dict)
//   X     = model.embed_tokens([1,2,3])    # numpy array (n, d)
//   logits= model.lm_head_logits(h_np)     # numpy array (V,)
// =============================================================================

#include "Input.cpp"    // transitively: Core.cpp + tokenizer bridge

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <unordered_map>

// =============================================================================
// EmbedResult — output of embed_tokens / embed_text
// =============================================================================
struct EmbedResult {
    std::vector<fp32>  data;      // flat fp32[n × d], row-major
    int                n;         // number of tokens
    int                d;         // model dimension
    std::vector<int>   token_ids; // the token IDs that were embedded
    bool               has_nan;   // quick NaN/Inf sentinel (§5.4)

    // Accessor: row i as pointer
    const fp32* row(int i) const {
        return data.data() + static_cast<ptrdiff_t>(i) * d;
    }
    fp32* row(int i) {
        return data.data() + static_cast<ptrdiff_t>(i) * d;
    }

    // Shape string for debugging
    std::string shape_str() const {
        return "[" + std::to_string(n) + " x " + std::to_string(d) + "]";
    }
};

// =============================================================================
// LogitResult — output of lm_head_logits
// =============================================================================
struct LogitResult {
    std::vector<fp32> logits;    // fp32[V]
    int               V;
    int               argmax;   // greedy next token
    fp32              max_logit;
    bool              has_nan;

    // Top-k indices by logit value
    std::vector<int> top_k(int k) const {
        k = std::min(k, V);
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b){ return logits[a] > logits[b]; });
        idx.resize(k);
        return idx;
    }
};


// =============================================================================
// HFAQEOutput — main public API object
// Owns (or borrows) an HFAQE model and an optional EmbedPipeline.
// =============================================================================
class HFAQEOutput {
public:

    // -----------------------------------------------------------------
    // Construction helpers
    // -----------------------------------------------------------------

    // (A) Build from scratch with Zipf frequency prior
    static std::unique_ptr<HFAQEOutput> from_config(
        int V, int d, int r, int K, int B = 64,
        uint64_t seed = 42,
        const std::string& tokenizer_path = "")
    {
        auto obj = std::make_unique<HFAQEOutput>();
        HFAQEConfig cfg;
        cfg.V = V; cfg.d = d; cfg.r = r; cfg.K = K; cfg.B = B;

        obj->model_  = std::make_unique<HFAQE>(cfg);
        auto freq    = zipf_frequencies(V);
        obj->model_->build_frequency_tiers(freq);
        obj->model_->initialize_weights(seed);

        obj->pipeline_ = std::make_unique<EmbedPipeline>(
            obj->model_.get(), tokenizer_path);

        obj->tied_lm_head_ = false;
        return obj;
    }

    // (B) Build from pre-trained HFAQE (takes ownership)
    static std::unique_ptr<HFAQEOutput> from_model(
        std::unique_ptr<HFAQE> model,
        const std::string& tokenizer_path = "")
    {
        auto obj = std::make_unique<HFAQEOutput>();
        obj->model_    = std::move(model);
        obj->pipeline_ = std::make_unique<EmbedPipeline>(
            obj->model_.get(), tokenizer_path);
        obj->tied_lm_head_ = false;
        return obj;
    }

    // (C) Load weights from checkpoint file (binary format from Train.cpp)
    static std::unique_ptr<HFAQEOutput> from_checkpoint(
        const std::string& ckpt_path,
        int V, int d, int r, int K, int B = 64,
        const std::string& tokenizer_path = "")
    {
        HFAQEConfig cfg;
        cfg.V = V; cfg.d = d; cfg.r = r; cfg.K = K; cfg.B = B;

        auto m = std::make_unique<HFAQE>(cfg);
        // Build tiers so idx maps exist before loading
        auto freq = zipf_frequencies(V);
        m->build_frequency_tiers(freq);
        m->initialize_weights(0);  // allocates buffers

        // Load weights from file (reuse save_checkpoint / load_checkpoint
        // from Train.cpp which is transitively included via Input.cpp)
        std::ifstream f(ckpt_path, std::ios::binary);
        if (!f) throw std::runtime_error(
            "HFAQEOutput: cannot open checkpoint: " + ckpt_path);

        int header[5];
        f.read(reinterpret_cast<char*>(header), sizeof(header));
        if (header[0]!=V||header[1]!=d||header[2]!=r||header[3]!=K||header[4]!=B)
            throw std::runtime_error(
                "HFAQEOutput: checkpoint config mismatch");

        f.read(reinterpret_cast<char*>(m->hot.Q_H.data()),
               static_cast<std::streamsize>(m->hot.Q_H.size()));
        f.read(reinterpret_cast<char*>(m->hot.S_H.data()),
               static_cast<std::streamsize>(m->hot.S_H.size()*sizeof(fp32)));
        f.read(reinterpret_cast<char*>(m->cold.A.data()),
               static_cast<std::streamsize>(m->cold.A.size()*sizeof(fp16)));
        f.read(reinterpret_cast<char*>(m->cold.Basis.data()),
               static_cast<std::streamsize>(m->cold.Basis.size()*sizeof(fp16)));
        f.read(reinterpret_cast<char*>(m->hot.global_ids.data()),
               static_cast<std::streamsize>(m->hot.global_ids.size()*sizeof(int)));
        f.read(reinterpret_cast<char*>(m->cold.global_ids.data()),
               static_cast<std::streamsize>(m->cold.global_ids.size()*sizeof(int)));
        // Rebuild lookup maps
        m->hot.idx.clear();
        for (int s=0; s<m->hot.K; ++s) m->hot.idx[m->hot.global_ids[s]]=s;
        m->cold.idx.clear();
        for (int s=0; s<m->cold.Vc; ++s) m->cold.idx[m->cold.global_ids[s]]=s;

        return from_model(std::move(m), tokenizer_path);
    }

    // -----------------------------------------------------------------
    // §2.2 — embed_tokens: int[] → EmbedResult
    // Primary C++ API for downstream transformer stages.
    // -----------------------------------------------------------------
    EmbedResult embed_tokens(const std::vector<int>& token_ids) const {
        if (token_ids.empty())
            return {{}, 0, model_->cfg.d, {}, false};

        int n = static_cast<int>(token_ids.size());
        int d = model_->cfg.d;

        EmbedResult res;
        res.n         = n;
        res.d         = d;
        res.token_ids = token_ids;
        res.data.resize(static_cast<size_t>(n) * d);

        model_->forward(token_ids.data(), n, res.data.data());

        // NaN/Inf sentinel (SPEC §5.4)
        res.has_nan = false;
        for (fp32 v : res.data)
            if (!std::isfinite(v)) { res.has_nan = true; break; }

        return res;
    }

    // Overload: raw pointer interface for zero-copy downstream use
    void embed_tokens(const int* ids, int n, fp32* out) const {
        model_->forward(ids, n, out);
    }

    // -----------------------------------------------------------------
    // §2.2 — embed_text: string → EmbedResult  (via tokenizer bridge)
    // -----------------------------------------------------------------
    EmbedResult embed_text(const std::string& text,
                           bool truncation  = false,
                           int  max_length  = 512) const
    {
        if (!pipeline_)
            throw std::logic_error("embed_text: no tokenizer pipeline attached");

        std::vector<int> ids = pipeline_->get_token_ids(text, truncation, max_length);
        // Clamp to [0, V)
        for (auto& id : ids)
            id = std::max(0, std::min(id, model_->cfg.V - 1));

        return embed_tokens(ids);
    }

    // Batch variant
    std::vector<EmbedResult> embed_texts(
        const std::vector<std::string>& texts,
        bool truncation = false,
        int  max_length = 512) const
    {
        std::vector<EmbedResult> out;
        out.reserve(texts.size());
        for (const auto& t : texts)
            out.push_back(embed_text(t, truncation, max_length));
        return out;
    }


    // -----------------------------------------------------------------
    // §2.4 — lm_head_logits: hidden state h[d] → LogitResult (logits[V])
    // Uses weight-tied HFAQE LM head (Algorithm 3 from SPEC).
    // -----------------------------------------------------------------
    LogitResult lm_head_logits(const fp32* h) const {
        LogitResult res;
        res.V = model_->cfg.V;
        res.logits.resize(res.V);

        model_->lm_head(h, res.logits.data());

        // Compute argmax and check for NaN/Inf
        res.has_nan   = false;
        res.max_logit = -std::numeric_limits<fp32>::infinity();
        res.argmax    = 0;
        for (int t = 0; t < res.V; ++t) {
            if (!std::isfinite(res.logits[t])) { res.has_nan = true; }
            if (res.logits[t] > res.max_logit) {
                res.max_logit = res.logits[t];
                res.argmax    = t;
            }
        }
        return res;
    }

    // Overload: vector
    LogitResult lm_head_logits(const std::vector<fp32>& h) const {
        if ((int)h.size() != model_->cfg.d)
            throw std::invalid_argument(
                "lm_head_logits: h.size() != d");
        return lm_head_logits(h.data());
    }

    // -----------------------------------------------------------------
    // §2.4 / §5.4 — Weight tying
    // Marks this object as the tied LM head so that basis_ptr() is
    // guaranteed to be the same object as the embedding's basis.
    // In a full framework, both the embedding and LM head would share
    // a single HFAQEOutput instance; this method enforces that contract.
    // Returns: raw pointer to Basis for pointer-equality assertion.
    // -----------------------------------------------------------------
    void set_tied_lm_head() {
        tied_lm_head_ = true;
        // No-op structurally — both embed and lm_head already use the
        // same model_->cold.Basis. This call records the contract.
    }

    bool is_tied() const { return tied_lm_head_; }

    // Returns raw Basis pointer (SPEC §5.4 weight tying pointer test)
    const fp16* basis_ptr() const { return model_->basis_ptr(); }
    fp16*       basis_ptr()       { return model_->basis_ptr(); }

    // -----------------------------------------------------------------
    // §7 — ARC: Dynamic vocabulary expansion
    // Adds a new cold token at runtime by learning only an r-dimensional
    // coefficient vector (no hot-tier or basis modification required).
    //
    // new_id:    the new global token ID (must be >= current V)
    // init_vec:  optional fp32[d] initializer; if nullptr, projects
    //            the given vector onto the cold basis: α = B^T · v
    // -----------------------------------------------------------------
    void add_cold_token(int new_id, const fp32* init_vec = nullptr) {
        HFAQE& m = *model_;
        int d = m.cfg.d;
        int r = m.cfg.r;

        // Grow vocabulary
        int new_V = std::max(m.cfg.V, new_id + 1);
        m.cfg.V   = new_V;

        // Allocate coefficient row α ∈ ℝ^r
        std::vector<fp16> alpha(r, fp16(0));

        if (init_vec != nullptr) {
            // Project init_vec onto basis B: α_k = Σ_j B[j,k] · init_vec[j]
            for (int k = 0; k < r; ++k) {
                const fp16* bk = m.cold.basis_col(k);
                fp32 dot = 0.0f;
                for (int j = 0; j < d; ++j)
                    dot += bf16_to_f32(bk[j]) * init_vec[j];
                alpha[k] = f32_to_bf16(dot);
            }
        }
        // else: zero initialisation — new token starts as zero embedding
        // and learns through backward pass

        // Append to cold tier
        int new_cslot = m.cold.Vc;
        m.cold.Vc += 1;
        m.cold.global_ids.push_back(new_id);
        m.cold.idx[new_id] = new_cslot;

        // Grow A storage
        m.cold.A.insert(m.cold.A.end(), alpha.begin(), alpha.end());

        // Grow grad_A if allocated
        if (!m.grad_A.empty())
            m.grad_A.resize(m.grad_A.size() + r, 0.0f);
    }

    // Overload: vector init
    void add_cold_token(int new_id, const std::vector<fp32>& init_vec) {
        if ((int)init_vec.size() != model_->cfg.d)
            throw std::invalid_argument("add_cold_token: init_vec.size() != d");
        add_cold_token(new_id, init_vec.data());
    }

    // -----------------------------------------------------------------
    // §3.3 — load_from_files: mmap cold coefficients for out-of-core use
    // -----------------------------------------------------------------
    bool load_cold_mmap(const std::string& cold_bin_path) {
        return model_->mmap_cold_coefficients(cold_bin_path);
    }

    // -----------------------------------------------------------------
    // §4.2 — memory_report: returns theoretical memory budget
    // -----------------------------------------------------------------
    MemoryBudget memory_report() const {
        return MemoryBudget::compute(model_->cfg);
    }

    // -----------------------------------------------------------------
    // Accessors for downstream C++ stages
    // -----------------------------------------------------------------
    const HFAQEConfig& config()    const { return model_->cfg; }
    HFAQE&             model()           { return *model_; }
    const HFAQE&       model()     const { return *model_; }
    EmbedPipeline*     pipeline()        { return pipeline_.get(); }

    // Default constructor (used by static factory methods)
    HFAQEOutput() = default;

private:
    std::unique_ptr<HFAQE>         model_;
    std::unique_ptr<EmbedPipeline> pipeline_;
    bool                           tied_lm_head_ = false;
};


// =============================================================================
// pybind11 Python Module — hfaqe
// =============================================================================
// Exposes HFAQEOutput to Python as a PyTorch-compatible API.
// Build with:
//   c++ -O3 -mavx512f -mavx512bw -DHFAQE_WITH_PYBIND11 \
//       -shared -fPIC $(python3 -m pybind11 --includes) \
//       Output.cpp -o hfaqe$(python3-config --extension-suffix)
//
// Python usage:
//   import hfaqe, numpy as np
//   model = hfaqe.HFAQEOutput.from_config(V=16000, d=512, r=64, K=1024)
//   X     = model.embed_tokens([1, 2, 3])   # np.ndarray shape (3, 512)
//   logits= model.lm_head_logits(h_np)      # np.ndarray shape (16000,)
// =============================================================================

#ifdef HFAQE_WITH_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// Helper: EmbedResult → numpy array (n, d) float32
static py::array_t<float> embed_result_to_numpy(const EmbedResult& res) {
    py::array_t<float> arr({static_cast<py::ssize_t>(res.n),
                             static_cast<py::ssize_t>(res.d)});
    auto buf = arr.request();
    std::memcpy(buf.ptr, res.data.data(),
                static_cast<size_t>(res.n) * res.d * sizeof(float));
    return arr;
}

PYBIND11_MODULE(hfaqe, m) {
    m.doc() = "HFAQE: Hierarchical Frequency-Adaptive Quantized Embedding";

    // ---- HFAQEConfig binding ----
    py::class_<HFAQEConfig>(m, "HFAQEConfig")
        .def(py::init<>())
        .def_readwrite("V",   &HFAQEConfig::V)
        .def_readwrite("d",   &HFAQEConfig::d)
        .def_readwrite("B",   &HFAQEConfig::B)
        .def_readwrite("r",   &HFAQEConfig::r)
        .def_readwrite("K",   &HFAQEConfig::K)
        .def_readwrite("tau", &HFAQEConfig::tau)
        .def("m", &HFAQEConfig::m, "Number of quantization blocks per row");

    // ---- EmbedResult binding ----
    py::class_<EmbedResult>(m, "EmbedResult")
        .def_readonly("n",         &EmbedResult::n)
        .def_readonly("d",         &EmbedResult::d)
        .def_readonly("token_ids", &EmbedResult::token_ids)
        .def_readonly("has_nan",   &EmbedResult::has_nan)
        .def("shape_str",          &EmbedResult::shape_str)
        .def("numpy", [](const EmbedResult& r) {
            return embed_result_to_numpy(r);
        }, "Return embeddings as a (n, d) float32 numpy array");

    // ---- LogitResult binding ----
    py::class_<LogitResult>(m, "LogitResult")
        .def_readonly("V",         &LogitResult::V)
        .def_readonly("argmax",    &LogitResult::argmax)
        .def_readonly("max_logit", &LogitResult::max_logit)
        .def_readonly("has_nan",   &LogitResult::has_nan)
        .def("top_k", &LogitResult::top_k, py::arg("k") = 10,
             "Return indices of top-k logits (descending)")
        .def("numpy", [](const LogitResult& r) {
            py::array_t<float> arr({static_cast<py::ssize_t>(r.V)});
            auto buf = arr.request();
            std::memcpy(buf.ptr, r.logits.data(), r.V * sizeof(float));
            return arr;
        }, "Return logits as a (V,) float32 numpy array");

    // ---- MemoryBudget binding ----
    py::class_<MemoryBudget>(m, "MemoryBudget")
        .def_readonly("hot_q_bytes",  &MemoryBudget::hot_q_bytes)
        .def_readonly("hot_s_bytes",  &MemoryBudget::hot_s_bytes)
        .def_readonly("cold_a_bytes", &MemoryBudget::cold_a_bytes)
        .def_readonly("basis_bytes",  &MemoryBudget::basis_bytes)
        .def_readonly("total_bytes",  &MemoryBudget::total_bytes)
        .def("total_mb", [](const MemoryBudget& b) {
            return static_cast<double>(b.total_bytes) / (1024.0 * 1024.0);
        }, "Total HFAQE memory in MB");

    // ---- HFAQEOutput binding ----
    py::class_<HFAQEOutput>(m, "HFAQEOutput")

        // Factory: from_config
        .def_static("from_config",
            [](int V, int d, int r, int K, int B,
               uint64_t seed, const std::string& tok_path)
            {
                return HFAQEOutput::from_config(V, d, r, K, B, seed, tok_path);
            },
            py::arg("V"), py::arg("d"), py::arg("r"), py::arg("K"),
            py::arg("B") = 64, py::arg("seed") = 42,
            py::arg("tokenizer_path") = "",
            "Build a fresh HFAQE model with Zipf frequency prior")

        // Factory: from_checkpoint
        .def_static("from_checkpoint",
            [](const std::string& path,
               int V, int d, int r, int K, int B,
               const std::string& tok_path)
            {
                return HFAQEOutput::from_checkpoint(path,V,d,r,K,B,tok_path);
            },
            py::arg("path"),
            py::arg("V"), py::arg("d"), py::arg("r"), py::arg("K"),
            py::arg("B") = 64, py::arg("tokenizer_path") = "",
            "Load HFAQE weights from a binary checkpoint file")

        // embed_tokens: List[int] → EmbedResult
        .def("embed_tokens",
            [](const HFAQEOutput& self, const std::vector<int>& ids) {
                return self.embed_tokens(ids);
            },
            py::arg("token_ids"),
            "Embed a list of token IDs → EmbedResult (.numpy() for ndarray)")

        // embed_tokens_numpy: List[int] → np.ndarray (n, d)
        .def("embed_tokens_numpy",
            [](const HFAQEOutput& self, const std::vector<int>& ids) {
                auto res = self.embed_tokens(ids);
                return embed_result_to_numpy(res);
            },
            py::arg("token_ids"),
            "Embed token IDs and return (n, d) float32 numpy array directly")

        // embed_text: str → EmbedResult
        .def("embed_text",
            [](const HFAQEOutput& self, const std::string& text,
               bool truncation, int max_length) {
                return self.embed_text(text, truncation, max_length);
            },
            py::arg("text"),
            py::arg("truncation") = false,
            py::arg("max_length") = 512,
            "Tokenize text and embed → EmbedResult")

        // embed_text_numpy: str → np.ndarray (n, d)
        .def("embed_text_numpy",
            [](const HFAQEOutput& self, const std::string& text,
               bool truncation, int max_length) {
                auto res = self.embed_text(text, truncation, max_length);
                return embed_result_to_numpy(res);
            },
            py::arg("text"),
            py::arg("truncation") = false,
            py::arg("max_length") = 512,
            "Tokenize text and return (n, d) float32 numpy array")

        // lm_head_logits: np.ndarray (d,) → LogitResult
        .def("lm_head_logits",
            [](const HFAQEOutput& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> h)
            {
                auto buf = h.request();
                if (buf.ndim != 1 || buf.shape[0] != self.config().d)
                    throw std::invalid_argument(
                        "lm_head_logits: expected 1-D array of length d="
                        + std::to_string(self.config().d));
                return self.lm_head_logits(
                    static_cast<const float*>(buf.ptr));
            },
            py::arg("h"),
            "Compute LM-head logits for hidden state h → LogitResult")

        // lm_head_numpy: np.ndarray (d,) → np.ndarray (V,)
        .def("lm_head_numpy",
            [](const HFAQEOutput& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> h)
            {
                auto buf = h.request();
                if (buf.ndim != 1 || buf.shape[0] != self.config().d)
                    throw std::invalid_argument("lm_head_numpy: shape mismatch");
                auto res = self.lm_head_logits(
                    static_cast<const float*>(buf.ptr));
                py::array_t<float> arr({static_cast<py::ssize_t>(res.V)});
                std::memcpy(arr.request().ptr, res.logits.data(),
                            res.V * sizeof(float));
                return arr;
            },
            py::arg("h"),
            "Compute LM-head logits → (V,) float32 numpy array")

        // §7 ARC: add_cold_token
        .def("add_cold_token",
            [](HFAQEOutput& self, int new_id,
               py::object init_vec_obj) {
                if (init_vec_obj.is_none()) {
                    self.add_cold_token(new_id);
                } else {
                    auto arr = py::array_t<float,
                        py::array::c_style|py::array::forcecast>(init_vec_obj);
                    auto buf = arr.request();
                    if (buf.ndim != 1 || buf.shape[0] != self.config().d)
                        throw std::invalid_argument(
                            "add_cold_token: init_vec must be shape (d,)");
                    self.add_cold_token(new_id,
                        static_cast<const float*>(buf.ptr));
                }
            },
            py::arg("new_id"),
            py::arg("init_vec") = py::none(),
            "Add a new cold token (§7 ARC). init_vec: optional (d,) float32")

        // Weight tying
        .def("set_tied_lm_head", &HFAQEOutput::set_tied_lm_head,
             "Mark this object as weight-tied embedding+LM-head (§2.4)")
        .def("is_tied", &HFAQEOutput::is_tied)

        // Memory report
        .def("memory_report", &HFAQEOutput::memory_report,
             "Return MemoryBudget with byte counts for each tier")
        .def("memory_mb",
            [](const HFAQEOutput& self) {
                return static_cast<double>(
                    self.memory_report().total_bytes) / (1024.0*1024.0);
            }, "Total HFAQE resident memory in MB")

        // Config access
        .def("config", &HFAQEOutput::config,
             py::return_value_policy::reference_internal)
        .def("vocab_size",
            [](const HFAQEOutput& self){ return self.config().V; })
        .def("dim",
            [](const HFAQEOutput& self){ return self.config().d; })

        // mmap cold tier
        .def("load_cold_mmap", &HFAQEOutput::load_cold_mmap,
             py::arg("cold_bin_path"),
             "Memory-map cold coefficient file (§3.3, Linux/macOS)")

        // String repr
        .def("__repr__",
            [](const HFAQEOutput& self) {
                auto& c = self.config();
                auto  m = self.memory_report();
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "HFAQEOutput(V=%d, d=%d, r=%d, K=%d, B=%d, "
                    "mem=%.1f MB)",
                    c.V, c.d, c.r, c.K, c.B,
                    static_cast<double>(m.total_bytes)/(1024.0*1024.0));
                return std::string(buf);
            });
}

#endif // HFAQE_WITH_PYBIND11


// =============================================================================
// output_demo — end-to-end demonstration of the public API
// Exercises every public method to confirm the wiring is correct.
// =============================================================================
static void output_demo() {
    std::printf("\n=== Output.cpp: Public API Demo ===\n");

    // ---- 1. Build model via from_config ----
    auto api = HFAQEOutput::from_config(
        /*V=*/16000, /*d=*/128, /*r=*/32, /*K=*/1024, /*B=*/64,
        /*seed=*/42, /*tokenizer_path=*/"");

    std::printf("[1] Model created: V=%d d=%d r=%d K=%d\n",
        api->config().V, api->config().d,
        api->config().r, api->config().K);

    // ---- 2. Memory report ----
    auto mem = api->memory_report();
    double total_mb = static_cast<double>(mem.total_bytes) / (1024.0*1024.0);
    std::printf("[2] HFAQE memory: %.2f MB  (baseline BF16: %.2f MB)\n",
        total_mb,
        static_cast<double>(api->config().V)
        * api->config().d * 2.0 / (1024.0*1024.0));

    // ---- 3. embed_tokens: hot path ----
    std::vector<int> hot_ids;
    for (int s = 0; s < 5; ++s)
        hot_ids.push_back(api->model().hot.global_ids[s]);
    auto emb_hot = api->embed_tokens(hot_ids);
    std::printf("[3] embed_tokens (hot)  shape=%s  has_nan=%d\n",
        emb_hot.shape_str().c_str(), (int)emb_hot.has_nan);

    // ---- 4. embed_tokens: cold path ----
    std::vector<int> cold_ids;
    for (int s = 0; s < 5; ++s)
        cold_ids.push_back(api->model().cold.global_ids[s]);
    auto emb_cold = api->embed_tokens(cold_ids);
    std::printf("[4] embed_tokens (cold) shape=%s  has_nan=%d\n",
        emb_cold.shape_str().c_str(), (int)emb_cold.has_nan);

    // ---- 5. embed_text (stub tokenizer) ----
    auto emb_text = api->embed_text("ሰላም ዓለም");
    std::printf("[5] embed_text          shape=%s  has_nan=%d\n",
        emb_text.shape_str().c_str(), (int)emb_text.has_nan);

    // ---- 6. lm_head_logits ----
    std::vector<fp32> h(api->config().d, 0.1f);
    auto logit_res = api->lm_head_logits(h);
    auto topk = logit_res.top_k(5);
    std::printf("[6] lm_head argmax=%d  max_logit=%.4f  has_nan=%d\n",
        logit_res.argmax, logit_res.max_logit, (int)logit_res.has_nan);
    std::printf("    top-5 token IDs: ");
    for (int id : topk) std::printf("%d ", id);
    std::printf("\n");

    // ---- 7. Weight tying (§2.4 / §5.4) ----
    api->set_tied_lm_head();
    std::printf("[7] Weight tied: %s  basis_ptr=%p\n",
        api->is_tied() ? "YES" : "NO",
        static_cast<const void*>(api->basis_ptr()));

    // ---- 8. §7 ARC: dynamic vocabulary expansion ----
    int old_V  = api->config().V;
    int new_id = old_V;  // next token ID
    std::vector<fp32> init_vec(api->config().d, 0.05f);
    api->add_cold_token(new_id, init_vec);
    std::printf("[8] ARC expand: V %d → %d  new cold_id=%d\n",
        old_V, api->config().V, new_id);

    // Verify new token can be embedded
    auto emb_new = api->embed_tokens({new_id});
    bool ok = !emb_new.has_nan && emb_new.n == 1;
    std::printf("    New token embedded OK: %s\n", ok ? "YES" : "NO");

    // ---- 9. Raw pointer API (zero-copy for C++ downstream stages) ----
    std::vector<int>  ids_raw = {0, 100, 200};
    std::vector<fp32> out_raw(static_cast<size_t>(3) * api->config().d);
    api->embed_tokens(ids_raw.data(), 3, out_raw.data());
    bool finite_raw = true;
    for (fp32 v : out_raw)
        if (!std::isfinite(v)) { finite_raw = false; break; }
    std::printf("[9] Raw pointer API: all finite=%s\n",
        finite_raw ? "YES" : "NO");

    std::printf("=== Output.cpp Demo Complete ===\n\n");
}

// =============================================================================
// main — runs all demos in sequence when Output.cpp is compiled standalone
//        (define HFAQE_OUTPUT_MAIN to activate)
// =============================================================================
#ifdef HFAQE_OUTPUT_MAIN
int main() {
    // Input pipeline demo
    input_demo();

    // Training demo
    train_demo();

    // Output / public API demo
    output_demo();

    // Metrics benchmark
    run_metrics(/*V=*/16000, /*d=*/128, /*r=*/32, /*K=*/1024, /*B=*/64);

    return 0;
}
#endif // HFAQE_OUTPUT_MAIN


#endif // HFAQE_OUTPUT_CPP
