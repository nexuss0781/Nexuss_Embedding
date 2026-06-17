// =============================================================================
// Input.cpp — Tokenizer → HFAQE Bridge
// =============================================================================
#ifndef HFAQE_INPUT_CPP
#define HFAQE_INPUT_CPP
//
// Tokenizer interface (from input/Tokenizer.md):
//   - Python class Tokenizer at ../Component-1.1_Tokenizer/tokenizer.py
//   - tok.encode(str)             → List[int]
//   - tok.encode_batch(List[str]) → List[List[int]]
//   - tok.vocab_size              → 16000 (V)
//   - Callable: tok(str|List[str])
//   - Truncation: tok.encode(str, truncation=True, max_length=N)
//
// This file provides:
//   1. TokenizerBridge — embeds the Python tokenizer via pybind11
//   2. encode_and_embed() — full pipeline: text → IDs → embeddings
//   3. A standalone C++ tokenizer adapter stub for compile-time testing
// =============================================================================

#include "Core.cpp"

// ---------------------------------------------------------------------------
// pybind11 integration (available when building the Python extension)
// ---------------------------------------------------------------------------
#ifdef HFAQE_WITH_PYBIND11
#  include <pybind11/embed.h>
#  include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>

// =============================================================================
// TokenizerBridge — wraps the Python EthioBBPE tokenizer
// Loaded once at program start, kept alive for the embedding session.
// =============================================================================
class TokenizerBridge {
public:
    int vocab_size = 0;

    // -----------------------------------------------------------------
    // Constructor: import the tokenizer module and instantiate.
    // Module path: "../Component-1.1_Tokenizer/tokenizer"  (relative import)
    // -----------------------------------------------------------------
    explicit TokenizerBridge(const std::string& module_path = "") {
#ifdef HFAQE_WITH_PYBIND11
        if (!py::interpreter_is_alive()) {
            throw std::runtime_error(
                "TokenizerBridge: Python interpreter not started. "
                "Call pybind11::initialize_interpreter() first.");
        }
        try {
            // Add the tokenizer's parent directory to sys.path
            py::module_ sys = py::module_::import("sys");
            std::string path = module_path.empty()
                ? "../Component-1.1_Tokenizer"
                : module_path;
            sys.attr("path").attr("insert")(0, path);

            // Import and instantiate
            py::module_ tok_mod = py::module_::import("tokenizer");
            tok_obj_ = tok_mod.attr("Tokenizer")();
            vocab_size = tok_obj_.attr("vocab_size").cast<int>();
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(
                std::string("TokenizerBridge: failed to load tokenizer: ") + e.what());
        }
#else
        // Stub: in a pure C++ build without pybind11, use the mock tokenizer
        vocab_size = 16000; // EthioBBPE default (Tokenizer.md §Scope)
        (void)module_path;
        std::fprintf(stderr,
            "[TokenizerBridge] pybind11 not available — running in stub mode\n");
#endif
    }

    // -----------------------------------------------------------------
    // encode: single string → token ID list
    // Wraps: tok.encode(text, truncation=truncate, max_length=max_len)
    // -----------------------------------------------------------------
    std::vector<int> encode(const std::string& text,
                            bool truncation = false,
                            int  max_length = 512) const {
#ifdef HFAQE_WITH_PYBIND11
        try {
            py::object result;
            if (truncation) {
                result = tok_obj_.attr("encode")(
                    text,
                    py::arg("truncation") = true,
                    py::arg("max_length") = max_length);
            } else {
                result = tok_obj_.attr("encode")(text);
            }
            return result.cast<std::vector<int>>();
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(
                std::string("TokenizerBridge::encode failed: ") + e.what());
        }
#else
        // Stub: deterministic hash-based encoding for offline testing
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char c : text) {
            int tok = static_cast<int>(c) % vocab_size;
            ids.push_back(tok);
            if (truncation && (int)ids.size() >= max_length) break;
        }
        return ids;
#endif
    }

    // -----------------------------------------------------------------
    // encode_batch: list of strings → list of token ID lists
    // Wraps: tok.encode_batch(texts)
    // -----------------------------------------------------------------
    std::vector<std::vector<int>> encode_batch(
        const std::vector<std::string>& texts) const
    {
#ifdef HFAQE_WITH_PYBIND11
        try {
            auto result = tok_obj_.attr("encode_batch")(texts);
            return result.cast<std::vector<std::vector<int>>>();
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(
                std::string("TokenizerBridge::encode_batch failed: ") + e.what());
        }
#else
        std::vector<std::vector<int>> out;
        out.reserve(texts.size());
        for (const auto& t : texts)
            out.push_back(encode(t));
        return out;
#endif
    }

    // Callable shorthand: tok(text) or tok(texts)
    std::vector<int> operator()(const std::string& text) const {
        return encode(text);
    }
    std::vector<std::vector<int>> operator()(
        const std::vector<std::string>& texts) const {
        return encode_batch(texts);
    }

private:
#ifdef HFAQE_WITH_PYBIND11
    mutable py::object tok_obj_;
#endif
};

// =============================================================================
// EmbedPipeline — combines TokenizerBridge with HFAQE for full text→embedding
// =============================================================================
class EmbedPipeline {
public:
    TokenizerBridge tokenizer;
    HFAQE*          embedder;  // owned externally (or shared with LM head)

    // -----------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------
    explicit EmbedPipeline(HFAQE* emb,
                           const std::string& tokenizer_module_path = "")
        : tokenizer(tokenizer_module_path), embedder(emb)
    {
        if (!embedder)
            throw std::invalid_argument("EmbedPipeline: embedder must not be null");
        // Validate vocab size consistency
        if (tokenizer.vocab_size > 0 && tokenizer.vocab_size != embedder->cfg.V) {
            std::fprintf(stderr,
                "[EmbedPipeline] WARNING: tokenizer.vocab_size=%d != embedder.V=%d\n",
                tokenizer.vocab_size, embedder->cfg.V);
        }
    }

    // -----------------------------------------------------------------
    // encode_and_embed: text → token IDs → embeddings  (fp32[n×d])
    // This is the primary integration function per the spec scope.
    // -----------------------------------------------------------------
    std::vector<fp32> encode_and_embed(const std::string& text,
                                       bool truncation = false,
                                       int  max_length = 512) const
    {
        std::vector<int> ids = tokenizer.encode(text, truncation, max_length);
        if (ids.empty()) return {};
        // Clamp IDs to [0, V) for safety before embedding
        for (auto& id : ids) {
            id = std::max(0, std::min(id, embedder->cfg.V - 1));
        }
        return embedder->forward(ids);
    }

    // -----------------------------------------------------------------
    // encode_and_embed_batch: batch of texts → list of embedding matrices
    // Each returned vector is flat [n_i × d] for the i-th sentence.
    // -----------------------------------------------------------------
    std::vector<std::vector<fp32>> encode_and_embed_batch(
        const std::vector<std::string>& texts,
        bool truncation = false,
        int  max_length = 512) const
    {
        auto batch_ids = tokenizer.encode_batch(texts);
        std::vector<std::vector<fp32>> out;
        out.reserve(texts.size());
        for (auto& ids : batch_ids) {
            if (ids.empty()) { out.emplace_back(); continue; }
            for (auto& id : ids)
                id = std::max(0, std::min(id, embedder->cfg.V - 1));
            out.push_back(embedder->forward(ids));
        }
        return out;
    }

    // -----------------------------------------------------------------
    // get_token_ids: utility — just tokenize, return IDs
    // -----------------------------------------------------------------
    std::vector<int> get_token_ids(const std::string& text,
                                   bool truncation = false,
                                   int  max_length = 512) const
    {
        return tokenizer.encode(text, truncation, max_length);
    }

    // -----------------------------------------------------------------
    // vocab_size: forwarded from tokenizer
    // -----------------------------------------------------------------
    int vocab_size() const { return tokenizer.vocab_size; }
};

// =============================================================================
// input_demo: demonstrates the full tokenizer→embedding pipeline
// =============================================================================
static void input_demo() {
    std::printf("\n=== Input.cpp: Tokenizer → Embedding Demo ===\n");

    // Build a small HFAQE model matching EthioBBPE vocab
    HFAQEConfig cfg;
    cfg.V = 16000;   // EthioBBPE vocabulary size (Tokenizer.md)
    cfg.d = 128;
    cfg.r = 32;
    cfg.K = 1024;
    cfg.B = 64;

    HFAQE model(cfg);
    auto freq = zipf_frequencies(cfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);

    EmbedPipeline pipeline(&model);

    // Single text (Amharic: "hello world")
    std::string text = "\xe1\x88\xb0\xe1\x88\xb3\xe1\x88\x9d \xe1\x8d\x93\xe1\x88\xb0\x6d";
    auto ids = pipeline.get_token_ids(text);
    std::printf("  Encoded '%s' → %zu tokens\n", text.c_str(), ids.size());

    auto emb = pipeline.encode_and_embed(text);
    std::printf("  Embedding shape: [%zu, %d]\n", ids.size(), cfg.d);

    bool finite = true;
    for (fp32 v : emb) if (!std::isfinite(v)) { finite = false; break; }
    std::printf("  All values finite: %s\n", finite ? "YES" : "NO");

    // Batch demo
    std::vector<std::string> batch = {"ሰላም", "ዓለም", "ሐዋርያ"};
    auto batch_embs = pipeline.encode_and_embed_batch(batch);
    std::printf("  Batch size: %zu sentences embedded\n", batch_embs.size());
    for (size_t i = 0; i < batch_embs.size(); ++i)
        std::printf("    [%zu] → %zu elements\n", i, batch_embs[i].size());

    std::printf("=== Input.cpp Demo Complete ===\n\n");
}


#endif // HFAQE_INPUT_CPP
