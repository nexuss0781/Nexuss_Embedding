// =============================================================================
// Embedding_test.cpp — Empirical Semantic Capability Test for HFAQE
// =============================================================================
#include "Storage.cpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Helper to get mean-pooled embedding for a string (Byte-level aggregation)
std::vector<fp32> get_string_embedding(const HFAQE& model, const std::string& text) {
    int d = model.cfg.d;
    if (text.empty()) return std::vector<fp32>(d, 0.0f);

    std::vector<int> tokens;
    for (unsigned char c : text) tokens.push_back(static_cast<int>(c));
    
    std::vector<fp32> byte_embs = model.forward(tokens);
    std::vector<fp32> pooled(d, 0.0f);
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        for (int j = 0; j < d; ++j) {
            pooled[j] += byte_embs[i * d + j];
        }
    }
    
    for (int j = 0; j < d; ++j) pooled[j] /= tokens.size();
    return pooled;
}

// Cosine similarity
double cosine_sim(const std::vector<fp32>& a, const std::vector<fp32>& b) {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm_a += static_cast<double>(a[i]) * a[i];
        norm_b += static_cast<double>(b[i]) * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void run_semantic_test(const HFAQE& model, const std::string& label, const std::string& s1, const std::string& s2) {
    auto e1 = get_string_embedding(model, s1);
    auto e2 = get_string_embedding(model, s2);
    double sim = cosine_sim(e1, e2);
    std::cout << std::left << std::setw(25) << label 
              << "| " << std::setw(15) << ("\"" + s1 + "\"") 
              << " vs " << std::setw(15) << ("\"" + s2 + "\"") 
              << " | Similarity: " << std::fixed << std::setprecision(4) << sim << std::endl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  HFAQE Empirical Semantic Capability Test                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

    const std::string ckpt_path = "checkpoints/hfaqe_best.nex";
    std::cout << "[load] Loading checkpoint: " << ckpt_path << "..." << std::endl;

    HFAQE model(HFAQEConfig{}); // Default config, will be overwritten by load
    try {
        NexCheckpointMeta meta;
        model = CheckpointManager::load_fresh(ckpt_path, &meta);
        std::cout << "[load] Success. V=" << model.cfg.V << " d=" << model.cfg.d << " Step=" << meta.global_step << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to load checkpoint: " << e.what() << std::endl;
        return 1;
    }

    std::cout << std::endl << "--- Semantic Grouping Tests ---" << std::endl;
    run_semantic_test(model, "Lowercase Letters", "apple", "banana");
    run_semantic_test(model, "Uppercase Letters", "APPLE", "BANANA");
    run_semantic_test(model, "Digits", "12345", "67890");
    run_semantic_test(model, "Mixed Alpha", "Apple", "Banana");

    std::cout << std::endl << "--- Cross-Category Tests (Should be Low) ---" << std::endl;
    run_semantic_test(model, "Letters vs Digits", "abcde", "12345");
    run_semantic_test(model, "Letters vs Symbols", "hello", "!@#$%");
    run_semantic_test(model, "Case Mismatch", "apple", "APPLE");

    std::cout << std::endl << "--- Structural/Morphological Similarity ---" << std::endl;
    run_semantic_test(model, "Prefix Overlap", "transport", "transplant");
    run_semantic_test(model, "Suffix Overlap", "walking", "talking");
    run_semantic_test(model, "Totally Different", "elephant", "xyzzy");

    std::cout << std::endl << "--- Logic/Content Probes ---" << std::endl;
    run_semantic_test(model, "Synonyms (Approx)", "cat", "kitten");
    run_semantic_test(model, "Numbers", "one", "two");

    std::cout << std::endl << "Test Completed." << std::endl;

    return 0;
}
