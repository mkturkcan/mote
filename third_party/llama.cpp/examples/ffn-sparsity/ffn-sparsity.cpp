#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include <clocale>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <map>
#include <cstdio>

// Accumulators: measure FFN intermediate (ffn_geglu) sparsity.
// A near-zero ffn_geglu[i] => ffn_down column i contributes ~nothing => skippable (lossless to precision).
struct Acc {
    long long total = 0;
    long long below[5] = {0,0,0,0,0}; // count below rel-threshold of per-position max
    double tail_energy[5] = {0,0,0,0,0}; // sum of squares of skipped (below-thr) neurons
    double total_energy = 0;             // sum of squares of all neurons
    double thr[5] = {1e-1, 3e-2, 1e-2, 3e-3, 1e-3};
};
static Acc g_acc;     // ffn_geglu (post gate*up) -> skippable down columns
static Acc g_gate;    // gelu(gate) only -> skippable up+down (read gate alone)
static inline float gelu_tanh(float x){ // gemma uses gelu_pytorch_tanh
    const float k=0.7978845608028654f; return 0.5f*x*(1.0f+tanhf(k*(x+0.044715f*x*x*x))); }

static bool cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) return true; // want data for everything; we filter below
    // APPEND-mode dump of real layer-20 FFN activations across ALL ubatches (calibration set)
    if (t->name && std::strcmp(t->name,"ffn_norm-20")==0 && t->type==GGML_TYPE_F32 && ggml_backend_buffer_is_host(t->buffer)) {
        FILE*fp=fopen("/tmp/x20.f32","ab"); if(fp){ fwrite(t->data,1,ggml_nbytes(t),fp); fclose(fp); }
    }
    if (t->name && std::strcmp(t->name,"ffn_geglu-20")==0 && t->type==GGML_TYPE_F32 && ggml_backend_buffer_is_host(t->buffer)) {
        FILE*fp=fopen("/tmp/geglu20.f32","ab"); if(fp){ fwrite(t->data,1,ggml_nbytes(t),fp); fclose(fp); }
    }
    if (t->name == nullptr) return true;
    bool is_gate = (std::strncmp(t->name, "ffn_gate", 8) == 0);
    bool is_geglu = (std::strncmp(t->name, "ffn_geglu", 9) == 0);
    if (!is_gate && !is_geglu) return true;
    if (t->type != GGML_TYPE_F32) return true;
    if (!ggml_backend_buffer_is_host(t->buffer)) return true;
    const float * d = (const float *) t->data;
    const int64_t ncol = t->ne[0];   // 6144 (intermediate dim)
    const int64_t ntok = t->ne[1];   // tokens in batch
    for (int64_t j = 0; j < ntok; ++j) {
        const float * v = d + j*ncol;
        Acc & A = is_gate ? g_gate : g_acc;
        float mx = 0.f;
        for (int64_t i = 0; i < ncol; ++i) { float a = std::fabs(v[i]); if (a>mx) mx=a; }
        if (mx <= 0.f) continue;
        for (int64_t i = 0; i < ncol; ++i) {
            float val = is_gate ? gelu_tanh(v[i]) : v[i];
            float a = std::fabs(val) / mx;
            double e = (double)val*(double)val;
            A.total++;
            A.total_energy += e;
            for (int k=0;k<5;k++) if (a < A.thr[k]) { A.below[k]++; A.tail_energy[k]+=e; }
        }
    }
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) { LOG_ERR("no tokens\n"); return false; }
    LOG_INF("n_input_tokens = %zu\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) { LOG_ERR("decode failed\n"); return false; }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    if (!run(ctx, params)) return 1;
    auto report=[&](const char*nm, Acc&A){
        LOG("\n===== %s  (thr rel to per-position max) =====\n", nm);
        LOG("  thr(rel)   %%skipped   %%L2-energy-lost\n");
        for (int k=0;k<5;k++) LOG("  %6.1e   %6.2f%%      %6.3f%%\n", A.thr[k],
            100.0*A.below[k]/(double)A.total, 100.0*A.tail_energy[k]/A.total_energy);
    };
    report("geglu -> skip ffn_down cols", g_acc);
    report("gelu(gate) -> skip ffn_up+down (read gate only)", g_gate);
    if(false){
    LOG("total intermediate activations sampled: %lld\n", g_acc.total);
    LOG("  thr(rel)   %%neurons-skipped   %%L2-energy-lost(=~output error bound)\n");
    for (int k=0;k<5;k++)
        LOG("  %6.1e      %6.2f%%            %6.3f%%\n",
            g_acc.thr[k], 100.0*g_acc.below[k]/(double)g_acc.total,
            100.0*g_acc.tail_energy[k]/g_acc.total_energy);
    }
    llama_backend_free();
    return 0;
}
