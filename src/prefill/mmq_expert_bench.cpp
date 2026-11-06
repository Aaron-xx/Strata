// src/prefill/mmq_expert_bench.cpp - the prompt path's routed-expert products (MMQ) timed at the engine's shapes.
//
//     [STRATA_PREFILL_NVFP4=w4a8|w4a4] mmq_expert_bench [rows per expert ...]
//
// One MMQ group as the prompt path launches it: 16 experts, R rows each, in one activation array in expert order.
// gate/up [1280 x 2560] and down [2560 x 640] in NVFP4, down in Q8_0 (the GPTQ + Q8_0-down pack's 27 layers), and
// UD-IQ4_XS's IQ3_S gate/up and IQ4_NL down for comparison.
// Synthetic weights of the right format (valid scales), random activations of row-varying magnitude.  Per product:
// the median of 25 runs, and the rate in int8-equivalent TOPS (2 x rows x out x in).  Default R: 12 (a 600-token
// chunk: 600 x 10 / 512), 160 (8K), 626 (32K).
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// valid synthetic weights: NVFP4 blocks are {e4m3 scales, e2m1 codes}; Q8_0 {f16 d, int8 q[32]}.  Random bytes with
// the scale fields overwritten by small positive values (an e4m3 / f16 in the model's range).
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(t), nblk = (size_t) experts * per / bs;
    for (size_t i = 0; i < nblk; ++i) {
        uint8_t* blk = w.data() + i * bs;
        if (t != GGML_TYPE_NVFP4) {                                   // Q8_0, IQ3_S, IQ4_NL: an f16 d leads the block
            const uint16_t d = 0x2000 + (uint16_t) byte(rng);         // f16 ~ 0.0078..0.016
            std::memcpy(blk, &d, 2);
        } else {                                                      // NVFP4: the scales lead the block
            const int nsc = (int) (bs - (size_t) ggml_blck_size(t) / 2);
            for (int s = 0; s < nsc; ++s) blk[s] = (uint8_t) (0x30 + byte(rng) % 16);   // e4m3 ~ 0.5..1
        }
    }
    return w;
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t rows, cols;
};

// STRATA_BENCH_E: how many experts one launch covers (the engine launches all 256 of a layer).
// STRATA_BENCH_TOTAL: when set, the T rows are spread over the E experts at random (a multinomial, the engine's
// routing), and R becomes only the largest count any expert sees.  Without it every expert gets exactly R.
double run_case(const Case& c, int R, std::mt19937& rng, float* ms_out) {
    static const int E = [] { const char* v = std::getenv("STRATA_BENCH_E"); return v && v[0] ? std::atoi(v) : 16; }();
    static const int total = [] { const char* v = std::getenv("STRATA_BENCH_TOTAL"); return v && v[0] ? std::atoi(v) : 0; }();
    const int T = total > 0 ? total : E * R;
    const size_t per = mmq::matrix_bytes(c.t, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.t, c.rows, c.cols, E, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < T; ++t) { const float m = mag(rng); for (int64_t k = 0; k < c.cols; ++k) x[(size_t) t * c.cols + k] = nd(rng) * m; }
    std::vector<int32_t> bounds(E + 1);
    std::vector<int> cntv(E, 0);   // rows each expert gets: the per-expert launch's max_rows
    // STRATA_BENCH_MAXR: the engine's real routing is far more skewed than a multinomial - one expert of a layer
    // can take a tenth of the chunk's entries, and the launch grid is laid for that expert's row count.  Give
    // expert 0 exactly this many rows and spread the rest evenly, so the grid's ntx is the engine's while the
    // total work stays the same: the A/B for "does the maxr grid cost anything".
    static const int forced = [] { const char* v = std::getenv("STRATA_BENCH_MAXR"); return v && v[0] ? std::atoi(v) : 0; }();
    if (forced > 0 && total > forced) {
        int acc = 0;
        bounds[0] = 0;
        for (int e = 0; e < E; ++e) {
            const int c = e == 0 ? forced : (total - forced) / (E - 1) + (e <= (total - forced) % (E - 1) ? 1 : 0);
            acc += c;
            bounds[e + 1] = acc;
        }
        R = forced;
        for (int e = 0; e < E; ++e) cntv[e] = bounds[e + 1] - bounds[e];
        std::printf("  [E=%d total=%d rows, FORCED maxr %d rows on expert 0]\n", E, T, R);
    } else if (total > 0) {
        std::vector<int> cnt(E, 0);
        std::uniform_int_distribution<int> pick(0, E - 1);
        for (int t = 0; t < T; ++t) cnt[pick(rng)]++;
        int acc = 0, mx = 0;
        bounds[0] = 0;
        for (int e = 0; e < E; ++e) { acc += cnt[e]; bounds[e + 1] = acc; if (cnt[e] > mx) mx = cnt[e]; }
        R = mx;
        cntv = cnt;
        std::printf("  [E=%d total=%d rows, busiest expert %d rows]\n", E, T, R);
    } else {
        for (int e = 0; e <= E; ++e) bounds[e] = e * R;
        cntv.assign(E, R);
    }

    uint8_t* dw; float *dx, *dd, *dys; void* dq; int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w"); ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&dys, (size_t) T * 4), "ys");
    ck(cudaMalloc(&dd, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, (E + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, c.t, c.cols, c.cols, T, nullptr, dys);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = R; p.dst = dd; p.ld_dst = c.rows;
    p.y_scale = mmq::fp4_activations(c.t) ? dys : nullptr;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    // STRATA_BENCH_SPLIT: one launch per expert (n=1, that expert's own row count as the grid's max_rows).
    // The wide launch pays for ceil(maxr/J) tile columns on every one of the E experts; this pays only
    // ceil(cnt_e/J) - at the price of E launches.
    static const int split = [] { const char* v = std::getenv("STRATA_BENCH_SPLIT"); return v && v[0] ? std::atoi(v) : 0; }();
    static const int grp = [] { const char* v = std::getenv("STRATA_BENCH_GRP"); return v && v[0] ? std::atoi(v) : 1; }();
    if (split) std::printf("  [split: %d launches a product, %d experts each]\n", (E + grp - 1) / grp, grp);
    std::vector<float> ts;
    for (int i = 0; i < 28; ++i) {
        cudaEventRecord(e0);
        if (split) {
            for (int e0 = 0; e0 < E; e0 += grp) {
                const int e1 = std::min(E, e0 + grp);
                int64_t mx = 0, tot = 0;
                for (int e = e0; e < e1; ++e) { if (cntv[e] > mx) mx = cntv[e]; tot += cntv[e]; }
                if (tot <= 0) continue;
                mmq::Product q = p;
                q.w = dw + (size_t) e0 * per;
                q.bounds = db + e0;
                q.n = e1 - e0;
                q.total_rows = tot;
                q.max_rows = mx;
                ctx.run(q, nullptr);
            }
        } else {
            ctx.run(p, nullptr);
        }
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "run");
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const float ms = ts[ts.size() / 2];
    // STRATA_BENCH_CHECK: the split launch must produce the same rows as the wide one.  Runs both once into
    // separate buffers and reports the largest absolute difference (and the largest value, for scale).
    static const int check = [] { const char* v = std::getenv("STRATA_BENCH_CHECK"); return v && v[0] ? std::atoi(v) : 0; }();
    if (check && split) {
        const size_t nd = (size_t) T * c.rows;
        float* da; float* db2;
        ck(cudaMalloc(&da, nd * 4), "da"); ck(cudaMalloc(&db2, nd * 4), "db");
        ck(cudaMemset(da, 0, nd * 4), "da0"); ck(cudaMemset(db2, 0, nd * 4), "db0");
        mmq::Product q0 = p; q0.dst = da;
        ctx.run(q0, nullptr);
        for (int e0 = 0; e0 < E; e0 += grp) {
            const int e1 = std::min(E, e0 + grp);
            int64_t mx = 0, tot = 0;
            for (int e = e0; e < e1; ++e) { if (cntv[e] > mx) mx = cntv[e]; tot += cntv[e]; }
            if (tot <= 0) continue;
            mmq::Product q = p; q.w = dw + (size_t) e0 * per; q.bounds = db + e0; q.n = e1 - e0;
            q.total_rows = tot; q.max_rows = mx; q.dst = db2;
            ctx.run(q, nullptr);
        }
        ck(cudaDeviceSynchronize(), "check");
        std::vector<float> ha(nd), hb(nd);
        ck(cudaMemcpy(ha.data(), da, nd * 4, cudaMemcpyDeviceToHost), "ha");
        ck(cudaMemcpy(hb.data(), db2, nd * 4, cudaMemcpyDeviceToHost), "hb");
        double mx = 0, scale = 0;
        for (size_t i = 0; i < nd; ++i) {
            const double d = std::fabs((double) ha[i] - (double) hb[i]);
            if (d > mx) mx = d;
            if (std::fabs((double) ha[i]) > scale) scale = std::fabs((double) ha[i]);
        }
        std::printf("  [check: wide vs split, max |diff| = %.6g, max |value| = %.6g]\n", mx, scale);
        cudaFree(da); cudaFree(db2);
    }
    if (ms_out) *ms_out = ms;
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dys); cudaFree(dd); cudaFree(dids); cudaFree(db);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 2.0 * T * (double) c.rows * c.cols / (ms * 1e9);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {12, 160, 626};
    const char* mode = std::getenv("STRATA_PREFILL_NVFP4");
    std::printf("mmq_expert_bench: mode %s (FP4 activations: %s), 16 experts a launch, medians of 25\n",
                mode ? mode : "w4a8", mmq::fp4_activations(GGML_TYPE_NVFP4) ? "yes" : "no");
    // the qwen3.6-35B-A3B UD-Q4_K_S shapes the engine actually launches (N = 2048, FF = 512)
    const Case cases[] = {{"q4_k gate/up", GGML_TYPE_Q4_K, 1024, 2048},
                          {"q4_k down", GGML_TYPE_Q4_K, 2048, 512},
                          {"nvfp4 gate/up", GGML_TYPE_NVFP4, 1280, 2560},
                          {"nvfp4 down", GGML_TYPE_NVFP4, 2560, 640},
                          {"q8_0 down", GGML_TYPE_Q8_0, 2560, 640},
                          {"iq3_s gate/up", GGML_TYPE_IQ3_S, 1280, 2560},
                          {"iq4_nl down", GGML_TYPE_IQ4_NL, 2560, 640}};
    std::mt19937 rng(7);
    for (int R : Rs)
        for (const Case& c : cases) {
            float ms = 0;
            const double tops = run_case(c, R, rng, &ms);
            std::printf("  R %4d  %-14s %8.3f ms  %7.1f TOPS\n", R, c.name, ms, tops);
        }
    return 0;
}
