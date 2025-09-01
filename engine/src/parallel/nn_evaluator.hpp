#pragma once

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <queue>
#include <thread>
#include <string>
#include <utility>
#include <vector>
#include <functional>
#include <algorithm>
#include <array>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <NvInfer.h>

#include "tokenizer.hpp"
#include <cppcoro/task.hpp>
#include "options.hpp"
#include "concurrent_position_queue.hpp"
#include "chess.hpp"

namespace engine_parallel {

struct EvalResult {
    float value = 0.0f;
    // Optional tensors returned by the network for one sample
    // Flat vectors in row-major order per sample
    std::vector<float> hl;       // e.g., logits over buckets
    std::vector<float> policy;   // e.g., move policy logits or probs
    std::vector<float> U;        // uncertainty head
    std::vector<float> Q;        // value head per-move or auxiliary
    bool canceled = false;
};

// Simplified callback-based request interface; no coroutine handles involved

class NNEvaluator {
public:
    static constexpr std::size_t kBatchSize = 25;

    NNEvaluator() = default;

    explicit NNEvaluator(const engine::Options& options) : options_(&options) {}

    void start() {
        stop.store(false, std::memory_order_release);
        initialize_trt("./p2.plan");
        worker = std::jthread([this](std::stop_token st) { run(st); });
    }

    void stop_and_join() {
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        if (worker.joinable()) worker.join();
        release_trt();
    }

    void enqueue(const std::array<std::uint8_t, 68>& tokens,
                 std::function<void(EvalResult)> on_ready) {
        if (stop.load(std::memory_order_acquire)) {
            EvalResult er; er.value = 0.0f; er.canceled = true; on_ready(std::move(er));
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(Request{tokens, std::move(on_ready)});
        }
        cv.notify_one();
    }

    // Optional: provide a prefetch queue and sink so we can fill underfilled batches
    void set_prefetch_queue(engine_parallel::ConcurrentPositionQueue* q) { prefetch_queue_ = q; }
    void set_prefetch_sink(std::function<void(engine_parallel::PositionTask, EvalResult)> sink) { prefetch_sink_ = std::move(sink); }
    void set_prefetch_min_batch_size(std::size_t n) { prefetch_min_batch_size_ = n; }
    void set_tt_query(std::function<bool(const chess::Board&)> tt_query) { tt_query_ = std::move(tt_query); }

private:
    std::mutex mutex;
    std::condition_variable cv;
    struct Request {
        std::array<std::uint8_t, 68> tokens;
        std::function<void(EvalResult)> on_ready;
    };
    std::queue<Request> queue;
    std::atomic<bool> stop{false};
    std::jthread worker;
    const engine::Options* options_{nullptr};

    // Prefetch configuration (not owning)
    engine_parallel::ConcurrentPositionQueue* prefetch_queue_{nullptr};
    std::function<void(engine_parallel::PositionTask, EvalResult)> prefetch_sink_{};
    std::function<bool(const chess::Board&)> tt_query_{};
    std::size_t prefetch_min_batch_size_{5};

    // TensorRT objects
    nvinfer1::IRuntime* trt_runtime{nullptr};
    nvinfer1::ICudaEngine* trt_engine{nullptr};
    nvinfer1::IExecutionContext* trt_context{nullptr};
    cudaStream_t trt_stream{};

    // CUDA Graph caching per batch size 1..kBatchSize
    struct BatchGraph {
        int batchSize{0};
        bool valid{false};
        // Graph handles
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t graphExec{nullptr};
        // Device buffers
        void* d_tokens{nullptr};
        void* d_value{nullptr};
        void* d_hl{nullptr};
        void* d_hard{nullptr};
        void* d_U{nullptr};
        void* d_Q{nullptr};
        // Pinned host buffers (fixed address for capture)
        void* h_tokens{nullptr}; // int32_t* or int64_t* depending on tokens_dtype
        void* h_value{nullptr};  // __half* or float*, matches dt_value
        void* h_hl{nullptr};     // __half* or float*
        void* h_hard{nullptr};   // __half* or float*
        void* h_U{nullptr};      // __half* or float*
        void* h_Q{nullptr};      // __half* or float*
        // Sizes and dtypes
        size_t n_value{0}; nvinfer1::DataType dt_value{nvinfer1::DataType::kFLOAT};
        size_t n_hl{0};    nvinfer1::DataType dt_hl{nvinfer1::DataType::kFLOAT};
        size_t n_hard{0};  nvinfer1::DataType dt_hard{nvinfer1::DataType::kFLOAT};
        size_t n_U{0};     nvinfer1::DataType dt_U{nvinfer1::DataType::kFLOAT};
        size_t n_Q{0};     nvinfer1::DataType dt_Q{nvinfer1::DataType::kFLOAT};
        size_t tokensCount{0}; // number of scalar token elements (B*68)
        size_t tokensBytes{0};
        size_t bytes_value{0};
        size_t bytes_hl{0};
        size_t bytes_hard{0};
        size_t bytes_U{0};
        size_t bytes_Q{0};
    };
    std::array<BatchGraph, kBatchSize + 1> graphs_{}; // index by batch size

    void release_one_graph(BatchGraph& g) {
        if (g.graphExec) { cudaGraphExecDestroy(g.graphExec); g.graphExec = nullptr; }
        if (g.graph) { cudaGraphDestroy(g.graph); g.graph = nullptr; }
        if (g.d_tokens) { cudaFree(g.d_tokens); g.d_tokens = nullptr; }
        if (g.d_value) { cudaFree(g.d_value); g.d_value = nullptr; }
        if (g.d_hl) { cudaFree(g.d_hl); g.d_hl = nullptr; }
        if (g.d_hard) { cudaFree(g.d_hard); g.d_hard = nullptr; }
        if (g.d_U) { cudaFree(g.d_U); g.d_U = nullptr; }
        if (g.d_Q) { cudaFree(g.d_Q); g.d_Q = nullptr; }
        if (g.h_tokens) { cudaFreeHost(g.h_tokens); g.h_tokens = nullptr; }
        if (g.h_value) { cudaFreeHost(g.h_value); g.h_value = nullptr; }
        if (g.h_hl) { cudaFreeHost(g.h_hl); g.h_hl = nullptr; }
        if (g.h_hard) { cudaFreeHost(g.h_hard); g.h_hard = nullptr; }
        if (g.h_U) { cudaFreeHost(g.h_U); g.h_U = nullptr; }
        if (g.h_Q) { cudaFreeHost(g.h_Q); g.h_Q = nullptr; }
        g.valid = false; g.batchSize = 0;
        g.n_value = g.n_hl = g.n_hard = g.n_U = g.n_Q = 0;
        g.tokensBytes = g.tokensCount = 0;
        g.bytes_value = g.bytes_hl = g.bytes_hard = g.bytes_U = g.bytes_Q = 0;
    }

    void release_graphs() {
        for (auto& g : graphs_) release_one_graph(g);
    }

    bool build_graph_for_batch(int B) {
        BatchGraph& G = graphs_[static_cast<size_t>(B)];
        release_one_graph(G);
        if (!trt_context) return false;

        // Configure input shape for this batch
        nvinfer1::Dims inputDims; inputDims.nbDims = 2; inputDims.d[0] = B; inputDims.d[1] = 68;
        if (!trt_context->setInputShape("tokens", inputDims)) {
            std::cerr << "[NNEvaluator] setInputShape failed for B=" << B << "\n";
            return false;
        }

        // Allocate tokens buffers
        G.batchSize = B;
        G.tokensCount = static_cast<size_t>(B) * 68;
        G.tokensBytes = G.tokensCount * (tokens_dtype == nvinfer1::DataType::kINT32 ? sizeof(int32_t) : sizeof(int64_t));
        if (cudaHostAlloc(&G.h_tokens, G.tokensBytes, cudaHostAllocPortable) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaHostAlloc h_tokens failed\n";
            return false;
        }
        if (cudaMalloc(&G.d_tokens, G.tokensBytes) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaMalloc d_tokens failed\n";
            return false;
        }
        trt_context->setTensorAddress("tokens", G.d_tokens);

        // Helper to prepare an output binding
        auto prep_out = [&](const char* name, int bindingIndex, void** d_ptr, void** h_ptr, size_t& n_elem, size_t& n_bytes, nvinfer1::DataType& dt) {
            if (bindingIndex < 0) return;
            nvinfer1::Dims d = trt_context->getTensorShape(name);
            n_elem = volume(d);
            dt = trt_engine->getTensorDataType(name);
            n_bytes = n_elem * elementSize(dt);
            if (n_bytes == 0) return;
            if (cudaMalloc(d_ptr, n_bytes) != cudaSuccess) {
                std::cerr << "[NNEvaluator] cudaMalloc output failed for " << name << "\n";
                return;
            }
            if (cudaHostAlloc(h_ptr, n_bytes, cudaHostAllocPortable) != cudaSuccess) {
                std::cerr << "[NNEvaluator] cudaHostAlloc output failed for " << name << "\n";
                return;
            }
            trt_context->setTensorAddress(name, *d_ptr);
        };

        // Outputs
        prep_out("value", binding_idx_value, &G.d_value, &G.h_value, G.n_value, G.bytes_value, G.dt_value);
        if (binding_idx_hl >= 0)    prep_out("hl", binding_idx_hl, &G.d_hl, &G.h_hl, G.n_hl, G.bytes_hl, G.dt_hl);
        if (binding_idx_hardest >= 0) prep_out("hardest_policy", binding_idx_hardest, &G.d_hard, &G.h_hard, G.n_hard, G.bytes_hard, G.dt_hard);
        if (binding_idx_U >= 0)     prep_out("U", binding_idx_U, &G.d_U, &G.h_U, G.n_U, G.bytes_U, G.dt_U);
        if (binding_idx_Q >= 0)     prep_out("Q", binding_idx_Q, &G.d_Q, &G.h_Q, G.n_Q, G.bytes_Q, G.dt_Q);

        // Begin capture
        if (cudaStreamBeginCapture(trt_stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
            std::cerr << "[NNEvaluator] cudaStreamBeginCapture failed\n";
            return false;
        }
        // H2D tokens copy from pinned host
        cudaMemcpyAsync(G.d_tokens, G.h_tokens, G.tokensBytes, cudaMemcpyHostToDevice, trt_stream);
        // Enqueue inference
        trt_context->setTensorAddress("tokens", G.d_tokens);
        if (G.d_value) trt_context->setTensorAddress("value", G.d_value);
        if (G.d_hl) trt_context->setTensorAddress("hl", G.d_hl);
        if (G.d_hard) trt_context->setTensorAddress("hardest_policy", G.d_hard);
        if (G.d_U) trt_context->setTensorAddress("U", G.d_U);
        if (G.d_Q) trt_context->setTensorAddress("Q", G.d_Q);
        trt_context->enqueueV3(trt_stream);
        // D2H copies into pinned host buffers
        if (G.d_value && G.h_value && G.bytes_value) cudaMemcpyAsync(G.h_value, G.d_value, G.bytes_value, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_hl && G.h_hl && G.bytes_hl) cudaMemcpyAsync(G.h_hl, G.d_hl, G.bytes_hl, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_hard && G.h_hard && G.bytes_hard) cudaMemcpyAsync(G.h_hard, G.d_hard, G.bytes_hard, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_U && G.h_U && G.bytes_U) cudaMemcpyAsync(G.h_U, G.d_U, G.bytes_U, cudaMemcpyDeviceToHost, trt_stream);
        if (G.d_Q && G.h_Q && G.bytes_Q) cudaMemcpyAsync(G.h_Q, G.d_Q, G.bytes_Q, cudaMemcpyDeviceToHost, trt_stream);

        if (cudaStreamEndCapture(trt_stream, &G.graph) != cudaSuccess || !G.graph) {
            std::cerr << "[NNEvaluator] cudaStreamEndCapture failed\n";
            release_one_graph(G);
            return false;
        }
        if (cudaGraphInstantiate(&G.graphExec, G.graph, 0) != cudaSuccess || !G.graphExec) {
            std::cerr << "[NNEvaluator] cudaGraphInstantiate failed\n";
            release_one_graph(G);
            return false;
        }

        G.valid = true;
        return true;
    }

    void initialize_graphs() {
        // Ensure stream and context available
        if (!trt_context || !trt_stream) return;
        for (int B = 1; B <= static_cast<int>(kBatchSize); ++B) {
            bool ok = build_graph_for_batch(B);
            if (!ok) {
                // Keep going; some batch sizes may be unsupported if outputs depend on B
                graphs_[static_cast<size_t>(B)].valid = false;
            }
        }
    }

    // Simple TensorRT logger
    class TrtLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TRT] " << msg << '\n';
            }
        }
    };
    TrtLogger trt_logger;

    int binding_idx_tokens{-1};
    int binding_idx_value{-1};
    int binding_idx_hl{-1};
    int binding_idx_hardest{-1};
    int binding_idx_U{-1};
    int binding_idx_Q{-1};

    nvinfer1::DataType tokens_dtype{nvinfer1::DataType::kINT32};

    static inline size_t elementSize(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kINT64: return 8;
            case nvinfer1::DataType::kBOOL: return 1;
            default: return 0;
        }
    }

    static inline size_t volume(const nvinfer1::Dims& d) {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i] < 0 ? 1 : d.d[i]);
        return v;
    }

    void release_trt() {
        release_graphs();
        if (trt_context) { delete trt_context; trt_context = nullptr; }
        if (trt_engine) { delete trt_engine; trt_engine = nullptr; }
        if (trt_runtime) { delete trt_runtime; trt_runtime = nullptr; }
        if (trt_stream) { cudaStreamDestroy(trt_stream); trt_stream = nullptr; }
    }

    void initialize_trt(const char* plan_path) {
        release_trt();
        // Read plan file
        FILE* f = fopen(plan_path, "rb");
        if (!f) {
            std::cerr << "[NNEvaluator] Could not open plan file: " << plan_path << "\n";
            return;
        }
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<unsigned char> blob(static_cast<size_t>(len));
        if (fread(blob.data(), 1, blob.size(), f) != blob.size()) {
            std::cerr << "[NNEvaluator] Failed to read plan file: " << plan_path << "\n";
            fclose(f);
            return;
        }
        fclose(f);

        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        if (!trt_runtime) {
            std::cerr << "[NNEvaluator] Failed to create TensorRT runtime\n";
            return;
        }
        trt_engine = trt_runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!trt_engine) {
            std::cerr << "[NNEvaluator] Failed to deserialize engine\n";
            release_trt();
            return;
        }
        trt_context = trt_engine->createExecutionContext();
        if (!trt_context) {
            std::cerr << "[NNEvaluator] Failed to create execution context\n";
            release_trt();
            return;
        }
        if (cudaStreamCreate(&trt_stream) != cudaSuccess) {
            std::cerr << "[NNEvaluator] Failed to create CUDA stream\n";
            release_trt();
            return;
        }

        // Resolve bindings by name
        int nb = trt_engine->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = trt_engine->getIOTensorName(i);
            auto mode = trt_engine->getTensorIOMode(name);
            std::string n{name ? name : ""};
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (n == "tokens") {
                    binding_idx_tokens = i;
                    tokens_dtype = trt_engine->getTensorDataType(name);
                }
            } else {
                if (n == "value") binding_idx_value = i;
                if (n == "hl") binding_idx_hl = i;
                if (n == "hardest_policy") binding_idx_hardest = i;
                if (n == "U") binding_idx_U = i;
                if (n == "Q") binding_idx_Q = i;
            }
        }
        if (binding_idx_tokens < 0 || binding_idx_value < 0) {
            std::cerr << "[NNEvaluator] Required tensors not found (tokens/value)\n";
            release_trt();
            return;
        }
        std::cout << "[NNEvaluator] TensorRT engine loaded from " << plan_path << "\n";

        // Initialize CUDA Graphs for batch sizes 1..kBatchSize
        initialize_graphs();
    }

    static inline char tokenIndexToChar(std::uint8_t idx) {
        static constexpr char LUT[26] = {
            '0','1','2','3','4','5','6','7','8','9',
            'p','b','n','r','c','k','q',
            'P','B','N','R','C','Q','K',
            'x','.'
        };
        return idx < 26 ? LUT[idx] : '.';
    }

    static inline std::string tokensToString(const std::array<std::uint8_t, 68>& tokens) {
        std::string s;
        s.reserve(68);
        for (std::uint8_t t : tokens) {
            s.push_back(tokenIndexToChar(t));
        }
        return s;
    }

    void run(std::stop_token) {
        for (;;) {
            std::vector<Request> batch;
            batch.reserve(kBatchSize);
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]{ return stop.load() || !queue.empty(); });
                if (stop.load() && queue.empty()) break;
                // Prefer odd batch sizes when N>=10 by keeping one item in the queue if N is even
                std::size_t available = queue.size();
                std::size_t to_take = std::min<std::size_t>(available, kBatchSize);
                if (to_take >= 10 && (to_take % 2 == 0)) {
                    to_take -= 1;
                }
                while (!queue.empty() && batch.size() < to_take) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop();
                }
            }

            // If batch is underfilled and we have a prefetch queue, top up from it
            if (!stop.load(std::memory_order_acquire)
                && prefetch_queue_
                && prefetch_min_batch_size_ > batch.size()
                && prefetch_sink_) {
                std::size_t can_top_up = std::min<std::size_t>(prefetch_min_batch_size_ - batch.size(), kBatchSize - batch.size());
                for (std::size_t i = 0; i < can_top_up; ++i) {
                    engine_parallel::PositionTask task;
                    if (!prefetch_queue_->try_pop(task)) break;
                    
                    // Check if position is already in transposition table
                    if (tt_query_ && tt_query_(task.board)) {
                        continue; // Skip this position, try next one
                    }
                    
                    auto tokens = engine_tokenizer::tokenizeBoard(task.board);
                    batch.push_back(Request{tokens, [this, t = std::move(task)](EvalResult er) mutable {
                        if (prefetch_sink_) prefetch_sink_(std::move(t), std::move(er));
                    }});
                }
            }

            if (!batch.empty()) {
                if (is_batch_verbose()) {
                    std::cout << "BATCHED " << batch.size() << " EVALS" << std::endl;
                }
            }

            if (stop.load(std::memory_order_acquire)) {
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                }
                continue;
            }

            // If TensorRT is initialized, run inference; otherwise fallback
            if (trt_context && binding_idx_tokens >= 0 && binding_idx_value >= 0) {
                const int B = static_cast<int>(batch.size());
                // Use CUDA Graph if available for this batch size
                if (B > 0 && B <= static_cast<int>(kBatchSize) && graphs_[static_cast<size_t>(B)].valid) {
                    BatchGraph& G = graphs_[static_cast<size_t>(B)];
                    // Pack tokens into pinned host input buffer
                    if (tokens_dtype == nvinfer1::DataType::kINT32) {
                        int32_t* ht = reinterpret_cast<int32_t*>(G.h_tokens);
                        for (int i = 0; i < B; ++i) {
                            const auto& arr = batch[static_cast<size_t>(i)].tokens;
                            for (int j = 0; j < 68; ++j) ht[i * 68 + j] = static_cast<int32_t>(arr[j]);
                        }
                    } else {
                        int64_t* ht = reinterpret_cast<int64_t*>(G.h_tokens);
                        for (int i = 0; i < B; ++i) {
                            const auto& arr = batch[static_cast<size_t>(i)].tokens;
                            for (int j = 0; j < 68; ++j) ht[i * 68 + j] = static_cast<int64_t>(arr[j]);
                        }
                    }

                    // Launch captured graph
                    cudaGraphLaunch(G.graphExec, trt_stream);
                    cudaStreamSynchronize(trt_stream);

                    // Read outputs from pinned host buffers and dispatch
                    auto pinned_to_float = [&](void* h_ptr, size_t n_elem, nvinfer1::DataType dt) -> std::vector<float> {
                        std::vector<float> out;
                        if (!h_ptr || n_elem == 0) return out;
                        out.resize(n_elem);
                        if (dt == nvinfer1::DataType::kHALF) {
                            const __half* p = reinterpret_cast<const __half*>(h_ptr);
                            for (size_t i = 0; i < n_elem; ++i) out[i] = __half2float(p[i]);
                        } else if (dt == nvinfer1::DataType::kFLOAT) {
                            const float* p = reinterpret_cast<const float*>(h_ptr);
                            std::copy(p, p + n_elem, out.begin());
                        } else {
                            // Unsupported dtype for outputs
                            out.assign(n_elem, 0.0f);
                        }
                        return out;
                    };

                    std::vector<float> host_values = pinned_to_float(G.h_value, G.n_value, G.dt_value);
                    std::vector<float> host_hl    = pinned_to_float(G.h_hl,    G.n_hl,    G.dt_hl);
                    std::vector<float> host_hard  = pinned_to_float(G.h_hard,  G.n_hard,  G.dt_hard);
                    std::vector<float> host_U     = pinned_to_float(G.h_U,     G.n_U,     G.dt_U);
                    std::vector<float> host_Q     = pinned_to_float(G.h_Q,     G.n_Q,     G.dt_Q);

                    auto per_or_zero = [&](size_t total) -> size_t { return (B > 0 && total % static_cast<size_t>(B) == 0) ? (total / static_cast<size_t>(B)) : 0; };
                    const size_t per_hl = per_or_zero(G.n_hl);
                    const size_t per_hard = per_or_zero(G.n_hard);
                    const size_t per_U = per_or_zero(G.n_U);
                    const size_t per_Q = per_or_zero(G.n_Q);

                    for (int i = 0; i < B; ++i) {
                        float score = (G.n_value >= static_cast<size_t>((i + 1))) ? host_values[static_cast<size_t>(i)] : 0.0f;
                        if (is_batch_verbose()) {
                            std::cout << tokensToString(batch[static_cast<size_t>(i)].tokens) << " => " << score << std::endl;
                        }
                        EvalResult er;
                        er.value = score;
                        er.canceled = false;
                        if (per_hl > 0) er.hl = std::vector<float>(host_hl.begin() + static_cast<std::ptrdiff_t>(i * per_hl), host_hl.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hl));
                        if (per_hard > 0) er.policy = std::vector<float>(host_hard.begin() + static_cast<std::ptrdiff_t>(i * per_hard), host_hard.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hard));
                        if (per_U > 0) er.U = std::vector<float>(host_U.begin() + static_cast<std::ptrdiff_t>(i * per_U), host_U.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_U));
                        if (per_Q > 0) er.Q = std::vector<float>(host_Q.begin() + static_cast<std::ptrdiff_t>(i * per_Q), host_Q.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_Q));
                        batch[static_cast<size_t>(i)].on_ready(std::move(er));
                    }
                } else {
                    // Fallback: dynamic path as before
                    // Set input shape [B, 68]
                    nvinfer1::Dims inputDims; inputDims.nbDims = 2; inputDims.d[0] = B; inputDims.d[1] = 68;
                    if (!trt_context->setInputShape("tokens", inputDims)) {
                        std::cerr << "[NNEvaluator] Failed to set input shape" << std::endl;
                    }

                    // Host staging for tokens cast
                    size_t tokensCount = static_cast<size_t>(B) * 68;
                    std::vector<int32_t> tokens_i32;
                    std::vector<int64_t> tokens_i64;
                    void* d_tokens = nullptr;
                    size_t tokensBytes = 0;
                    if (tokens_dtype == nvinfer1::DataType::kINT32) {
                        tokens_i32.resize(tokensCount);
                        for (size_t i = 0; i < batch.size(); ++i) {
                            const auto& arr = batch[i].tokens;
                            for (int j = 0; j < 68; ++j) tokens_i32[i * 68 + j] = static_cast<int32_t>(arr[j]);
                        }
                        tokensBytes = tokens_i32.size() * sizeof(int32_t);
                        cudaMalloc(&d_tokens, tokensBytes);
                        cudaMemcpyAsync(d_tokens, tokens_i32.data(), tokensBytes, cudaMemcpyHostToDevice, trt_stream);
                    } else {
                        tokens_i64.resize(tokensCount);
                        for (size_t i = 0; i < batch.size(); ++i) {
                            const auto& arr = batch[i].tokens;
                            for (int j = 0; j < 68; ++j) tokens_i64[i * 68 + j] = static_cast<int64_t>(arr[j]);
                        }
                        tokensBytes = tokens_i64.size() * sizeof(int64_t);
                        cudaMalloc(&d_tokens, tokensBytes);
                        cudaMemcpyAsync(d_tokens, tokens_i64.data(), tokensBytes, cudaMemcpyHostToDevice, trt_stream);
                    }
                    trt_context->setTensorAddress("tokens", d_tokens);

                    // Prepare outputs; we need at least 'value', but provide all if present
                    auto prepare_output = [&](const char* name, int bindingIndex, void** d_ptr, nvinfer1::DataType& dtype_out, size_t& elems) {
                        if (bindingIndex < 0) return;
                        nvinfer1::Dims d = trt_context->getTensorShape(name);
                        elems = volume(d);
                        dtype_out = trt_engine->getTensorDataType(name);
                        size_t bytes = elems * elementSize(dtype_out);
                        cudaMalloc(d_ptr, bytes);
                        trt_context->setTensorAddress(name, *d_ptr);
                    };

                    void* d_value = nullptr; size_t n_value = 0; nvinfer1::DataType dt_value = nvinfer1::DataType::kFLOAT;
                    void* d_hl = nullptr; size_t n_hl = 0; nvinfer1::DataType dt_hl = nvinfer1::DataType::kFLOAT;
                    void* d_hard = nullptr; size_t n_hard = 0; nvinfer1::DataType dt_hard = nvinfer1::DataType::kFLOAT;
                    void* d_U = nullptr; size_t n_U = 0; nvinfer1::DataType dt_U = nvinfer1::DataType::kFLOAT;
                    void* d_Q = nullptr; size_t n_Q = 0; nvinfer1::DataType dt_Q = nvinfer1::DataType::kFLOAT;

                    prepare_output("value", binding_idx_value, &d_value, dt_value, n_value);
                    if (binding_idx_hl >= 0) prepare_output("hl", binding_idx_hl, &d_hl, dt_hl, n_hl);
                    if (binding_idx_hardest >= 0) prepare_output("hardest_policy", binding_idx_hardest, &d_hard, dt_hard, n_hard);
                    if (binding_idx_U >= 0) prepare_output("U", binding_idx_U, &d_U, dt_U, n_U);
                    if (binding_idx_Q >= 0) prepare_output("Q", binding_idx_Q, &d_Q, dt_Q, n_Q);

                    // Execute
                    if (!trt_context->enqueueV3(trt_stream)) {
                        std::cerr << "[NNEvaluator] enqueueV3 failed" << std::endl;
                        // Fail the batch safely
                        for (auto& r : batch) {
                            EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                        }
                        // Skip device copies and continue loop
                        continue;
                    }

                    // Copy back outputs and distribute results
                    std::vector<float> host_values(n_value);
                    if (dt_value == nvinfer1::DataType::kHALF) {
                        std::vector<__half> tmp(n_value);
                        cudaMemcpyAsync(tmp.data(), d_value, n_value * sizeof(__half), cudaMemcpyDeviceToHost, trt_stream);
                        cudaStreamSynchronize(trt_stream);
                        for (size_t i = 0; i < n_value; ++i) host_values[i] = __half2float(tmp[i]);
                    } else {
                        cudaMemcpyAsync(host_values.data(), d_value, n_value * sizeof(float), cudaMemcpyDeviceToHost, trt_stream);
                    }
                    cudaStreamSynchronize(trt_stream);

                    auto copy_out_float = [&](void* d_ptr, size_t n_elem, nvinfer1::DataType dt) -> std::vector<float> {
                        std::vector<float> host;
                        if (!d_ptr || n_elem == 0) return host;
                        host.resize(n_elem);
                        if (dt == nvinfer1::DataType::kHALF) {
                            std::vector<__half> tmp(n_elem);
                            cudaMemcpyAsync(tmp.data(), d_ptr, n_elem * sizeof(__half), cudaMemcpyDeviceToHost, trt_stream);
                            cudaStreamSynchronize(trt_stream);
                            for (size_t i = 0; i < n_elem; ++i) host[i] = __half2float(tmp[i]);
                        } else {
                            cudaMemcpyAsync(host.data(), d_ptr, n_elem * sizeof(float), cudaMemcpyDeviceToHost, trt_stream);
                            cudaStreamSynchronize(trt_stream);
                        }
                        return host;
                    };

                    std::vector<float> host_hl = copy_out_float(d_hl, n_hl, dt_hl);
                    std::vector<float> host_hard = copy_out_float(d_hard, n_hard, dt_hard);
                    std::vector<float> host_U = copy_out_float(d_U, n_U, dt_U);
                    std::vector<float> host_Q = copy_out_float(d_Q, n_Q, dt_Q);

                    // Determine per-sample sizes (assume leading batch dim)
                    auto per_or_zero2 = [&](size_t total) -> size_t { return (B > 0 && total % static_cast<size_t>(B) == 0) ? (total / static_cast<size_t>(B)) : 0; };
                    const size_t per_hl = per_or_zero2(n_hl);
                    const size_t per_hard = per_or_zero2(n_hard);
                    const size_t per_U = per_or_zero2(n_U);
                    const size_t per_Q = per_or_zero2(n_Q);

                    for (int i = 0; i < B; ++i) {
                        float score = (n_value >= static_cast<size_t>((i + 1))) ? host_values[static_cast<size_t>(i)] : 0.0f;
                        if (is_batch_verbose()) {
                            std::cout << tokensToString(batch[static_cast<size_t>(i)].tokens) << " => " << score << std::endl;
                        }
                        EvalResult er;
                        er.value = score;
                        er.canceled = false;
                        if (per_hl > 0) er.hl = std::vector<float>(host_hl.begin() + static_cast<std::ptrdiff_t>(i * per_hl), host_hl.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hl));
                        if (per_hard > 0) er.policy = std::vector<float>(host_hard.begin() + static_cast<std::ptrdiff_t>(i * per_hard), host_hard.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_hard));
                        if (per_U > 0) er.U = std::vector<float>(host_U.begin() + static_cast<std::ptrdiff_t>(i * per_U), host_U.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_U));
                        if (per_Q > 0) er.Q = std::vector<float>(host_Q.begin() + static_cast<std::ptrdiff_t>(i * per_Q), host_Q.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_Q));
                        batch[static_cast<size_t>(i)].on_ready(std::move(er));
                    }

                    // Cleanup device buffers
                    if (d_tokens) cudaFree(d_tokens);
                    if (d_value) cudaFree(d_value);
                    if (d_hl) cudaFree(d_hl);
                    if (d_hard) cudaFree(d_hard);
                    if (d_U) cudaFree(d_U);
                    if (d_Q) cudaFree(d_Q);
                }
            } else {
                // No TensorRT available: cancel all requests and continue without throwing
                for (auto& r : batch) {
                    EvalResult er; er.value = 0.0f; er.canceled = true; r.on_ready(std::move(er));
                }
                continue;
            }
            // No explicit resumption here; callbacks have already notified waiters
        }
    }

    bool is_batch_verbose() const {
        if (!options_) return false;
        const std::string v = options_->get("batchverbose", "");
        return !v.empty() && v != "0" && v != "false" && v != "False";
    }
};

} // namespace engine_parallel


