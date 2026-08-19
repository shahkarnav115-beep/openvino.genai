// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "extract_cgraph.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace llama_adapter {

void LlamaModelDeleter::operator()(llama_model* model) const {
#ifdef RUN_WITH_LLAMA
    if (model) {
        llama_free_model(model);
    }
#endif
}

void LlamaContextDeleter::operator()(llama_context* ctx) const {
#ifdef RUN_WITH_LLAMA
    if (ctx) {
        llama_free(ctx);
    }
#endif
}

ExtractedCGraph extract_cgraph(const std::filesystem::path& model_path) {
#ifdef RUN_WITH_LLAMA
    OPENVINO_ASSERT(std::filesystem::exists(model_path), "GGUF model file does not exist at: ", model_path.string());

    llama_backend_init();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    static ggml_backend_dev_t cpu_devices[] = { cpu_dev, nullptr };

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = false;
    mparams.n_gpu_layers = 0; // CPU graph extraction; disable GPU offloading
    mparams.no_alloc = false;    // Keep false for mmap tensor metadata loading
    if (cpu_dev) {
        mparams.devices = cpu_devices;
    }

    llama_model* raw_model = llama_model_load_from_file(model_path.string().c_str(), mparams);
    OPENVINO_ASSERT(raw_model != nullptr, "Failed to load GGUF model via llama.cpp from path: ", model_path.string());

    std::unique_ptr<llama_model, LlamaModelDeleter> model(raw_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;

    llama_context* raw_ctx = llama_new_context_with_model(model.get(), cparams);
    OPENVINO_ASSERT(raw_ctx != nullptr, "Failed to initialize llama_context for model: ", model_path.string());

    std::unique_ptr<llama_context, LlamaContextDeleter> ctx(raw_ctx);

    const uint32_t n_seqs = llama_n_seq_max(ctx.get());
    const uint32_t n_tokens = std::min(llama_n_ctx(ctx.get()), llama_n_ubatch(ctx.get()));

    struct ggml_cgraph* gf = llama_graph_reserve(ctx.get(), n_tokens, n_seqs, n_tokens);
    OPENVINO_ASSERT(gf != nullptr, "Failed to reserve compute graph from llama_context");

    ExtractedCGraph result;
    result.model = std::move(model);
    result.ctx = std::move(ctx);
    result.cgraph = gf;

    return result;
#else
    OPENVINO_THROW("llama.cpp adapter support is disabled. Please rebuild with CMake option '-DRUN_WITH_LLAMA=ON'");
#endif
}

}  // namespace llama_adapter
}  // namespace genai
}  // namespace ov
