// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#ifdef RUN_WITH_LLAMA
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-ext.h"
#else
struct llama_model;
struct llama_context;
struct ggml_cgraph;
#endif

namespace ov {
namespace genai {
namespace llama_adapter {

struct LlamaModelDeleter {
    void operator()(llama_model* model) const;
};

struct LlamaContextDeleter {
    void operator()(llama_context* ctx) const;
};

struct ExtractedCGraph {
    std::unique_ptr<llama_model, LlamaModelDeleter> model;
    std::unique_ptr<llama_context, LlamaContextDeleter> ctx;
    struct ggml_cgraph* cgraph = nullptr;
};

ExtractedCGraph extract_cgraph(const std::filesystem::path& model_path);

}  // namespace llama_adapter
}  // namespace genai
}  // namespace ov
