// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_reader_v2.hpp"
#include "orchestrator.hpp"

#include <memory>

#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"

#ifdef ENABLE_OV_GGUF_FRONTEND
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#endif

#include "utils.hpp"

namespace ov {
namespace genai {

std::shared_ptr<ov::Model> GgufReaderV2(
    const std::filesystem::path& model_path,
    bool enable_save_ov_model
) {
#ifdef RUN_WITH_LLAMA
    std::shared_ptr<ov::Model> model = llama_adapter::get_stateful_ov_model(model_path, /*for_genai_pipeline=*/true);

    // 2. Apply AdaptToGenAI pass to rewrite GGUF-native inputs/outputs into GenAI pipeline contract:
    // (input_ids [b, seq], attention_mask [b, kv_len], position_ids [b, seq], beam_idx [b] -> logits [b, seq, vocab])
#ifdef ENABLE_OV_GGUF_FRONTEND
    ov::pass::Manager pm;
    pm.register_pass<ov::frontend::gguf::pass::AdaptToGenAI>();
    pm.run_passes(model);
#endif

    // 3. Save OpenVINO IR model if requested
    if (enable_save_ov_model) {
        std::filesystem::path save_path = model_path.parent_path() / "openvino_model.xml";
        utils::save_openvino_model(model, save_path.string(), true);
    }

    return model;
#else
    OPENVINO_THROW("llama.cpp adapter support is disabled. Please rebuild with CMake option '-DRUN_WITH_LLAMA=ON'");
#endif
}

}  // namespace genai
}  // namespace ov
