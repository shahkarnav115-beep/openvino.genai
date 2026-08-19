// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_reader_v1.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"

#ifdef ENABLE_OV_GGUF_FRONTEND
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#endif

#include "utils.hpp"

namespace ov {
namespace genai {

std::shared_ptr<ov::Model> GgufReaderV1(
    const std::filesystem::path& model_path,
    bool enable_save_ov_model
) {
#ifdef ENABLE_OV_GGUF_FRONTEND
    ov::frontend::gguf::FrontEnd frontend;

    // 1. Register MakeStateful extension BEFORE conversion so normalization
    // converts KV cache SetRows ops into stateful Variables + ReadValue + Assign sinks
    frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
        ov::frontend::gguf::pass::MakeStateful()
    ));

    // 2. Load and convert GGUF model into stateful ov::Model
    auto input_model = frontend.load(model_path.string());
    auto model = frontend.convert(input_model);

    // 3. Apply AdaptToGenAI pass to rewrite GGUF-native inputs/outputs
    // into the OpenVINO GenAI LLMPipeline contract:
    // (input_ids [b, seq], attention_mask [b, kv_len], position_ids [b, seq], beam_idx [b] -> logits [b, seq, vocab])
    ov::pass::Manager pm;
    pm.register_pass<ov::frontend::gguf::pass::AdaptToGenAI>();
    pm.run_passes(model);

    // Save OpenVINO IR model if requested
    if (enable_save_ov_model) {
        std::filesystem::path save_path = model_path.parent_path() / "openvino_model.xml";
        utils::save_openvino_model(model, save_path.string(), true);
    }

    return model;
#else
    OPENVINO_THROW("GGUF support is disabled. Please build with -DENABLE_GGUF=ON");
#endif
}

}  // namespace genai
}  // namespace ov
