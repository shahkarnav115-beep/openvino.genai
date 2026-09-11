// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "orchestrator.hpp"
#include "extract_cgraph.hpp"

#include "openvino/core/except.hpp"
#include <map>

#ifdef RUN_WITH_LLAMA
#include "ggml.h"
#include "../../../thirdparty/llama.cpp/ggml/src/ggml-impl.h"
#include "ggml-decoder.h"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/llama_cpp_to_stateful.h"
#endif

namespace ov {
namespace genai {
namespace llama_adapter {

std::shared_ptr<ov::Model> get_stateful_ov_model(const std::filesystem::path& model_path, bool for_genai_pipeline) {
#ifdef RUN_WITH_LLAMA
    auto ext = extract_cgraph(model_path);

    static constexpr bool is_static = false;
    static constexpr bool is_stateful = true;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(ext.cgraph, is_static);

    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto decoder = std::make_shared<GgmlOvDecoder>(ext.cgraph, m_params, c_params, model_weights, is_static, is_stateful);

    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
        ggml::pass::LlamaCppToStateful(for_genai_pipeline)
    ));

    auto input_model = frontend.load(std::static_pointer_cast<ov::frontend::gguf::GgufDecoder>(decoder));
    std::shared_ptr<ov::Model> model = frontend.convert(input_model);
    OPENVINO_ASSERT(model != nullptr, "Frontend failed to convert GgmlOvDecoder to stateful ov::Model");

    // Bind ExtractedCGraph resources (llama_model, llama_context) to ov::Model runtime info so tensor buffers remain alive with the model
    struct ExtractedCGraphHolder {
        ExtractedCGraph ext;
    };
    auto holder = std::make_shared<ExtractedCGraphHolder>();
    holder->ext = std::move(ext);
    model->get_rt_info()["llama_cgraph_holder"] = holder;

    return model;
#else
    OPENVINO_THROW("llama.cpp adapter support is disabled. Please rebuild with CMake option '-DRUN_WITH_LLAMA=ON'");
#endif
}

}  // namespace llama_adapter
}  // namespace genai
}  // namespace ov
