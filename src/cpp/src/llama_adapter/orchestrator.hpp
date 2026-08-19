// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace llama_adapter {

std::shared_ptr<ov::Model> get_stateful_ov_model(const std::filesystem::path& model_path, bool for_genai_pipeline = true);

}  // namespace llama_adapter
}  // namespace genai
}  // namespace ov
