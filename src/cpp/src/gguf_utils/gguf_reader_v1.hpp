// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"

namespace ov {
namespace genai {

std::shared_ptr<ov::Model> GgufReaderV1(
    const std::filesystem::path& model_path,
    bool enable_save_ov_model = false
);

}  // namespace genai
}  // namespace ov
