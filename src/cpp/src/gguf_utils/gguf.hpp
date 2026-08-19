// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "openvino/openvino.hpp"

using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

