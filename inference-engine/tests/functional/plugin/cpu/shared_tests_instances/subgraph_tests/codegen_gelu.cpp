
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_gelu.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    INSTANTIATE_TEST_CASE_P(NoReshape, CodegenGelu,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
//            ::testing::Values(InferenceEngine::SizeVector({1, 42, 128, 768})),
           ::testing::Values(InferenceEngine::SizeVector({/*1, */1, 384, 4096})),
            // ::testing::Values(InferenceEngine::SizeVector({1, /*42*/1, 16, 64})),
            ::testing::Values(true, false),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenGelu::getTestCaseName);
}  // namespace