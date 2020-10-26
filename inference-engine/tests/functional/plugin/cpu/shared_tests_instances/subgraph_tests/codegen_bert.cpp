
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_bert.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    INSTANTIATE_TEST_CASE_P(NoReshape, CodegenBert,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
//            ::testing::Values(InferenceEngine::SizeVector({1, 42, 128, 768})),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 16, 64})),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 16, 64})//,
                        //       InferenceEngine::SizeVector({1, 42, 16,  1}),
                        //       InferenceEngine::SizeVector({1, 42,  1, 64}),
                              /*InferenceEngine::SizeVector({1,  1, 16, 64})*/),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenBert::getTestCaseName);
}  // namespace