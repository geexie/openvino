// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <mkldnn.h>

#include "mkldnn_node.h"
#include "transformations/snippets/subgraph.hpp"

namespace MKLDNNPlugin {

class MKLDNNSnippetNode : public MKLDNNNode {
public:
    MKLDNNSnippetNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNSnippetNode() override = default;

    // It should be initSupportedDescriptors after all
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;

    // Here we convert to canonical for & jit everything
    void createPrimitive() override;

    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(mkldnn::stream strm) override;

private:

    bool evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const;
    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<ngraph::op::Subgraph> snippet;
    // Original subgraph node for fallback and regression testing
    // store it here since MKLDNN eraces CNNLayers at some point
    std::shared_ptr<ngraph::op::Subgraph> snippet_ref;

    ngraph::snippets::Schedule schedule;

    // FIXME: refactor to schedule later
    int optimalTensorRank = 6;
    bool canUseOptimizedImpl = /*false*/true;
    bool isDynBatchEnabled = false;
    size_t batchDimIdx = 0;
    size_t tensorRank = 0;
    size_t fullWorkAmount = 0;
    size_t schedulerWorkAmount = 0;

    std::vector<std::vector<size_t>> dims_in = {};
    std::vector<std::vector<size_t>> offsets_in = {};
    std::vector<size_t> dims_out = {};
    std::vector<size_t> offsets_out = {};
    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;

    struct jit_eltwise_params {
        size_t inputs_number;
        size_t input_size;

        InferenceEngine::Precision src_prc[7];
        InferenceEngine::Precision dst_prc;

        std::vector<size_t> src_offsets[7];
        std::vector<size_t> dst_offsets;

        size_t src_size[7];
        size_t dst_size;
        size_t oc_size;
    };

    jit_eltwise_params jep = {};

    // std::vector<size_t> offsets_oc = {};
};

}  // namespace MKLDNNPlugin
