// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <mkldnn.h>

#include "mkldnn_node.h"
#include "ngraph_ops/subgraph.hpp"

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
    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<ngraph::op::Subgraph> snippet;
    // Original subgraph node for fallback and regression testing
    // store it here since MKLDNN eraces CNNLayers at some point
    std::shared_ptr<ngraph::op::Subgraph> snippet_ref;
};

}  // namespace MKLDNNPlugin
