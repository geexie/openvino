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

    // Here we jit everything
    void createPrimitive() override;

    // Here is the check, it seem that every node override it
    bool created() const override;

    // Here we execute
    void execute(mkldnn::stream strm) override;

private:
    // store it here since MKLDNN eraces CNNLayers at some point
    std::shared_ptr<ngraph::op::Subgraph> snippet;
    // store original snippet node for fallback and regression testing
    std::shared_ptr<ngraph::op::Subgraph> snippet_ref;

    // FIXME: Disable this for POC
    bool letMeSupportDynamicBatch() const {return false;}

    std::vector<mkldnn::memory::format> getSupportedFormats() const {
        return {
            mkldnn::memory::nChw8c,
            mkldnn::memory::nchw//,
            // mkldnn::memory::nCdhw8c,
            // mkldnn::memory::ncdhw,
            // mkldnn::memory::format::any
        };
    }

    bool useReference() const {
        return false;
    }

    bool runComparisonToReference() const {
        return false;
    }
};

}  // namespace MKLDNNPlugin
