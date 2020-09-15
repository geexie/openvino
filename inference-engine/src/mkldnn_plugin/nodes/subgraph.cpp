// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_layers.h>
#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>

#include <mkldnn.hpp>
#include <mkldnn_debug.h>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>

#include "ngraph_ops/subgraph.hpp"
#include "transformations/snippets/remarks.hpp"

#include "emitters/cpu_generator.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
/// ====================================================================================================================================================
/// MKLDNNSnippetNode encapsulates represents subgraph node in MKLDNN plugin
/// potentially, snippet can be placed as a postop to any support operation while it doesn't support postops itself
///
/// Supported formats
/// If tensor has less than 4 dims all formats are treated to be any-any
/// Snippet declares it can handle any format.
/// ToDo: Should we also declare that we han handle and dims? And presision? How is to declare this?
/// Real limitation is any but all blocked or all not blocked, try to make it happen later
///
/// presision: fp32
/// to support multiple presisions it seems we need to declare cartesian product of all possible presisions on inputs and outputs.
/// Could it be done in a mode elegant way?
/// ====================================================================================================================================================
MKLDNNPlugin::MKLDNNSnippetNode::MKLDNNSnippetNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    if ((snippet_ref = ngraph::as_type_ptr<ngraph::op::Subgraph>(this->getCnnLayer()->getNode()))) {
        ngraph::OutputVector subgraph_node_inputs;
        for (auto input : snippet_ref->input_values()) {
            subgraph_node_inputs.push_back(input);
        }
        auto new_body = ngraph::clone_function(*snippet_ref->get_body().get());
        snippet = std::make_shared<ngraph::op::Subgraph>(subgraph_node_inputs, new_body);
        ngraph::copy_runtime_info(snippet_ref, snippet);
        snippet->set_friendly_name(snippet_ref->get_friendly_name());
        // Disable for ref mode
        snippet->set_generator(std::make_shared<ngraph::snippet::CPUGenerator>());
    } else {
        snippet_ref.reset();
        snippet.reset();
    }

    remark(1) << "created MKLDNNSnippetNode for " << snippet->get_friendly_name() << "\n" << snippet_ref << "\n" << snippet << std::endl;
}

// It's actually initSupportedDescriptors
void MKLDNNPlugin::MKLDNNSnippetNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;
}

void MKLDNNPlugin::MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    // FIXME: dynamic batch requires dynamic shape on N dimencion in a canonical form
    // in ideal case all but broadcasted dimension and simd size should be dynamic
    config.dynBatchSupport = false;

    config.inConfs.resize(this->inDims.size());
    config.outConfs.resize(this->outDims.size());

    // keep them both default non-constant & non-inplace
    for (auto i = std::make_pair(config.inConfs.begin(), this->getParentEdges().begin()); i.first != config.inConfs.end(); ++i.first, ++i.second) {
        // i.first->constant = false;
        // i.first->inPlace = -1; // would rether disable it since no performance gain if running inside the topology
    }

    for (auto i = std::make_pair(config.outConfs.begin(), this->getChildEdges().begin()); i.first != config.outConfs.end(); ++i.first, ++i.second) {
        // i.first->constant = false;
        // i.first->inPlace = -1;
    }

    auto pushDesc = [&](mkldnn::memory::format format, memory::data_type precision) {
        auto adjast_format = [](const MKLDNNDims& dims, mkldnn::memory::format format) -> mkldnn::memory::format {
            auto current_format = dims.ndims() < 4 ? mkldnn::memory::any : format;
            if (dims.ndims() == 5 && current_format == mkldnn::memory::nchw)
                current_format = mkldnn::memory::ncdhw;
            if (dims.ndims() == 5 && current_format == mkldnn::memory::nChw8c)
                current_format = mkldnn::memory::nCdhw8c;

            auto block_szie = mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::avx512_common) ? 16 : 8;
            if (block_szie == 16 && current_format == mkldnn::memory::nChw8c)
                current_format = mkldnn::memory::nChw16c;
            if (block_szie == 16 && current_format == mkldnn::memory::nCdhw8c)
                current_format = mkldnn::memory::nCdhw16c;

            remark(1) << "current_format = " << mkldnn_fmt2str(memory::convert_to_c(current_format)) << std::endl;
            return current_format;
        };

        remark(1) << this->snippet->get_friendly_name() << " in "
            << " " << config.inConfs.size()
            << " " << getCnnLayer()->insData.size()
            << " " << this->getParentEdges().size()
            << " " << this->inDims.size() << std::endl;

        for (int k = 0; k < this->inDims.size(); k++) {
            config.inConfs[k].desc = MKLDNNMemoryDesc(this->inDims[k], precision, adjast_format(inDims[k], format));
        }

        remark(1) << this->snippet->get_friendly_name() << " out "
            << " " << config.outConfs.size()
            << " " << getCnnLayer()->outData.size()
            << " " << this->getChildEdges().size()
            << " " << this->getChildEdgesAtPort(0).size()
            << " " << this->outDims.size() << std::endl;

        for (int k = 0; k < this->outDims.size(); k++) {
            config.outConfs[k].desc = MKLDNNMemoryDesc(this->outDims[k], precision, adjast_format(outDims[k], format));
        }

        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    auto hasBroadcastByC = [this]() -> bool {
        for (auto op : ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet)->get_body()->get_ops()) {
            if (ngraph::op::supports_auto_broadcast(op)) {
                auto shape = op->input(0).get_shape();
                for (auto input : op->inputs()) {
                    if (input.get_shape().size() > 1 && shape[1] != input.get_shape()[1] && ngraph::shape_size(input.get_shape()) != 1) {
                        remark(11) << " POSSIBLE C BROADCAST IS DETECTED" << shape << " " << input.get_shape() << std::endl;
                        return true;
                    }
                }
            }
        }
        return false;
    };
    // FIXME: check if non-4 or 5 dimension, since no blocking in this case anyway
    if (!hasBroadcastByC())
        pushDesc(mkldnn::memory::nChw8c, memory::f32);
    pushDesc(mkldnn::memory::nchw, memory::f32);
}

void MKLDNNPlugin::MKLDNNSnippetNode::createPrimitive() {
    // should be something like the following
    // snippet = snippet_ref.canonical_from_this();
    // snippet->print_statistics(false);

    std::vector<MKLDNNEdgePtr> input_first_row;
    for (size_t i = 0; i < inDims.size(); i++) {
        auto edges = getParentEdgesAtPort(i);
        if (getParentEdgesAtPort(i).size() != 1) {
            THROW_IE_EXCEPTION << "Snippet layer " << getName() << " has >= 1 number of parent edges at port " << i;
        }

        input_first_row.push_back(edges[0]);

        // remark(11) << "parent " << i << " " << edge->getDesc().getLayout() << " " << edge->getDesc().getPrecision().name() << std::endl;
        // print_dims(edge->getDesc().getDims());
        // print_dims(edge->getDesc().getBlockingDesc().getBlockDims());
        // print_dims(edge->getDesc().getBlockingDesc().getOrder());
    }

    ngraph::op::Subgraph::BlockedShapeVector input_shapes;
    std::transform(input_first_row.begin(), input_first_row.end(), std::back_inserter(input_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::op::Subgraph::BlockedShape {
        ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
        ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return {shape, blocking, precision};
    });


    std::vector<MKLDNNEdgePtr> output_first_row;
    for (size_t i = 0; i < outDims.size(); i++) {
        auto edges = getChildEdgesAtPort(i);
        // Can it go with difference shape or presision to different edges? I assume no.
        output_first_row.push_back(edges[0]);

        for (auto& edge : edges) {
            // remark(11) << "child " << i << " " << edge->getDesc().getLayout() << std::endl;
            // print_dims(edge->getDesc().getDims());
            // print_dims(edge->getDesc().getBlockingDesc().getBlockDims());
            // print_dims(edge->getDesc().getBlockingDesc().getOrder());
        }
    }

    ngraph::op::Subgraph::BlockedShapeVector output_shapes;
    std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::op::Subgraph::BlockedShape {
        ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
        ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return {shape, blocking, precision};
    });

    auto res = snippet->generate(output_shapes, input_shapes);
    // snippet->print();
}

// FIXME: it seems that scheduler is something plugin dependent so it might be better to call generated code out from here,
// or pass executor functor to
void MKLDNNPlugin::MKLDNNSnippetNode::execute(mkldnn::stream strm) {
    auto& subgraph = snippet;

    ngraph::HostTensorVector inputs;
    auto params = subgraph->get_body()->get_parameters();
    for (size_t i = 0; i < inDims.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        IE_ASSERT(parents.size() == 1);
        auto &mem = parents[0]->getMemory();

        auto type = subgraph->input(i).get_element_type();
        auto ptr = reinterpret_cast<uint8_t *>(mem.GetData())
                        + mem.GetDescriptor().data.layout_desc.blocking.offset_padding *
                        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(mem.GetDescriptor().data.data_type));

        inputs.push_back(std::make_shared<ngraph::HostTensor>(type, params[i]->get_shape(), reinterpret_cast<void *>(ptr)));
    }

    ngraph::HostTensorVector outputs;
    auto results = subgraph->get_body()->get_results();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();

        auto type = subgraph->output(i).get_element_type();
        auto ptr = reinterpret_cast<uint8_t *>(mem.GetData())
                        + mem.GetDescriptor().data.layout_desc.blocking.offset_padding *
                        MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(mem.GetDescriptor().data.data_type));

        outputs.push_back(std::make_shared<ngraph::HostTensor>(type, results[i]->get_shape(), reinterpret_cast<void *>(ptr)));
    }

    subgraph->evaluate(outputs, inputs);
}

bool MKLDNNPlugin::MKLDNNSnippetNode::created() const {
    return getType() == Subgraph;
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);
