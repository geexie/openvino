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

/////////// Utility functions for Snippets ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

auto static print_subgraph(const std::shared_ptr<ngraph::Function>& body) -> void {
    int qqq = 0;
    for (auto op : body->get_ordered_ops()) {
        remark(10) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }
}

auto static print_snippet_statistics(const std::shared_ptr<ngraph::Node>& snippet, bool verbose) -> void {
    auto getNodeInventory = [](std::shared_ptr<ngraph::Node> n) -> size_t {
        size_t total = 0;

        for (auto input : n->inputs()) {
            total += input.get_tensor().size();
        }

        for (auto output : n->outputs()) {
            total += output.get_tensor().size();
        }

        if (auto subgraph = ngraph::as_type_ptr<ngraph::op::Subgraph>(n)) {
            for (auto op : subgraph->get_body()->get_ordered_ops()) {
                if (ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                    total += op->output(0).get_tensor().size();
                }
            }
        }

        return total;
    };

    auto getFunctionInventory = [getNodeInventory](std::shared_ptr<ngraph::Function> f) -> size_t {
        size_t total = 0;
        for (auto op : f->get_ordered_ops()) {
            // Results and parameters are artificially introduced,
            // while Constants are already considered if they are inputs of other operation
            // this should lead to 1:1 inventory for single node operations
            if (!ngraph::as_type_ptr<ngraph::opset1::Parameter>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Result>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                total += getNodeInventory(op);
            }
        }
        return total;
    };

    auto countConstants = [](std::shared_ptr<ngraph::Function> f) -> size_t {
        size_t count = 0;
        for (auto op : f->get_ordered_ops()) {
            count += !!ngraph::as_type_ptr<ngraph::opset1::Constant>(op) ? 1 : 0;
        }
        return count;
    };

    if (auto subgraph = ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet)) {
        auto body = subgraph->get_body();

        std::cout << subgraph->get_friendly_name()
                  << ";" << subgraph
                  << ";" << body->get_ops().size()
                  << ";" << body->get_parameters().size()
                  << ";" << body->get_results().size()
                  << ";" << countConstants(body)
                  << ";" << getFunctionInventory(body)
                  << ";" << getNodeInventory(snippet) << std::endl;

        if (verbose)
            print_subgraph(body);
    }
}

auto static print_dims = [](const InferenceEngine::SizeVector& dims) -> void {
    for (auto dim : dims) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        snippet->set_generator(std::make_shared<ngraph::snippet::CPUGenerator>());
    } else {
        snippet_ref.reset();
        snippet.reset();
    }

    remark(1) << snippet->get_friendly_name() << " is stored to NODE\n"
              << snippet_ref << "\n"
              << snippet << std::endl;
}

// It's actually initSupportedDescriptors
void MKLDNNPlugin::MKLDNNSnippetNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;
}

void MKLDNNPlugin::MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // Pos ops are usually set here.
    // here we set primitives descriptors and post ops, learn about them more
    // setPostOps(attr, true);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = letMeSupportDynamicBatch();

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

    // Snippet declare it can handle any format.
    // ToDo: Should we also declare that we han handle and dims? And presision? How is to declare this?
    // Real limitation is any but all blocked or all not blocked, try to make it happen later
    auto pushDesc = [&](mkldnn::memory::format format) {
        using triple = decltype(std::make_tuple(config.inConfs.begin(), this->getParentEdges().begin(), getCnnLayer()->insData.begin()));

        auto precision = [&](triple& a) {
            return MKLDNNExtensionUtils::IEPrecisionToDataType(std::get<2>(a)->lock()->getPrecision());
        };

        remark(1) << this->snippet->get_friendly_name() << " in "
            << " " << config.inConfs.size()
            << " " << getCnnLayer()->insData.size()
            << " " << this->getParentEdges().size()
            << " " << this->inDims.size() << std::endl;

        for (int k = 0; k < this->inDims.size(); k++) {
            remark(1) << "  MKLDNNSnippetNode input dims " << this->inDims[k].ndims()  << std::endl;
            // print_dims(this->inDims[k].ToSizeVector());

            auto current_format = this->inDims[k].ndims() < 4 ? mkldnn::memory::any : format;
            if (this->inDims[k].ndims() == 5 && current_format == mkldnn::memory::nchw)
                current_format = mkldnn::memory::ncdhw;
            if (this->inDims[k].ndims() == 5 && current_format == mkldnn::memory::nChw8c)
                current_format = mkldnn::memory::nCdhw8c;

            remark(1) << "current_format = " << mkldnn_fmt2str(memory::convert_to_c(current_format)) << std::endl;

            if (auto cnnlayer_data = getCnnLayer()->insData[k].lock()) {
                config.inConfs[k].desc = MKLDNNMemoryDesc(this->inDims[k],
                MKLDNNExtensionUtils::IEPrecisionToDataType(/*cnnlayer_data->getPrecision()*/Precision::FP32), current_format);
            } else {
                THROW_IE_EXCEPTION << "nulptr trying to lock getCnnLayer()->insData[" << k << "] for " << this->getName();
            }
        }

        remark(1) << this->snippet->get_friendly_name() << " out "
            << " " << config.outConfs.size()
            << " " << getCnnLayer()->outData.size()
            << " " << this->getChildEdges().size()
            << " " << this->getChildEdgesAtPort(0).size()
            << " " << this->outDims.size() << std::endl;

        // it may be multiple edges per output port
        for (int k = 0; k < this->outDims.size(); k++) {
            remark(1) << "  MKLDNNSnippetNode output dims " << this->outDims[k].ndims() << std::endl;
            // print_dims(this->outDims[k].ToSizeVector());

            auto current_format = this->outDims[k].ndims() < 4 ? mkldnn::memory::any : format;
            if (this->outDims[k].ndims() == 5 && current_format == mkldnn::memory::nchw)
                current_format = mkldnn::memory::ncdhw;
            if (this->outDims[k].ndims() == 5 && current_format == mkldnn::memory::nChw8c)
                current_format = mkldnn::memory::nCdhw8c;

            remark(1) << "current_format = " << mkldnn_fmt2str(memory::convert_to_c(current_format)) << std::endl;

            auto precision = MKLDNNExtensionUtils::IEPrecisionToDataType(/*getCnnLayer()->outData[k]->getPrecision()*/Precision::FP32);
            config.outConfs[k].desc = MKLDNNMemoryDesc(this->outDims[k], precision, current_format);
        }

        remark(1) << "descriptor created!!" << std::endl;

        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    bool hasBroadcastByC = [this]() -> bool {
        for (auto op : ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet)->get_body()->get_ops()) {
            if (ngraph::op::supports_auto_broadcast(op)) {
                auto shape = op->input(0).get_shape();
                for (auto input : op->inputs()) {
                    if (input.get_shape().size() > 1 && shape[1] != input.get_shape()[1] && ngraph::shape_size(input.get_shape()) != 1) {
                        remark(1) << " POSSIBLE C BROADCAST IS DETECTED" << shape << " " << input.get_shape() << std::endl;
                        return true;
                    }
                }
            }
        }
        return false;
    }();

    // FIXME: temporary disables blocking if broadcast by C, since it requires modifications in reference nGraph code to run correctly
    for (auto fmt : (!hasBroadcastByC) ? getSupportedFormats() : std::vector<mkldnn::memory::format>({mkldnn::memory::nchw})) {
        pushDesc(fmt);
    }

    remark(1) << "MKLDNNSnippetNode: supportedPrimitiveDescriptors are empty -DONE " << std::endl;
}

void MKLDNNPlugin::MKLDNNSnippetNode::createPrimitive() {
    remark(1) << "MKLDNNSnippetNode:  createPrimitive" << std::endl << std::endl;
    // We also can think of canonization as of pass to copy original subgraph and transforming it to canonical form suitable for code generation

    // print_snippet_statistics(snippet, false);

    if (auto subgraph = ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet)) {
        // here we have information about inputs and outputs we will actually have

        // std::cout << "MKLDNN inDims.size() = " << inDims.size()
        //           << " getParentEdges().size() = " << getParentEdges().size()
        //           << " outDims.size() = " << outDims.size()
        //           << " getChildEdges().size() = " << getChildEdges().size()
        //           << " subgraph->get_body()->get_parameters().size() = " << subgraph->get_body()->get_parameters().size()
        //           << " subgraph->get_body()->get_results().size() = " << subgraph->get_body()->get_results().size()
        //           << std::endl;


        // pass actual parameters and results shapes to generate for as well as channel mapping,
        // we need to distinguish between 5d tensors that represents <N, C, H, W, c> and <N, C, D, H, W> somehow like locked dimensions
        // ngraph::AxisVector to code
        //
        // Dunamic dimension like <N, C, H, W> = <?, ?, ?, ?> or <N, C, H, W> = <?, ?, ?, W> means that we can merge the consecutive and linearise
        // <N, C, H, W> = <?> or <N, C, H, W> = <?, W> folding consecutive dimensions
        ngraph::op::Subgraph::BlockedShapeVector input_shapes;
        std::vector<MKLDNNEdgePtr> input_first_row;

        // parent/child edges
        for (size_t i = 0; i < inDims.size(); i++) {
            auto edges = getParentEdgesAtPort(i);
            if (getParentEdgesAtPort(i).size() != 1) {
                THROW_IE_EXCEPTION << "Snippet layer " << getName() << " has >= 1 number of parent edges at port " << i;
            }

            auto edge = edges[0];
            input_first_row.push_back(edge);

            ////////////
            // remark(11) << "parent " << i << " " << edge->getDesc().getLayout() << " " << edge->getDesc().getPrecision().name() << std::endl;

            auto dims = edge->getDesc().getDims();
            // print_dims(dims);

            auto blocked_dims = edge->getDesc().getBlockingDesc().getBlockDims();
            // print_dims(blocked_dims);

            auto orders = edge->getDesc().getBlockingDesc().getOrder();
            // print_dims(orders);
        }

        std::transform(input_first_row.begin(), input_first_row.end(), std::back_inserter(input_shapes),
                   [](const MKLDNNEdgePtr& edge) -> ngraph::op::Subgraph::BlockedShape {
            ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
            ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
            ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
            return {shape, blocking, precision};
        });

        ngraph::op::Subgraph::BlockedShapeVector output_shapes;
        std::vector<MKLDNNEdgePtr> output_first_row;

        for (size_t i = 0; i < outDims.size(); i++) {
            auto edges = getChildEdgesAtPort(i);

            // Can it go with difference shape or presision to different edges. I assume no.
            // std::cout << edges.size() << std::endl;
            auto edge = edges[0];
            output_first_row.push_back(edge);

            for (auto& edge : edges) {
                // remark(11) << "child " << i << " " << edge->getDesc().getLayout() << std::endl;
                auto dims = edge->getDesc().getDims();
                // print_dims(dims);

                auto blocked_dims = edge->getDesc().getBlockingDesc().getBlockDims();
                // print_dims(blocked_dims);

                auto orders = edge->getDesc().getBlockingDesc().getOrder();
                // print_dims(orders);
            }
        }

        std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_shapes),
                   [](const MKLDNNEdgePtr& edge) -> ngraph::op::Subgraph::BlockedShape {
            ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
            ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
            ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
            return {shape, blocking, precision};
        });

        auto res = subgraph->generate(output_shapes, input_shapes);

        remark(1) << "transformad to " << res <<  "\n";
        // print_subgraph(subgraph->get_body());
    }
}

void MKLDNNPlugin::MKLDNNSnippetNode::execute(mkldnn::stream strm) {
    if (auto subgraph = ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet_ref)) {
        ngraph::HostTensorVector inputs;
#if 1
        for (size_t i = 0; i < inDims.size(); i++) {
            auto & parents = getParentEdgesAtPort(i);
            IE_ASSERT(parents.size() == 1);
            auto &mem = parents[0]->getMemory();

            auto input = subgraph->input(i);
            auto ptr = reinterpret_cast<uint8_t *>(mem.GetData())
                            + mem.GetDescriptor().data.layout_desc.blocking.offset_padding *
                            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(mem.GetDescriptor().data.data_type));

            // FIXME: check if types are compatible
            inputs.push_back(std::make_shared<ngraph::HostTensor>(
                input.get_element_type(), subgraph->get_body()->get_parameters()[i]->get_shape(), reinterpret_cast<void *>(ptr)));
        }
#endif
        ngraph::HostTensorVector outputs;
#if 1
        for (size_t i = 0; i < outDims.size(); i++) {
            auto & child = getChildEdgesAtPort(i);
            auto &mem = child[0]->getMemory();

            auto output = subgraph->output(i);
            auto ptr = reinterpret_cast<uint8_t *>(mem.GetData())
                            + mem.GetDescriptor().data.layout_desc.blocking.offset_padding *
                            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(mem.GetDescriptor().data.data_type));

            // FIXME: check if types are compatible
            outputs.push_back(std::make_shared<ngraph::HostTensor>(
                output.get_element_type(), subgraph->get_body()->get_results()[i]->get_shape(), reinterpret_cast<void *>(ptr)));
        }

#endif
        if (useReference())
            subgraph->get_body()->evaluate(outputs, inputs);
        else
            subgraph->evaluate(outputs, inputs);

        // for (int k = 0; k < inputs.size(); k++) {
        //     auto ptr = reinterpret_cast<float*>(inputs[k]->get_data_ptr());
        //     std::cout << ptr[0] << " "  << ptr[1] << " " << ptr[2] << " " << ptr[3] << std::endl;
        // }
        // std::cout << std::endl << std::endl;
        // print_subgraph(subgraph->get_body());
        // print_subgraph(ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet_ref)->get_body());
        // for (int k = 0; k < outputs.size(); k++) {
        //     auto ptr = reinterpret_cast<float*>(outputs[k]->get_data_ptr());
        //     std::cout << ptr[0] << " "  << ptr[1] << " " << ptr[2] << " " << ptr[3] << std::endl;
        // }
        // for (int k = 0; k < outputs.size(); k++) {
        //     auto ptr = reinterpret_cast<unsigned*>(outputs[k]->get_data_ptr());
        //     std::cout << ptr[0] << " "  << ptr[1] << " " << ptr[2] << " " << ptr[3] << std::endl;
        // }
        // std::cout << std::endl << std::endl;

        if (runComparisonToReference()) {
            print_subgraph(subgraph->get_body());

            auto body_ref = ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet_ref)->get_body();

            // FIXME: reshape inputs. works only if no blocking, should add reorder to compare with original graph
            ngraph::HostTensorVector inputs_ref;
            for (size_t i = 0; i < inDims.size(); i++) {
                auto & parents = getParentEdgesAtPort(i);
                IE_ASSERT(parents.size() == 1);
                auto &mem = parents[0]->getMemory();

                auto input = snippet_ref->input(i);

                auto ptr = reinterpret_cast<uint8_t *>(mem.GetData())
                                + mem.GetDescriptor().data.layout_desc.blocking.offset_padding *
                                MKLDNNExtensionUtils::sizeOfDataType(
                                mkldnn::memory::data_type(mem.GetDescriptor().data.data_type));
                // check if types are compatible
                inputs_ref.push_back(std::make_shared<ngraph::HostTensor>(input.get_element_type(),
                body_ref->get_parameters()[i]->get_shape(),
                reinterpret_cast<void *>(ptr)));
            }

            ngraph::HostTensorVector outputs_ref;
            std::vector<std::vector<float>> ref_data(outDims.size());
            for (size_t i = 0; i < outDims.size(); i++) {
                auto output = snippet_ref->output(i);
                ref_data[i].resize(ngraph::shape_size(output.get_shape()));

                auto ptr = reinterpret_cast<uint8_t *>(&(ref_data[i][0]));
                outputs_ref.push_back(std::make_shared<ngraph::HostTensor>(output.get_element_type(),
                                body_ref->get_results()[i]->get_shape(), reinterpret_cast<void *>(ptr)));
            }

            remark(10) << "Evaluating reference " << outputs_ref.size() << " " << inputs_ref.size() << std::endl;
            print_subgraph(body_ref);
            body_ref->evaluate(outputs_ref, inputs_ref);
            remark(10) << "Evaluating reference - done" << std::endl;

            auto ref = std::vector<std::vector<std::uint8_t>>(outDims.size());
            for (const auto &result : body_ref->get_results()) {
                const auto &resultIndex = body_ref->get_result_index(result);
                auto &data_ref = ref[resultIndex];
                data_ref.resize(shape_size(result->get_shape()) * result->get_element_type().size());
                outputs_ref[resultIndex]->read(data_ref.data(), data_ref.size());
            }

            auto act = std::vector<std::vector<std::uint8_t>>(outDims.size());
            for (const auto &result : subgraph->get_body()->get_results()) {
                const auto &resultIndex = subgraph->get_body()->get_result_index(result);
                auto &data_act = act[resultIndex];
                data_act.resize(shape_size(result->get_shape()) * result->get_element_type().size());
                outputs[resultIndex]->read(data_act.data(), data_act.size());
            }

            // reshape ref
            float epsilon = std::numeric_limits<float>::epsilon();
            for (auto op : body_ref->get_ops()) {
                if (ngraph::as_type_ptr<ngraph::opset1::Erf>(op)) {
                    epsilon = 1e-3;
                }
            }

            for (int k = 0; k < ref.size(); k++) {
                const float* ref1 = reinterpret_cast<float*>(&ref[k][0]);
                const float* act1 = reinterpret_cast<float*>(&act[k][0]);

                int differ = 0;
                for (int i = 0; i < ref[k].size()/sizeof(float); i++) {
                    if (std::abs(ref1[i]-act1[i]) > epsilon || std::isnan(ref1[i]) != std::isnan(act1[i])) {
                        std::cout << i << ": " << ref1[i] << " " << act1[i] << " " << std::abs(ref1[i]-act1[i]) << std::endl;
                        // THROW_IE_EXCEPTION << ref1[i] << " " << act1[i] << " " << std::abs(ref1[i]-act1[i]);
                        differ++;

                        if (differ > 100) {
                            break;
                        }
                    }
                }

                if (differ) {
                    THROW_IE_EXCEPTION << "reference and generated results appear to differ";//ref1[i] << " " << act1[i] << " " << std::abs(ref1[i]-act1[i]);
                }
            }

            remark(10) << "Code - done" << std::endl;
        }
    }
}

bool MKLDNNPlugin::MKLDNNSnippetNode::created() const {
    return getType() == Subgraph;
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);
