// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>
#include <tuple>

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

// void MKLDNNPlugin::MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
//     std::vector<Precision> supportedPrecisions = {
//             Precision::FP32,
//             Precision::U8,
//             Precision::I8,
//             Precision::U16,
//             Precision::I16,
//             Precision::BF16,
//             Precision::I32
//     };

//     if (!supportedPrimitiveDescriptors.empty())
//         return;

//     canUseOptimizedImpl = mayiuse(cpu::sse42);

//     size_t expectedInputsNum = getOpInputsNum();
//     for (auto& postOp : fusedWith) {
//         auto* eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode*>(postOp.get());
//         if (eltwiseNode != nullptr) {
//             expectedInputsNum += eltwiseNode->getOpInputsNum() - 1;
//         }
//     }
//     if (getParentEdges().size() > MAX_ELTWISE_INPUTS)
//         THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support more than " << MAX_ELTWISE_INPUTS
//                            << " inputs (actual = " << getParentEdges().size() << ")";

//     if (expectedInputsNum != getParentEdges().size())
//         THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` has invalid input number of inputs: expected = " << expectedInputsNum
//                            << " (actual = " << getParentEdges().size() << ")";

//     std::vector<InferenceEngine::Precision> inputPrecisions;
//     for (int i = 0; i < getCnnLayer()->insData.size(); i++) {
//         inputPrecisions.push_back(getCnnLayer()->insData[i].lock()->getPrecision());
//     }

//     for (auto& fusedNode : fusedWith) {
//         if (fusedNode->getType() == Eltwise) {
//             for (int i = 1; i < fusedNode->getCnnLayer()->insData.size(); i++) {
//                 inputPrecisions.push_back(fusedNode->getCnnLayer()->insData[i].lock()->getPrecision());
//             }
//         }
//     }

//     if (inputPrecisions.size() != getParentEdges().size())
//         THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` has invalid input precisions configuration.";

//     InferenceEngine::Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
//     if (!fusedWith.empty()) {
//         auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
//         if (lastFusedLayer) {
//             outputPrecision = lastFusedLayer->outData[0]->getPrecision();
//         }
//     }

//     if (!mayiuse(avx512_core)) {
//         bool hasBF16 = false;
//         for (auto &inPrc : inputPrecisions)
//             if (inPrc == Precision::BF16)
//                 hasBF16 = true;

//         if (outputPrecision == Precision::BF16 || hasBF16)
//             THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support BF16 precision on this target.";
//     }

//     auto filterPrecision = [&](Precision& prc) {
//         if (!canUseOptimizedImpl) {
//             return Precision(Precision::FP32);
//         } else if (std::find(supportedPrecisions.begin(), supportedPrecisions.end(), prc) == supportedPrecisions.end()) {
//             if (prc == Precision::U32 || prc == Precision::I64 || prc == Precision::U64) {
//                 return Precision(Precision::I32);
//             } else {
//                 THROW_IE_EXCEPTION << "Eltwise node with name `" << getName() << "` doesn't support " << prc << " precision.";
//             }
//         } else {
//             return prc;
//         }
//     };

//     for (int i = 0; i < inputPrecisions.size(); i++) {
//         inputPrecisions[i] = filterPrecision(inputPrecisions[i]);
//     }
//     outputPrecision = filterPrecision(outputPrecision);

//     // TODO: delete after new LPT (ngraph based) is merged
//     // WA is needed to handle bug in LPT that produces wrong precision after average pooling (I8/U8 instead of FP32)
//     if (eltwiseOp == MulAdd && (inputPrecisions[0] == Precision::U8 || inputPrecisions[0] == Precision::I8)) {
//         auto poolingLayer = dynamic_cast<PoolingLayer*>(getParentEdgesAtPort(0)[0]->getParent()->getCnnLayer().get());
//         if (poolingLayer && poolingLayer->_type == PoolingLayer::AVG) {
//             inputPrecisions[0] = Precision::FP32;
//         }
//     }

//     enum LayoutType {
//         Planar,
//         ChannelsFirst,
//         Blocked
//     };

//     auto initDesc = [&] (LayoutType lt) -> PrimitiveDescInfo {
//         auto createMemoryDesc = [lt](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
//             if (lt == ChannelsFirst) {
//                 std::vector<size_t> blocks = edge->getDims().ToSizeVector();
//                 std::vector<size_t> order;
//                 order.push_back(0);
//                 for (size_t j = 2; j < blocks.size(); j++)
//                     order.push_back(j);
//                 if (blocks.size() > 1)
//                     order.push_back(1);

//                 return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
//             } else if (lt == Blocked && edge->getDims()[1] != 1) {
//                 size_t blockSize = mayiuse(cpu::avx512_common) ? 16 : 8;

//                 std::vector<size_t> blocks = edge->getDims().ToSizeVector();
//                 std::vector<size_t> order(blocks.size());
//                 for (size_t j = 0; j < order.size(); j++)
//                     order[j] = j;

//                 blocks[1] = div_up(blocks[1], blockSize);
//                 blocks.push_back(blockSize);
//                 order.push_back(1);

//                 return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
//             } else {
//                 std::vector<size_t> blocks = edge->getDims().ToSizeVector();
//                 std::vector<size_t> order(blocks.size());
//                 for (size_t j = 0; j < order.size(); j++)
//                     order[j] = j;

//                 return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
//             }
//         };

//         size_t offset = std::numeric_limits<size_t>::max();
//         InferenceEngine::LayerConfig config;
//         config.dynBatchSupport = getChildEdgeAt(0)->getDims().ndims() > 1 && getChildEdgeAt(0)->getDims() == getParentEdgeAt(0)->getDims();

//         for (size_t i = 0; i < getParentEdges().size(); i++) {
//             InferenceEngine::DataConfig dataConfig;
//             dataConfig.inPlace = (!i && canBeInPlace() && inputPrecisions[i] == outputPrecision) ? 0 : -1;
//             dataConfig.constant = false;


//             dataConfig.desc = createMemoryDesc(getParentEdgeAt(i), inputPrecisions[i], offset);

//             config.inConfs.push_back(dataConfig);
//         }

//         InferenceEngine::DataConfig dataConfig;
//         dataConfig.inPlace = -1;
//         dataConfig.constant = false;

//         dataConfig.desc = createMemoryDesc(getChildEdgeAt(0), outputPrecision, offset);

//         config.outConfs.push_back(dataConfig);

//         impl_desc_type impl_type;
//         if (mayiuse(cpu::avx512_common)) {
//             impl_type = impl_desc_type::jit_avx512;
//         } else if (mayiuse(cpu::avx2)) {
//             impl_type = impl_desc_type::jit_avx2;
//         } else if (mayiuse(cpu::sse42)) {
//             impl_type = impl_desc_type::jit_sse42;
//         } else {
//             impl_type = impl_desc_type::ref;
//         }

//         return {config, impl_type, MKLDNNMemoryDesc(config.outConfs[0].desc).getFormat()};
//     };

//     bool isChannelsFirstApplicable = one_of(getChildEdgeAt(0)->getDims().ndims(), 1, 2, 4, 5);
//     for (size_t i = 0; i < getParentEdges().size(); i++) {
//         isChannelsFirstApplicable = isChannelsFirstApplicable && one_of(getParentEdgeAt(i)->getDims().ndims(), 1, 2, 4, 5);
//         isChannelsFirstApplicable = isChannelsFirstApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
//     }

//     bool isBlockedApplicable = one_of(getChildEdgeAt(0)->getDims().ndims(), 4, 5);
//     for (size_t i = 0; i < getParentEdges().size(); i++) {
//         isBlockedApplicable = isBlockedApplicable && one_of(getParentEdgeAt(i)->getDims().ndims(), 4, 5);
//         isBlockedApplicable = isBlockedApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
//     }

//     if (isChannelsFirstApplicable)
//         supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
//     if (isBlockedApplicable)
//         supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
//     supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
// }

#if 1
void MKLDNNPlugin::MKLDNNSnippetNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };

    // InferenceEngine::LayerConfig config;
    // // FIXME: dynamic batch requires dynamic shape on N dimencion in a canonical form
    // // in ideal case all but broadcasted dimension and simd size should be dynamic
    // config.dynBatchSupport = false;

    // config.inConfs.resize(this->inDims.size());
    // config.outConfs.resize(this->outDims.size());

    // // keep them both default non-constant & non-inplace
    // for (auto i = std::make_pair(config.inConfs.begin(), this->getParentEdges().begin()); i.first != config.inConfs.end(); ++i.first, ++i.second) {
    //     // i.first->constant = false;
    //     // i.first->inPlace = -1; // would rether disable it since no performance gain if running inside the topology
    // }

    // for (auto i = std::make_pair(config.outConfs.begin(), this->getChildEdges().begin()); i.first != config.outConfs.end(); ++i.first, ++i.second) {
    //     // i.first->constant = false;
    //     // i.first->inPlace = -1;
    // }

    // auto pushDesc = [&](mkldnn::memory::format format, memory::data_type precision) {
    //     std::cout << "this->inDims.size() = " << this->inDims.size() << std::endl;
    //     auto adjast_format = [](const MKLDNNDims& dims, mkldnn::memory::format format) -> mkldnn::memory::format {
    //         auto current_format = dims.ndims() < 4 ? mkldnn::memory::any : format;
    //         if (dims.ndims() == 5 && current_format == mkldnn::memory::nchw)
    //             current_format = mkldnn::memory::ncdhw;
    //         if (dims.ndims() == 5 && current_format == mkldnn::memory::nChw8c)
    //             current_format = mkldnn::memory::nCdhw8c;

    //         auto block_szie = /*mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::avx512_common) ? 16 :*/ 8;
    //         if (block_szie == 16 && current_format == mkldnn::memory::nChw8c)
    //             current_format = mkldnn::memory::nChw16c;
    //         if (block_szie == 16 && current_format == mkldnn::memory::nCdhw8c)
    //             current_format = mkldnn::memory::nCdhw16c;

    //         remark(11) << "current_format = " << mkldnn_fmt2str(memory::convert_to_c(current_format)) << std::endl;
    //         return current_format;
    //     };

    //     remark(11) << this->snippet->get_friendly_name() << " in "
    //         << " " << config.inConfs.size()
    //         << " " << getCnnLayer()->insData.size()
    //         << " " << this->getParentEdges().size()
    //         << " " << this->inDims.size()
    //         << " " << precision << std::endl;

    //     for (int k = 0; k < this->inDims.size(); k++) {
    //         // return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
    //         config.inConfs[k].desc = MKLDNNMemoryDesc(this->inDims[k], precision, adjast_format(inDims[k], format));
    //     }

    //     remark(11) << this->snippet->get_friendly_name() << " out "
    //         << " " << config.outConfs.size()
    //         << " " << getCnnLayer()->outData.size()
    //         << " " << this->getChildEdges().size()
    //         << " " << this->getChildEdgesAtPort(0).size()
    //         << " " << this->outDims.size() << std::endl;

    //     for (int k = 0; k < this->outDims.size(); k++) {
    //         config.outConfs[k].desc = MKLDNNMemoryDesc(this->outDims[k], precision, adjast_format(outDims[k], format));
    //     }

    //     supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    // };

    auto hasBroadcastByC = [this]() -> bool {
        for (auto op : ngraph::as_type_ptr<ngraph::op::Subgraph>(snippet)->get_body()->get_ops()) {
            if (ngraph::op::supports_auto_broadcast(op)) {
                auto shape = op->input(0).get_shape();
                // Filter out scalar empty shape Shape{}
                if (ngraph::shape_size(shape) != 1) {
                    for (auto input : op->inputs()) {
                        if (input.get_shape().size() > 1 && shape[1] != input.get_shape()[1] && ngraph::shape_size(input.get_shape()) != 1) {
                            remark(11) << " POSSIBLE C BROADCAST IS DETECTED" << shape << " " << input.get_shape() << std::endl;
                            return true;
                        }
                    }
                } else {
                    return false;
                }
            }
        }
        return false;
    };
    // FIXME: check if non-4 or 5 dimension, since no blocking in this case anyway
    // if (!hasBroadcastByC())
    //     pushDesc(mkldnn::memory::nChw8c, memory::f32);
    // pushDesc(mkldnn::memory::nchw, memory::f32);

    auto initDesc = [&] (LayoutType lt) -> PrimitiveDescInfo {
        auto createMemoryDesc = [lt](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
            if (lt == ChannelsFirst) {
                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order;
                order.push_back(0);
                for (size_t j = 2; j < blocks.size(); j++)
                    order.push_back(j);
                if (blocks.size() > 1)
                    order.push_back(1);

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            } else if (lt == Blocked && edge->getDims()[1] != 1) {
                size_t blockSize = /*mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::avx512_common) ? 16 :*/ 8;

                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                for (size_t j = 0; j < order.size(); j++)
                    order[j] = j;

                blocks[1] = div_up(blocks[1], blockSize);
                blocks.push_back(blockSize);
                order.push_back(1);

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            } else {
                std::vector<size_t> blocks = edge->getDims().ToSizeVector();
                std::vector<size_t> order(blocks.size());
                for (size_t j = 0; j < order.size(); j++)
                    order[j] = j;

                return MKLDNNMemoryDesc(TensorDesc(prc, edge->getDims().ToSizeVector(), {blocks, order, offset}));
            }
        };

        size_t offset = std::numeric_limits<size_t>::max();
        InferenceEngine::LayerConfig config;
            config.dynBatchSupport = false;

        // config.inConfs.resize(this->inDims.size());
        // config.outConfs.resize(this->outDims.size());
        // config.dynBatchSupport = getChildEdgeAt(0)->getDims().ndims() > 1 && getChildEdgeAt(0)->getDims() == getParentEdgeAt(0)->getDims();

        for (auto k = 0; k < this->inDims.size(); k++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;//(!i && canBeInPlace() && inputPrecisions[i] == outputPrecision) ? 0 : -1;
            dataConfig.constant = false;

            dataConfig.desc = createMemoryDesc(getParentEdgesAtPort(k)[0], /*inputPrecisions[i]*/Precision(Precision::FP32), offset);

            config.inConfs.push_back(dataConfig);
        }

        for (auto k = 0; k < this->outDims.size(); k++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;

            dataConfig.desc = createMemoryDesc(getChildEdgeAt(k), Precision(Precision::FP32), offset);

            config.outConfs.push_back(dataConfig);
        }

        return {config, impl_desc_type::unknown, MKLDNNMemoryDesc(config.outConfs[0].desc).getFormat()};
        // supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    bool isChannelsFirstApplicable = mkldnn::impl::utils::one_of(getChildEdgeAt(0)->getDims().ndims(), 1, 2, 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isChannelsFirstApplicable = isChannelsFirstApplicable && mkldnn::impl::utils::one_of(getParentEdgeAt(i)->getDims().ndims(), 1, 2, 4, 5);
        isChannelsFirstApplicable = isChannelsFirstApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
    }

    bool isBlockedApplicable = mkldnn::impl::utils::one_of(getChildEdgeAt(0)->getDims().ndims(), 4, 5);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        isBlockedApplicable = isBlockedApplicable && mkldnn::impl::utils::one_of(getParentEdgeAt(i)->getDims().ndims(), 4, 5);
        isBlockedApplicable = isBlockedApplicable && getChildEdgeAt(0)->getDims().ndims() == getParentEdgeAt(i)->getDims().ndims();
    }

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable && !hasBroadcastByC())
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}
#endif

void MKLDNNPlugin::MKLDNNSnippetNode::createPrimitive() {
    std::cout << "createPrimitive " << this->getName() << std::endl;
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
        std::cout << "input_shapes!!!!!! " << edge->getDesc().getPrecision() << std::endl;
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::Type(ngraph::element::f32)
            : ngraph::element::Type(ngraph::element::undefined);
        return std::make_tuple(shape, blocking, precision);
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
        std::cout << "output_shapes!!!!!! " << edge->getDesc().getPrecision() << std::endl;
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return std::make_tuple(shape, blocking, precision);
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
