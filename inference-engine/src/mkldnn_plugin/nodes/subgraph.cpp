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

#include "transformations/snippets/subgraph.hpp"
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
        snippet->set_generator(std::make_shared<CPUGenerator>());
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
        auto div_up = [](const int a, const int b) -> int {
            if (!b)
                return 0;
            return (a + b - 1) / b;
        };
        auto createMemoryDesc = [lt, div_up](MKLDNNEdgePtr edge, Precision prc, size_t offset) -> TensorDesc {
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
    // std::cout << "create pritimive: plugin part" << std::endl;
    // auto config = getSelectedPrimitiveDescriptor()->getConfig();

    // auto initDims = [this, config](size_t maxInputSize) {
    //     size_t inputNum = getParentEdges().size();

    //     dims_in.resize(inputNum);
    //     for (int i = 0; i < inputNum; i++) {
    //         dims_in[i].resize(maxInputSize, 1);
    //     }

    //     dims_out.resize(maxInputSize, 1);

    //     std::vector<size_t> order(maxInputSize);
    //     auto outOrder = config.outConfs[0].desc.getBlockingDesc().getOrder();
    //     for (size_t i = 0; i < order.size(); i++) {
    //         if (i < order.size() - outOrder.size())
    //             order[i] = i;
    //         else
    //             order[i] = outOrder[i - (order.size() - outOrder.size())] + (order.size() - outOrder.size());
    //     }

    //     size_t outRank = config.outConfs[0].desc.getBlockingDesc().getBlockDims().size();
    //     for (int i = 0; i < outRank; i++) {
    //         dims_out[dims_out.size() - 1 - i] = config.outConfs[0].desc.getBlockingDesc().getBlockDims()[outRank - 1 - i];
    //     }

    //     for (int i = 0; i < inputNum; i++) {
    //         size_t inRank = config.inConfs[i].desc.getBlockingDesc().getBlockDims().size();

    //         // WA to normalize blocked and planar layouts
    //         auto inOrder = config.inConfs[i].desc.getBlockingDesc().getOrder();
    //         size_t startOff = outOrder.size() != config.outConfs[0].desc.getDims().size() &&
    //                           outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

    //         for (int j = 0; j < inRank; j++) {
    //             dims_in[i][dims_in[i].size() - 1 - j - startOff] = config.inConfs[i].desc.getBlockingDesc().getBlockDims()[inRank - 1 - j];
    //         }
    //     }

    //     for (int i = 0; i < dims_in.size(); i++) {
    //         for (int j = 0; j < dims_in[i].size(); j++) {
    //             if (dims_in[i][j] != dims_out[j] && dims_in[i][j] != 1)
    //                 THROW_IE_EXCEPTION << "Subgraph node with name `" << getName()
    //                                    << "` has invalid input/output dims configuration.";
    //         }
    //     }
    // };

    // auto initOffsets = [this, config](size_t maxInputSize) {
    //     auto offset_out_calc = [](std::vector<size_t>& offset, std::vector<size_t>& dims) -> void{
    //         int k = 1;
    //         for (int i = offset.size() - 1; i >= 0; i--) {
    //             offset[i] = k;
    //             k *= dims[i];
    //         }
    //     };

    //     auto offset_in_calc = [](std::vector<size_t>& offset, std::vector<size_t>& dims_in,
    //                              std::vector<size_t>& dims_out) -> void {
    //         int k = 1;
    //         for (int i = offset.size() - 1; i >= 0; i--) {
    //             offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
    //             k *= dims_in[i];
    //         }
    //     };

    //     size_t inputNum = getParentEdges().size();

    //     offsets_out.resize(maxInputSize, 1);
    //     offset_out_calc(offsets_out, dims_out);
    //     for (int j = 0; j < maxInputSize; j++) {
    //         offsets_out[j] *= config.outConfs[0].desc.getPrecision().size();
    //     }

    //     offsets_in.resize(inputNum);
    //     for (int i = 0; i < inputNum; i++) {
    //         offsets_in[i].resize(maxInputSize, 1);
    //         offset_in_calc(offsets_in[i], dims_in[i], dims_out);
    //         for (int j = 0; j < maxInputSize; j++) {
    //             offsets_in[i][j] *= config.inConfs[i].desc.getPrecision().size();
    //         }
    //     }

    //     start_offset_in.resize(inputNum);
    //     for (size_t i = 0; i < inputNum; i++) {
    //         start_offset_in[i] = getParentEdgeAt(i)->getMemory().GetDescriptor().data.offset0 *
    //                            MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getParentEdgeAt(i)->getMemory().GetDescriptor().data.data_type));
    //     }
    //     start_offset_out = getChildEdgeAt(0)->getMemory().GetDescriptor().data.offset0 *
    //                      MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type));
    // };

    // auto collapseLastDims = [](std::vector<size_t>& dims, int dimsToCollapse) {
    //     for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
    //         dims[dims.size() - 1] *= dims[i];
    //     }

    //     for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
    //         dims[i] = dims[i - dimsToCollapse];
    //     }

    //     for (int i = dimsToCollapse - 1; i >= 0; i--) {
    //         dims[i] = 1;
    //     }
    // };

    // // auto collapseLastOffsets = [](std::vector<size_t>& dims, int dimsToCollapse) {
    // //     for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
    // //         if (dims[dims.size() - 1] > 0 || dims[i] > 0)
    // //             dims[dims.size() - 1] = std::max(dims[dims.size() - 1], static_cast<size_t>(1)) * std::max(dims[i], static_cast<size_t>(1));
    // //         else
    // //             dims[dims.size() - 1] *= dims[i];
    // //     }

    // //     for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
    // //         dims[i] = dims[i - dimsToCollapse];
    // //     }

    // //     for (int i = dimsToCollapse - 1; i >= 0; i--) {
    // //         dims[i] = 0;
    // //     }
    // // };

    // tensorRank = std::max(static_cast<size_t>(optimalTensorRank), config.outConfs[0].desc.getBlockingDesc().getBlockDims().size());
    // initDims(tensorRank);

    // auto outOrder = config.outConfs[0].desc.getBlockingDesc().getOrder();
    // size_t oc_size = 0;
    // // offsets_oc.resize(tensorRank, 0);
    // // if (isFusedWith(Quantize)) {
    // //     size_t offset_oc = 1;
    // //     for (int i = outOrder.size() - 1; i >= 0; i--) {
    // //         if (outOrder[i] == 1) {
    // //             int oc_dim_idx = i + (tensorRank - outOrder.size());
    // //             offsets_oc[oc_dim_idx] = offset_oc;
    // //             offset_oc *= dims_out[oc_dim_idx];
    // //         }
    // //     }
    // //     oc_size = offsets_oc[dims_out.size() - 1] != 0 ? dims_out[dims_out.size() - 1] : 1;
    // // }

    // fullWorkAmount = 1;
    // for (int i = 0; i < dims_out.size(); i++) {
    //     fullWorkAmount *= dims_out[i];
    // }

    // isDynBatchEnabled = config.dynBatchSupport;

    // size_t minimalConcurrency = parallel_get_max_threads();
    // size_t minimalJitWorkAmount = 256;
    // size_t currentJitWorkAmount = dims_out[dims_out.size() - 1];
    // int collapsedDims = 0;
    // if (canUseOptimizedImpl) {
    //     bool hasDifferentDims = false;
    //     while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
    //            // we shouldn't collapse batch dimension in case dynamic batch is enabled
    //            (!isDynBatchEnabled || (config.outConfs[0].desc.getBlockingDesc().getBlockDims().size() - collapsedDims > 2))) {
    //         if (dims_out.size() - collapsedDims - 2 < 0)
    //             break;

    //         for (int j = 1; j < dims_in.size(); j++) {
    //             if (dims_in[j][dims_in[j].size() - 1] != dims_in[0][dims_in[0].size() - 1]) {
    //                 hasDifferentDims = true;
    //             }
    //         }

    //         if (oc_size > 1 && oc_size != dims_in[0][dims_in[0].size() - 1]) {
    //             hasDifferentDims = true;
    //         }

    //         bool canCollapse = true;
    //         for (int i = 0; i < dims_in.size(); i++) {
    //             if (dims_in[i][dims_in[i].size() - 2] != 1) {
    //                 if (dims_in[i][dims_in[i].size() - 1] == 1) {
    //                     canCollapse = false;
    //                     break;
    //                 }

    //                 if (hasDifferentDims) {
    //                     canCollapse = false;
    //                     break;
    //                 }
    //             }
    //         }

    //         if (!canCollapse) {
    //             break;
    //         }

    //         size_t nextJitWorkAmount = currentJitWorkAmount * dims_out[dims_out.size() - 2];
    //         if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
    //             currentJitWorkAmount = nextJitWorkAmount;
    //             collapsedDims++;

    //             for (int i = 0; i < dims_in.size(); i++) {
    //                 collapseLastDims(dims_in[i], 1);
    //             }
    //             collapseLastDims(dims_out, 1);

    //             // if (isFusedWith(Quantize)) {
    //             //     collapseLastOffsets(offsets_oc, 1);
    //             // }
    //         } else {
    //             break;
    //         }
    //     }
    // }

    // batchDimIdx = tensorRank - config.outConfs[0].desc.getBlockingDesc().getBlockDims().size() + collapsedDims;
    // schedulerWorkAmount = fullWorkAmount / dims_out[dims_out.size() - 1];

    // initOffsets(tensorRank);

    // jep.inputs_number = config.inConfs.size();
    // jep.input_size = tensorRank;

    // for (int i = 0; i < config.inConfs.size(); i++) {
    //     jep.src_size[i] = dims_in[i][dims_in[i].size() - 1];
    //     jep.src_prc[i] = config.inConfs[i].desc.getPrecision();
    // }
    // jep.dst_size = dims_out[dims_out.size() - 1];
    // jep.dst_prc = config.outConfs[0].desc.getPrecision();

    // for (int i = 0; i < config.inConfs.size(); i++) {
    //     jep.src_offsets[i] = offsets_in[i];
    // }
    // jep.dst_offsets = offsets_out;

    // jep.oc_size = oc_size;

    // if (mayiuse(x64::avx512_common)) {
    //     eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::avx512_common>(jep, *this));
    // } else if (mayiuse(x64::avx2)) {
    //     eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::avx2>(jep, *this));
    // } else if (mayiuse(x64::sse41)) {
    //     eltwise_kernel.reset(new jit_uni_eltwise_generic<x64::sse41>(jep, *this));
    // }

    // if (eltwise_kernel)
    //     eltwise_kernel->create_ker();


    // std::cout << "createPrimitive: snippets part" << this->getName() << std::endl;
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
        // std::cout << "input_shapes!!!!!! " << edge->getDesc().getPrecision() << std::endl;
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::Type(ngraph::element::f32)
            : ngraph::element::Type(ngraph::element::undefined);
        return std::make_tuple(shape, blocking, precision);
    });


    std::vector<MKLDNNEdgePtr> output_first_row;
    for (size_t i = 0; i < outDims.size(); i++) {
        auto edges = getChildEdgesAtPort(i);
        // Can it go with difference shape or presision to different edges? I assume no.
        output_first_row.push_back(edges[0]);

        // for (auto& edge : edges) {
        //     // remark(11) << "child " << i << " " << edge->getDesc().getLayout() << std::endl;
        //     // print_dims(edge->getDesc().getDims());
        //     // print_dims(edge->getDesc().getBlockingDesc().getBlockDims());
        //     // print_dims(edge->getDesc().getBlockingDesc().getOrder());
        // }
    }

    ngraph::op::Subgraph::BlockedShapeVector output_shapes;
    std::transform(output_first_row.begin(), output_first_row.end(), std::back_inserter(output_shapes),
                [](const MKLDNNEdgePtr& edge) -> ngraph::op::Subgraph::BlockedShape {
        ngraph::Shape shape(edge->getDesc().getBlockingDesc().getBlockDims());
        ngraph::AxisVector blocking(edge->getDesc().getBlockingDesc().getOrder());
        // std::cout << "output_shapes!!!!!! " << edge->getDesc().getPrecision() << std::endl;
        ngraph::element::Type precision = (edge->getDesc().getPrecision() == Precision::FP32) ? ngraph::element::f32 : ngraph::element::undefined;
        return std::make_tuple(shape, blocking, precision);
    });

    schedule = snippet->generate(output_shapes, input_shapes);
    // snippet->print();
}

// FIXME: it seems that scheduler is something plugin dependent so it might be better to call generated code out from here,
// or pass executor functor to
void MKLDNNPlugin::MKLDNNSnippetNode::execute(mkldnn::stream strm) {
    auto& subgraph = snippet;
    // throw 1;

    ngraph::HostTensorVector inputs;
    auto params = subgraph->get_body()->get_parameters();
    for (size_t i = 0; i < inDims.size(); i++) {
        auto & parents = getParentEdgesAtPort(i);
        IE_ASSERT(parents.size() == 1);
        auto &mem = parents[0]->getMemory();
        auto type = subgraph->input(i).get_element_type();
        inputs.push_back(std::make_shared<ngraph::HostTensor>(type, params[i]->get_shape(), mem.GetPtr()));
    }

    ngraph::HostTensorVector outputs;
    auto results = subgraph->get_body()->get_results();
    for (size_t i = 0; i < outDims.size(); i++) {
        auto & child = getChildEdgesAtPort(i);
        auto &mem = child[0]->getMemory();
        auto type = subgraph->output(i).get_element_type();
        outputs.push_back(std::make_shared<ngraph::HostTensor>(type, results[i]->get_shape(), mem.GetPtr()));
    }

    // FIXME: strided tensor support for concat
    if (schedule.ptr == nullptr) {
        subgraph->evaluate(outputs, inputs);
    } else {
        evaluate(outputs, inputs);
    }
}

bool MKLDNNPlugin::MKLDNNSnippetNode::created() const {
    return getType() == Subgraph;
}

bool MKLDNNPlugin::MKLDNNSnippetNode::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    // if (!m_generator) {
    //     return m_body->evaluate(outputs, inputs);
    // }
    // return true;
    // std::cout << "We are evaluating " << inputs.size() << " -> " << outputs.size() << std::endl;

    // make codegen here just as an example;
    // if (ptr == nullptr) {
    //     std::cout << "Warning: generation is done during execution time" << std::endl;
    //     if (!generate()) {
    //         throw ngraph_error("Code generation failed!");
    //     }
    // }

    union param {
        float* ptr;
        size_t len;
    };

    std::array<param, 8> args;

    // if (inputs.size()+outputs.size()+m_constants.size() > args.size()-1)
    //     throw ngraph_error("Too much parameters for snippet. Up to 7 is expected");

    auto work_size = schedule.work_size;
    size_t in_size = inputs.size();
    size_t out_size = outputs.size();
    // size_t const_size = 0;//m_constants.size();

    // FixMe: linearization conflicts with post increment generation logic for now...
    if (false && schedule.is_flat) {
        for (size_t i = 0; i < in_size; i++) {
            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr());
        }

        for (size_t i = 0; i < out_size; i++) {
            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr());
        }

        args[in_size+out_size].len = ngraph::shape_size(work_size);

        // for (size_t i = 0; i < const_size; i++) {
        //     args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>());
        // }

        typedef void (*ker)(const void *);
        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
        k(&args[0]);
    } else if (work_size.size() <= 4) {
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& w) -> std::array<size_t, 4> {
            size_t h = (p[2] != w[2] ? 0 : p[3]);
            size_t c = (p[1] != w[1] ? 0 : p[3]*p[2]);
            size_t n = (p[0] != w[0] ? 0 : p[3]*p[2]*p[1]);
            return std::array<size_t, 4> {1, n, c, h};
        };

        std::vector<std::array<size_t, 4>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        std::vector<std::array<size_t, 4>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        // std::vector<std::array<size_t, 4>> constant_shapes;
        // std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
        //     [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
        //     auto paramShape = tensor->get_shape();
        //     return deduce_strides(paramShape, workSize);
        // });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                    for (size_t i = 0; i < in_size; i++) {
                        auto paramShape = in_shapes[i];
                        args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    for (size_t i = 0; i < out_size; i++) {
                        auto paramShape = out_shapes[i];
                        args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    args[in_size+out_size].len = work_size[3];

                    // for (size_t i = 0; i < const_size; i++) {
                    //     auto paramShape = constant_shapes[i];
                    //     args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                    //         + h*paramShape[3]
                    //         + c*paramShape[2]
                    //         + n*paramShape[1];
                    // }

                    typedef void (*ker)(const void *);
                    ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                    k(&args[0]);
                }
            }
        }
    } else if (work_size.size() == 5) {
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& ws) -> std::array<size_t, 5> {
            size_t w = (p[3] != ws[3] ? 0 : p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[4]*p[3]*p[2]*p[1]);

            // std::cout << ws << " " << p << std::endl;
            // std::cout << n << " " << c << " " << h << " " << w << std::endl;
            return std::array<size_t, 5> {1, n, c, h, w};
        };

        // std::cout << "in" << std::endl;
        std::vector<std::array<size_t, 5>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        // std::cout << "out" << std::endl;
        std::vector<std::array<size_t, 5>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        // std::cout << "const " << m_constants.size() << std::endl;
        // std::vector<std::array<size_t, 5>> constant_shapes;
        // std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
        //     [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
        //     auto paramShape = tensor->get_shape();
        //     return deduce_strides(paramShape, workSize);
        // });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                        for (size_t i = 0; i < in_size; i++) {
                            auto paramShape = in_shapes[i];
                            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        for (size_t i = 0; i < out_size; i++) {
                            auto paramShape = out_shapes[i];
                            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        args[in_size+out_size].len = work_size[4];

                        // for (size_t i = 0; i < const_size; i++) {
                        //     auto paramShape = constant_shapes[i];
                        //     args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                        //         + w*paramShape[4]
                        //         + h*paramShape[3]
                        //         + c*paramShape[2]
                        //         + n*paramShape[1];
                        // }

                        typedef void (*ker)(const void *);
                        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                        k(&args[0]);
                    }
                }
            }
        }
    } else {
        // std::cout << "!!!!!!!!workSize = " << workSize << std::endl;
        auto deduce_strides = [](const ngraph::Shape& p, const ngraph::Shape& ws) -> std::array<size_t, 6> {
            size_t v = (p[4] != ws[4] ? 0 : p[5]);
            size_t w = (p[3] != ws[3] ? 0 : p[5]*p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[5]*p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[5]*p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[5]*p[4]*p[3]*p[2]*p[1]);

            // std::cout << ws << " " << p << std::endl;
            // std::cout << n << " " << c << " " << h << " " << w << " " << v << std::endl;
            return std::array<size_t, 6> {1, n, c, h, w, v};
        };

        // std::cout << "in" << std::endl;
        std::vector<std::array<size_t, 6>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        // std::cout << "out" << std::endl;
        std::vector<std::array<size_t, 6>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [work_size, deduce_strides](const ngraph::HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, work_size);
        });

        // std::cout << "const " << m_constants.size() << std::endl;
        // std::vector<std::array<size_t, 6>> constant_shapes;
        // std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
        //     [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
        //     auto paramShape = tensor->get_shape();
        //     return deduce_strides(paramShape, workSize);
        // });

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        for (size_t v = 0; v < work_size[4]; v++) {
                            // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                            for (size_t i = 0; i < in_size; i++) {
                                auto paramShape = in_shapes[i];
                                args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            for (size_t i = 0; i < out_size; i++) {
                                auto paramShape = out_shapes[i];
                                args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            args[in_size+out_size].len = work_size[5];

                            // for (size_t i = 0; i < const_size; i++) {
                            //     auto paramShape = constant_shapes[i];
                            //     args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                            //         + v*paramShape[5]
                            //         + w*paramShape[4]
                            //         + h*paramShape[3]
                            //         + c*paramShape[2]
                            //         + n*paramShape[1];
                            // }

                            typedef void (*ker)(const void *);
                            ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(schedule.ptr));
                            k(&args[0]);
                        }
                    }
                }
            }
        }
    }

    return true;
}

REG_MKLDNN_PRIM_FOR(MKLDNNSnippetNode, Subgraph);
