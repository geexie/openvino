// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/register_info.hpp"

template class ngraph::VariantImpl<std::vector<size_t>>;

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<std::vector<size_t>>::type_info;

namespace ngraph {

template class ngraph::VariantImpl<AxisVector>;

constexpr VariantTypeInfo VariantWrapper<AxisVector>::type_info;

std::shared_ptr<ngraph::Variant> VariantWrapper<AxisVector>::merge(const ngraph::NodeVector & nodes) {
    // auto isConvolutionBased = [](const std::shared_ptr<Node> & node) -> bool {
    //     if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
    //         std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
    //         std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
    //         std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node) ||
    //         std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node) ||
    //         std::dynamic_pointer_cast<ngraph::op::DeconvolutionIE>(node)) {
    //         return true;
    //     }
    //     return false;
    // };

    // std::set<std::string> unique_pp;

    // for (auto &node : nodes) {
    //     if (isConvolutionBased(node)) {
    //         std::string pp = getPrimitivesPriority(node);
    //         if (!pp.empty()) unique_pp.insert(pp);
    //     }
    // }

    // if (unique_pp.size() > 1) {
    //     throw ngraph_error(std::string(type_info.name) + " no rule defined for multiple values.");
    // }

    // std::string final_primitives_priority;
    // if (unique_pp.size() == 1) {
    //     final_primitives_priority = *unique_pp.begin();
    // }
    return std::make_shared<VariantWrapper<AxisVector> >(AxisVector(/*final_primitives_priority*/));
}

std::shared_ptr<ngraph::Variant> VariantWrapper<AxisVector>::init(const std::shared_ptr<ngraph::Node> & node) {
    throw ngraph_error(std::string(type_info.name) + " has no default initialization.");
}

AxisVector getResultBlocking(const std::shared_ptr<ngraph::Node> &node) {
    // const auto &rtInfo = node->get_rt_info();
    // using PrimitivesPriorityWraper = VariantWrapper<AxisVector>;

    // if (!rtInfo.count(PrimitivesPriorityWraper::type_info.name)) return {};

    // const auto &attr = rtInfo.at(PrimitivesPriorityWraper::type_info.name);
    // AxisVector pp = as_type_ptr<PrimitivesPriorityWraper>(attr)->get();
    return {};//pp;
}

}  // namespace ngraph