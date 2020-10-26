// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <iterator>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "snippets/register_info.hpp"

namespace ngraph {

std::vector<int> RegisterInfo::getRegisterInfo() const {
    return writes_to;
}

std::shared_ptr<ngraph::Variant> VariantWrapper<RegisterInfo>::merge(const ngraph::NodeVector & nodes) {
    auto isSnippetable = [](const std::shared_ptr<Node> & node) -> bool {
        if (std::dynamic_pointer_cast<ngraph::opset1::Parameter>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::Result>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::Add>(node)) {
            return true;
        }
        return false;
    };

    std::vector<int> all_registers;

    for (auto &node : nodes) {
        if (isSnippetable(node)) {
            auto regs = getRegisterInfo(node);
            std::copy(regs.begin(), regs.end(), all_registers.begin());
        }
    }

    return std::make_shared<VariantWrapper<RegisterInfo> >(RegisterInfo(all_registers));
}

std::shared_ptr<ngraph::Variant> VariantWrapper<RegisterInfo>::init(const std::shared_ptr<ngraph::Node> & node) {
    throw ngraph_error(std::string(type_info.name) + " has no default initialization.");
}

std::vector<int> getRegisterInfo(const std::shared_ptr<ngraph::Node> &node) {
    const auto &rtInfo = node->get_rt_info();

    if (!rtInfo.count(VariantWrapper<RegisterInfo>::type_info.name)) {
        return std::vector<int>();
    }

    const auto &attr = rtInfo.at(VariantWrapper<RegisterInfo>::type_info.name);
    RegisterInfo pp = as_type_ptr<VariantWrapper<RegisterInfo>>(attr)->get();
    return pp.getRegisterInfo();
}


template <typename T>
ngraph::VariantImpl<T>::~VariantImpl() { }

template class ngraph::VariantImpl<RegisterInfo>;

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<RegisterInfo>::type_info;

}  // namespace ngraph
