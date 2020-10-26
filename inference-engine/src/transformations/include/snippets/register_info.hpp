// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <transformations_visibility.hpp>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

namespace ngraph {

class TRANSFORMATIONS_API RegisterInfo {
private:
    std::vector<int> writes_to;

public:
    RegisterInfo() = default;

    explicit RegisterInfo(const int reg_idx) {
        writes_to.push_back(reg_idx);
    }

    explicit RegisterInfo(std::vector<int> new_regs) {
        std::copy(new_regs.begin(), new_regs.end(), writes_to.begin());
    }

    std::vector<int> getRegisterInfo() const;
};

template<>
class TRANSFORMATIONS_API VariantWrapper<RegisterInfo> : public VariantImpl<RegisterInfo> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::CodeGenAttribute::RegisterInfo", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
};

TRANSFORMATIONS_API std::vector<int> getRegisterInfo(const std::shared_ptr<ngraph::Node> & node);

}  // namespace ngraph
