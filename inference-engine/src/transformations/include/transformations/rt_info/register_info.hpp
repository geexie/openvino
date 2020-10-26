// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/axis_vector.hpp>

namespace ngraph {

// extern template class TRANSFORMATIONS_API VariantImpl<std::vector<size_t>>;

template <>
class TRANSFORMATIONS_API VariantWrapper<std::vector<size_t>> : public VariantImpl<std::vector<size_t>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RegInfo|Variant::RuntimeAttribute::AxisVector", 0};

    const VariantTypeInfo& get_type_info() const override { return type_info; }
    VariantWrapper(const value_type& value)
        : VariantImpl<value_type>(value) {
    }
};

// template<>
// class TRANSFORMATIONS_API VariantWrapper<AxisVector> : public VariantImpl<AxisVector> {
// public:
//     static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::AxisVector", 0};

//     const VariantTypeInfo &get_type_info() const override {
//         return type_info;
//     }

//     VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

//     std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

//     std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
// };

TRANSFORMATIONS_API AxisVector getResultBlocking(const std::shared_ptr<ngraph::Node> & node);

} // namespace ngraph