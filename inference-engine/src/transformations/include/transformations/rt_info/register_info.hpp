// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations_visibility.hpp>
#include <ngraph/variant.hpp>

namespace ngraph {

extern template class TRANSFORMATIONS_API VariantImpl<std::vector<size_t>>;

template <>
class TRANSFORMATIONS_API VariantWrapper<std::vector<size_t>> : public VariantImpl<std::vector<size_t>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::reginfo", 0};

    const VariantTypeInfo& get_type_info() const override { return type_info; }
    VariantWrapper(const value_type& value)
        : VariantImpl<value_type>(value) {
    }
};

} // namespace ngraph