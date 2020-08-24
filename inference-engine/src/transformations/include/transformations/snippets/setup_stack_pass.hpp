// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SetupStackTemporalsOffsetPass : public ngraph::pass::FunctionPass {
public:
    SetupStackTemporalsOffsetPass() : FunctionPass() {
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

} // namespace pass

// class TRANSFORMATIONS_API StackInfo {
// private:
//     size_t offset;
//     size_t usage;

// public:
//     StackInfo() = default;

//     explicit StackInfo(size_t offset, size_t usage) : offset(offset), usage(usage) {
//     }

//     std::string getStackInfo() const;
// };

// template <>
// class TRANSFORMATIONS_API VariantWrapper<std::vector<size_t>> : public VariantImpl<std::vector<size_t>> {
// public:
//     static constexpr VariantTypeInfo type_info{"Variant::reginfo", 0};
//     const VariantTypeInfo& get_type_info() const override { return type_info; }
//     VariantWrapper(const value_type& value)
//         : VariantImpl<value_type>(value) {
//     }
// };

} // namespace ngraph