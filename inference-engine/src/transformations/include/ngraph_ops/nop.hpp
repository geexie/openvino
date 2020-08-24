// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API Nop : public Op {
public:
    static constexpr NodeTypeInfo type_info{"isa-nop", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Nop(const Output<Node>& x);
    Nop() = default;

private:
};

} // namespace op
} // namespace ngraph