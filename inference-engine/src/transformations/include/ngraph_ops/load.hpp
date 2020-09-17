// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API Load : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Load", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Load(const Output<Node>& x);
    Load() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
};

class TRANSFORMATIONS_API ScalarLoad : public Load {
public:
    static constexpr NodeTypeInfo type_info{"ScalarLoad", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ScalarLoad(const Output<Node>& x);
    ScalarLoad() = default;
private:
};


} // namespace op
} // namespace ngraph