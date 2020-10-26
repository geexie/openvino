// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace snippet {

class NGRAPH_API Scalar : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Scalar", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Scalar(const Output<Node>& x);
    Scalar() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
};

} // namespace snippet
using snippet::Scalar;
} // namespace op
} // namespace ngraph