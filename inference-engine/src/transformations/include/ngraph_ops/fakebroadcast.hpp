// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API FakeBroadcast : public Op {
public:
    static constexpr NodeTypeInfo type_info{"FakeBroadcast", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    FakeBroadcast(const Output<Node>& x, Shape output_shape);
    FakeBroadcast() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    // FIXME: reenable after ShapeOf
    // void set_broadcast_info(const Shape& bct) {
    //     broadcast_info = bct;
    // }

    // bool is_broadcast(size_t idx) {
    //     return broadcast_info[idx] == 1;
    // }

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
    Shape output_shape;
    // Shape broadcast_info;
};

} // namespace op
} // namespace ngraph