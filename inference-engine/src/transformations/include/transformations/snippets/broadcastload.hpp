// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

/// ====================================================================================================================================================
/// This only generated for broadcasting by W as leas varying dimension for unblocked cases and the second one for blocked
/// ====================================================================================================================================================
class TRANSFORMATIONS_API BroadcastLoad : public Op {
public:
    static constexpr NodeTypeInfo type_info{"BroadcastLoad", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    BroadcastLoad(const Output<Node>& x, Shape output_shape);
    BroadcastLoad() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void set_broadcast_info(const Shape& bct) {
        broadcast_info = bct;
    }

    bool is_broadcast(size_t idx) {
        return broadcast_info[idx] == 1;
    }

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
    Shape output_shape;
    Shape broadcast_info;
};

} // namespace op
} // namespace ngraph