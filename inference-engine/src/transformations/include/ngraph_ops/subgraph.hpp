// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>
#include "ngraph/function.hpp"
#include "ngraph/op/op.hpp"

#include "transformations/snippets/generator.hpp"

namespace ngraph {
namespace op {


// \brief An operation that is implemented by a function.
class TRANSFORMATIONS_API Subgraph : public Op {
public:
    using BlockedShape = std::tuple<ngraph::Shape, ngraph::AxisVector, ngraph::element::Type>;
    using BlockedShapeVector = std::vector<BlockedShape>;

    static constexpr NodeTypeInfo type_info{"Subgraph", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Subgraph(const OutputVector& args, std::shared_ptr<Function> body);

    Subgraph(const NodeVector& args, std::shared_ptr<Function> body);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<Function> get_body() const {
        return m_body;
    }

    std::shared_ptr<Generator> get_generator() const {
        return m_generator;
    }

    bool generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes);
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

    /// Set a new body for the op; body needs to satisfy requirements on inputs/outputs
    void set_body(std::shared_ptr<Function> body);

    // plugin sets generator for a snippet to some specific generator.
    // if there is no such generator it evaluates using nGraph references
    // it's going to be replaced with Jitters table later
    void set_generator(std::shared_ptr<Generator> generator);

private:
    std::shared_ptr<Function> m_body;
    std::vector<std::shared_ptr<opset1::Constant>> m_constants;

    Shape work_size;
    bool canBeLinearized {false};

    std::shared_ptr<Generator> m_generator;
    code ptr {nullptr};
};

std::ostream& operator<<(std::ostream& os, const op::Subgraph::BlockedShape& blocked_shape) {
    os << std::get<0>(blocked_shape) << " " << std::get<1>(blocked_shape) << " " << std::get<2>(blocked_shape);
    return os;
}

}  // namespace op
}  // namespace ngraph
