// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/scalar.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Scalar::type_info;

op::Scalar::Scalar(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

bool op::Scalar::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> op::Scalar::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Scalar>(new_args.at(0));
}

void op::Scalar::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::Scalar::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    throw ngraph_error("Scalar::evaluate is not implemented");
}