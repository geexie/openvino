// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/load.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Load::type_info;
constexpr NodeTypeInfo op::ScalarLoad::type_info;
constexpr NodeTypeInfo op::VectorLoad::type_info;
constexpr NodeTypeInfo op::BlockedLoad::type_info;
constexpr NodeTypeInfo op::BlockedParameter::type_info;

op::Load::Load(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

op::ScalarLoad::ScalarLoad(const Output<Node>& x) : Load({x}) {
}

op::VectorLoad::VectorLoad(const Output<Node>& x) : Load({x}) {
}

op::BlockedLoad::BlockedLoad(const Output<Node>& x) : Load({x}) {
}

bool op::Load::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> op::Load::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Load>(new_args.at(0));
}

void op::Load::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::Load::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    // std::cout << "called evaluate for snippets::Load "
    //           << input_values[0]->get_shape()
    //           << " (" << this->input(0).get_shape() << ")"
    //           << " -> "
    //           << output_values[0]->get_shape()
    //           << " (" << this->output(0).get_shape() << ")" << std::endl;

    // number of inputs and outputs match
    if (input_values.size() != this->inputs().size()) {
        throw ngraph_error("Load::evaluate wrong input config");
    }

    if (output_values.size() != this->outputs().size()) {
        throw ngraph_error("Load::evaluate wrong input config");
    }

    if (input_values.size() != output_values.size() || input_values.size() != 1) {
        throw ngraph_error("Load::evaluate must be 1->1 operation");
    }

    // shapes match
    if (input_values[0]->get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Load::evaluate input and output must have same shape");
    }

    if (this->output(0).get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Load::evaluate output vector must have the same shape as output port ");
    }

    if (this->input(0).get_shape() != input_values[0]->get_shape()) {
        throw ngraph_error("Load::evaluate input and output must have same shape");
    }

    std::copy(input_values[0]->get_data_ptr<uint8_t>(),
        input_values[0]->get_data_ptr<uint8_t>() + shape_size(get_output_shape(0))*output_values[0]->get_element_type().size(),
        output_values[0]->get_data_ptr<uint8_t>());

    return true;
}

constexpr NodeTypeInfo op::Store::type_info;
constexpr NodeTypeInfo op::ScalarStore::type_info;
constexpr NodeTypeInfo op::VectorStore::type_info;

op::Store::Store(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

op::ScalarStore::ScalarStore(const Output<Node>& x) : Store({x}) {
}

op::VectorStore::VectorStore(const Output<Node>& x) : Store({x}) {
}

bool op::Store::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> op::Store::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Store>(new_args.at(0));
}

void op::Store::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::Store::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    // std::cout << "called evaluate for snippets::Load "
    //           << input_values[0]->get_shape()
    //           << " (" << this->input(0).get_shape() << ")"
    //           << " -> "
    //           << output_values[0]->get_shape()
    //           << " (" << this->output(0).get_shape() << ")" << std::endl;

    // number of inputs and outputs match
    if (input_values.size() != this->inputs().size()) {
        throw ngraph_error("Store::evaluate wrong input config");
    }

    if (output_values.size() != this->outputs().size()) {
        throw ngraph_error("Store::evaluate wrong input config");
    }

    if (input_values.size() != output_values.size() || input_values.size() != 1) {
        throw ngraph_error("Store::evaluate must be 1->1 operation");
    }

    // shapes match
    if (input_values[0]->get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate input and output must have same shape");
    }

    if (this->output(0).get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate output vector must have the same shape as output port ");
    }

    if (this->input(0).get_shape() != input_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate input and output must have same shape");
    }

    std::copy(input_values[0]->get_data_ptr<uint8_t>(),
        input_values[0]->get_data_ptr<uint8_t>() + shape_size(get_output_shape(0))*output_values[0]->get_element_type().size(),
        output_values[0]->get_data_ptr<uint8_t>());

    return true;
}

constexpr NodeTypeInfo op::LEA::type_info;

op::LEA::LEA(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

void op::LEA::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::LEA::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    // std::cout << "called evaluate for snippets::Load "
    //           << input_values[0]->get_shape()
    //           << " (" << this->input(0).get_shape() << ")"
    //           << " -> "
    //           << output_values[0]->get_shape()
    //           << " (" << this->output(0).get_shape() << ")" << std::endl;

    // number of inputs and outputs match
    if (input_values.size() != this->inputs().size()) {
        throw ngraph_error("Store::evaluate wrong input config");
    }

    if (output_values.size() != this->outputs().size()) {
        throw ngraph_error("Store::evaluate wrong input config");
    }

    if (input_values.size() != output_values.size() || input_values.size() != 1) {
        throw ngraph_error("Store::evaluate must be 1->1 operation");
    }

    // shapes match
    if (input_values[0]->get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate input and output must have same shape");
    }

    if (this->output(0).get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate output vector must have the same shape as output port ");
    }

    if (this->input(0).get_shape() != input_values[0]->get_shape()) {
        throw ngraph_error("Store::evaluate input and output must have same shape");
    }

    std::copy(input_values[0]->get_data_ptr<uint8_t>(),
        input_values[0]->get_data_ptr<uint8_t>() + shape_size(get_output_shape(0))*output_values[0]->get_element_type().size(),
        output_values[0]->get_data_ptr<uint8_t>());

    return true;
}

constexpr NodeTypeInfo op::PowerStatic::type_info;
