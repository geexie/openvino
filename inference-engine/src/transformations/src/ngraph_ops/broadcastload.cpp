// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/broadcastload.hpp"

#include <ngraph/runtime/host_tensor.hpp>
#include <ngraph/runtime/reference/broadcast.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::BroadcastLoad::type_info;

op::BroadcastLoad::BroadcastLoad(const Output<Node>& x, Shape shape)
: Op({x}), output_shape(shape), broadcast_info(x.get_shape().size(), 0) {
    constructor_validate_and_infer_types();
}

bool op::BroadcastLoad::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

std::shared_ptr<Node> op::BroadcastLoad::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    auto other = std::make_shared<BroadcastLoad>(new_args.at(0), output_shape);
    other->set_broadcast_info(this->broadcast_info);
    return other;
}

void op::BroadcastLoad::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), output_shape);
}

bool op::BroadcastLoad::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    // std::cout << "called evaluate for snippets::BroadcastLoad "
    //           << input_values[0]->get_shape()
    //           << " (" << this->input(0).get_shape() << ")"
    //           << " -> "
    //           << output_values[0]->get_shape()
    //           << " (" << this->output(0).get_shape() << ") "
    //           << broadcast_info<< std::endl;

    // number of inputs and outputs match
    if (input_values.size() != this->inputs().size()) {
        throw ngraph_error("BroadcastLoad::evaluate wrong input config");
    }

    if (output_values.size() != this->outputs().size()) {
        throw ngraph_error("BroadcastLoad::evaluate wrong input config");
    }

    if (input_values.size() != output_values.size() || input_values.size() != 1) {
        throw ngraph_error("BroadcastLoad::evaluate must be 1->1 operation");
    }

    ///////
    if (this->output(0).get_shape() != output_values[0]->get_shape()) {
        throw ngraph_error("BroadcastLoad::evaluate output vector must have the same shape as output port ");
    }

    if (this->input(0).get_shape() != input_values[0]->get_shape()) {
        throw ngraph_error("BroadcastLoad::evaluate input and output must have same shape");
    }

    auto ishape = input_values[0]->get_shape();
    auto oshape = output_values[0]->get_shape();

    if (ishape.size() != oshape.size()) {
        throw ngraph_error("BroadcastLoad::evaluate input and output must have 4-d shape");
    }

    for (int k = 0; k < ishape.size(); k++) {
        if (!((ishape[k] == oshape[k] && broadcast_info[k] == 0)
           || (ishape[k] != oshape[k] && broadcast_info[k] == 1 && ((ishape[k] == 1) != (oshape[k] == 1) ) ))) {
            throw ngraph_error("BroadcastLoad::evaluate incompatible shapes");
        }
    }

    AxisSet broadcast_axes;
    for (int k = 0; k < ishape.size(); k++) {
        if (!((ishape[k] == oshape[k])
           || (ishape[k] != oshape[k] && ((ishape[k] == 1) != (oshape[k] == 1) ) ))) {
            throw ngraph_error("FakeBroadcast::evaluate incompatible shapes");
        }

        if (ishape[k] != oshape[k]) {
            broadcast_axes.insert(k);
        }
    }

    runtime::reference::broadcast(input_values[0]->get_data_ptr<char>(),
                                  output_values[0]->get_data_ptr<char>(),
                                  input_values[0]->get_shape(),
                                  output_values[0]->get_shape(),
                                  broadcast_axes,
                                  sizeof(float));
    return true;
}