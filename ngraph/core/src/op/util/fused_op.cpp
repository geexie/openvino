//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/util/fused_op.hpp"

#include "ngraph/graph_util.hpp"
// #include "ngraph/op/get_output_element.hpp"

using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

op::util::FusedOp::FusedOp()
    : Op()
{
}

op::util::FusedOp::FusedOp(const OutputVector& args)
    : Op(args)
{
}

void op::util::FusedOp::validate_and_infer_types()
{
    pre_validate_and_infer_types();

    if (!can_decompose_with_partial_shapes() && is_dynamic())
    {
        return;
    }

    auto subgraph_outputs = decompose_op();
    NodeVector nodes;
    for (auto& val : input_values())
        nodes.emplace_back(val.get_node_shared_ptr());
    auto subgraph = extract_subgraph(ngraph::as_node_vector(subgraph_outputs), nodes);

    // FIXME: check if fused op has multiple outputs
    // auto args = get_arguments();
    // decltype(args) arg_nodes;
    // std::transform(std::begin(args), std::end(args), std::back_inserter(arg_nodes), [] (const std::shared_ptr<Node>& arg) {
    //     if (as_type_ptr<ngraph::op::v0::GetOutputElement>(arg)) {
    //         return arg->get_input_node_shared_ptr(0);
    //     } else {
    //         return arg;
    //     }
    // });

    // auto subgraph = extract_subgraph(subgraph_outputs, arg_nodes);

    validate_nodes_and_infer_types(subgraph);

    size_t i = 0;
    for (const auto& output : subgraph_outputs)
    {
        if (i >= get_output_size())
        {
            set_output_size(i + 1);
        }
        set_output_type(i, output.get_element_type(), output.get_shape());
        i++;
    }

    post_validate_and_infer_types();
}
