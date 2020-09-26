// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/collapse_subgraph.hpp"
#include "transformations/snippets/remarks.hpp"
#include "ngraph_ops/subgraph.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>

/// ====================================================================================================================================================
/// This pass tokenizes topology graph into subgraphs.
/// Those subgraphs consists of unary or binary layout-oblivious (LO) opetations found in subset 1.
///
/// Non-layout-oblivious (NLO) operations operations (called also support in this context) are ignored and become a fullstop in tokenization routine
///
/// 1. if a considered LO operation doesn't have any unput subgraphs
///     -> a new single-op subgraph is introduced
/// 1. if a considered LO operation is a binary or an unary operation with at least one subgraph as an input
///     -> 1. all inputs from the conput subgraphs are collected together
///        1. non-subgraph inputs are wrapped into parameters
///        1. all input bodies are merged and
///        1. this new operation is added to a body of input subgraph
///        1. outputs are collected subgraph (outputs consumed by some other node & subgraph outputs consumed by the node to be merged)
///        1. finally current node is replaced with the new subgraph. We cannot use replace_node because multiple nodes are replaced so
///        make the replacement manually by redirecting ports
/// Input subgraph is prefented from visiting twice if more than one output of it consumed by currently considered node
/// New subgraph is introduced, if there is a loop introduced
/// New subgraph is introduced, if number of inputs and outputs exceeds 7 due to scheduling limitation
/// New subgraph is introduced, if multiple outputs of merged nodes are not broadcastable to each other (equality of all outputs is too much on the other hand)
/// Scalar constants are placed as is into subgraph due to optimization purpose
///
/// FIXME: other propertie except data dependencies are not transferred does it mean that if we don't do this it wont be an effect on original graph?
/// ====================================================================================================================================================
auto outputs_are_not_broadcastable(const std::shared_ptr<ngraph::Node>& node) -> bool {
    auto outputs = node->outputs();
    auto find_smallest_output_shape = [](const std::vector<ngraph::Output<ngraph::Node>>& outputs) -> ngraph::Shape {
        return std::accumulate(std::begin(outputs), std::end(outputs), ngraph::Shape(outputs.begin()->get_shape()),
            [](ngraph::Shape other_shape, ngraph::Output<ngraph::Node> output){
                return ngraph::shape_size(output.get_shape()) < ngraph::shape_size(other_shape) ? output.get_shape() : other_shape;
            });
    };
    auto ref_shape = find_smallest_output_shape(outputs);

    auto check_shapes_broadcastable = [ref_shape](const ngraph::Output<ngraph::Node>& output) -> bool {
        auto other_shape = output.get_shape();

        if (other_shape.size() != ref_shape.size()) {
            return false;
        }

        return std::inner_product(std::begin(other_shape), std::end(other_shape), std::begin(ref_shape), true,
                            std::logical_and<bool>(), [](ngraph::Shape::value_type lsh, ngraph::Shape::value_type rsh){
                                return rsh == 1 || lsh == rsh;
                            });
    };

    return std::find_if_not(std::begin(outputs), std::end(outputs), check_shapes_broadcastable) != std::end(outputs);
};

auto has_cycles_of_dependencies(const std::vector<std::set<ngraph::Input<ngraph::Node>>>& results,
                                const std::vector<ngraph::Input<ngraph::Node>>& inputs) -> bool {
    auto BFS_from_to = [](ngraph::Node* from, ngraph::Node* to) -> bool {
        std::unordered_set<ngraph::Node*> visited;
        std::queue<ngraph::Node*> stack;
        stack.push(from);

        while (stack.size() > 0) {
            ngraph::Node* curr = stack.front();
            visited.insert(curr);

            if (ngraph::op::is_output(curr)) {
                return false;
            }

            stack.pop();

            if (curr != to) {
                for (const auto& next : curr->get_users()) {
                    if (visited.count(next.get()) == 0) {
                        stack.push(next.get());
                    }
                }
            } else {
                return true;
            }
        }
        return false;
    };

    for (auto& result : results) {
        for (auto& user : result) {
            for (auto& input : inputs) {
                auto source = input.get_source_output().get_node();
                auto containsLoop = BFS_from_to(user.get_node(), source);

                remark(3) <<  "checking path from "
                        << user.get_node()->get_friendly_name()
                        << " to " << source->get_friendly_name()
                        << " resulted in " << containsLoop << std::endl;

                if (containsLoop) {
                    return true;
                }
            }
        }
    }
    return false;
}

ngraph::pass::CollapseSubgraph::CollapseSubgraph(bool tokenize_by_node) : GraphRewrite() {
    ngraph::graph_rewrite_callback continuation_callback = [](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        remark(3) << "Match root " << node->get_friendly_name() << " " << node << std::endl;

        // inputs that are already subgraphs
        std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
        // clone bodies because we need a rollback if loop is found
        std::map<std::shared_ptr<Node>, std::shared_ptr<ngraph::Function>> clones;

        ParameterVector body_parameters;
        OutputVector external_inputs;
        OutputVector internal_inputs;

        auto inputs = node->inputs();

        auto is_recurrent = [inputs](const ngraph::Output<ngraph::Node>& to_find) -> bool {
            for (auto in : inputs) {
                if (in.get_source_output().get_node_shared_ptr() == to_find.get_node_shared_ptr()) {
                    return true;
                }
            }
            return false;
        };

        auto get_input_index = [](const Output<Node>& found) -> size_t {
            for (auto& input : found.get_target_inputs()) {
                remark(3) << input.get_source_output() << " vs " << found << std::endl;
                if (input.get_source_output() == found) {
                    return input.get_index();
                }
            }
            return 0;
        };

        for (auto input : inputs) {
            auto input_node = input.get_source_output().get_node_shared_ptr();

            if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(input_node)) {
                if (!clones.count(input_node)) {
                    auto f = ngraph::clone_function(*subgraph->get_body().get());
                    f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                    clones[input_node] = f;
                }
            }
        }

        for (auto input : inputs) {
            auto input_node = input.get_source_output().get_node_shared_ptr();

            if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(input_node)) {
                // subgraph->print();

                if (!input_subgraphs.count(input_node)) {
                    input_subgraphs.insert(input_node);

                    auto f = clones[input_node];
                    const auto& input_body_parameters = f->get_parameters();

                    for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                        auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                        if (found != external_inputs.end()) {
                            auto current_input_index = get_input_index(*found);
                            remark(3) << "replacing " << *found << " " << current_input_index << " with " << body_parameters[current_input_index] << std::endl;
                            f->replace_parameter(i, body_parameters[current_input_index]);
                        } else if (is_recurrent(subgraph->input_value(i))) {
                            remark(3) << "ternary merge is conducted " << subgraph->input_value(i).get_node_shared_ptr() << std::endl;

                            auto internal = input_body_parameters[i];
                            auto internal_consumers = internal->outputs();

                            for (auto output : internal->outputs()) {
                                for (auto consumer : output.get_target_inputs()) {
                                    if (auto to_replace_with = as_type_ptr<ngraph::op::Subgraph>(subgraph->input_value(i).get_node_shared_ptr())) {
                                        auto other_body = clones[subgraph->input_value(i).get_node_shared_ptr()];
                                        auto other_body_result = other_body->get_results()[consumer.get_source_output().get_index()];
                                        auto result_producer = other_body_result->input(0).get_source_output();

                                        consumer.replace_source_output(result_producer.get_node_shared_ptr());
                                    }
                                }
                            }
                        } else {
                            external_inputs.push_back(subgraph->input_value(i));
                            body_parameters.push_back(input_body_parameters[i]);
                        }
                    }
                }

                // this is there stitching happens, get result of a copy of a body of currently processed input and put it to the new inputs
                // internal output index == external output index
                auto& input_body = clones[input_node];
                size_t source_output_index = input.get_source_output().get_index();
                auto source_result = input_body->get_results()[source_output_index];
                // Result op has a single input
                internal_inputs.push_back(source_result->input_value(0));
            } else {
                if (op::is_scalar_constant(input_node)) {
                    internal_inputs.push_back(input_node->output(0));
                } else {
                    external_inputs.push_back(input.get_source_output());
                    auto new_parameter = std::make_shared<opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                    body_parameters.push_back(new_parameter);
                    internal_inputs.push_back(new_parameter->output(0));
                }
            }
        }

        auto body_node = node->copy_with_new_inputs(internal_inputs);
        remark(3) << "Original node outputs = " << node->get_output_size()
                    << " body node outputs = " << body_node->get_output_size() << std::endl;

        if (node->get_output_size() != body_node->get_output_size()) {
            throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
        }

        ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        for (auto subgraph : input_subgraphs) {
            for (auto output : subgraph->outputs()) {
                bool first_side_consumer = true;

                for (auto target_input : output.get_target_inputs()) {
                    auto target_node = target_input.get_node()->shared_from_this();

                    if (input_subgraphs.count(target_node)) {
                        remark(3) << "ternary merge is conducted " << subgraph << " -> " << target_node << std::endl;
                    }

                    if (!input_subgraphs.count(target_node) && target_node != node) {
                        if (first_side_consumer) {
                            auto& input_subgraph_body = clones[subgraph];
                            body_results.push_back(std::make_shared<opset1::Result>(input_subgraph_body->get_results()[output.get_index()]->input_value(0)));
                            subgraph_result_inputs.push_back({});

                            first_side_consumer = false;
                        }

                        if (!!subgraph_result_inputs.back().count(target_input)) {
                            throw ngraph_error("target input added twice!!!");
                        }
                        // save target input port outside the body
                        subgraph_result_inputs.back().insert(target_input);
                    }
                }
            }
        }

        for (auto output : node->outputs()) {
            body_results.push_back(std::make_shared<opset1::Result>(body_node->output(output.get_index())));
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            throw ngraph_error("body results and node results size mismatch during subgraph collaps");
        }

        if (body_parameters.size() + body_results.size() > 7) {
            remark(3) << "new subgraph is created. Impossible to schedule subgraph with "
                      << body_parameters.size() << " inputs and " << body_results.size() << " outputs." << std::endl;

            auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
            ngraph::replace_node(node, single_node_subgraph);
            return true;
        }

        auto body = op::create_body(node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = op::build_subgraph(node, external_inputs, body);

        if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
            throw ngraph_error("newly create subgraph doesn't much number of results");
        }

        if (outputs_are_not_broadcastable(subgraph)) {
            remark(3) << "New subgraph is created due to outputs of a subgraph not broadcastable." << std::endl;

            auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
            single_node_subgraph->validate_and_infer_types();
            ngraph::replace_node(node, single_node_subgraph);
            return true;
        }

        if (has_cycles_of_dependencies(subgraph_result_inputs, subgraph->inputs())) {
            remark(3) << "New subgraph is created due to loop dependency introduced by one of input subgraphs." << std::endl;

            auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
            single_node_subgraph->validate_and_infer_types();
            ngraph::replace_node(node, single_node_subgraph);
            return true;
        }

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (auto target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        subgraph->validate_and_infer_types();

        remark(3) << "Replacement (merge) done for: "
                    << subgraph->get_friendly_name()
                    << " with " << subgraph->inputs().size()
                    << " inputs and " << subgraph->outputs().size()
                    << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";

        // subgraph->print();
        return true;
    };

    auto hasSomeSubgraphInput = [](std::shared_ptr<Node> node) -> bool {
        auto inputs = node->inputs();
        for (auto input : inputs) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (!!as_type_ptr<ngraph::op::Subgraph>(parent)) {
                return true;
            }
        }
        return false;
    };

    auto is_lob = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::Add>(n)
            || !!as_type_ptr<opset1::Divide>(n)
            || !!as_type_ptr<opset1::Equal>(n)
            || !!as_type_ptr<opset1::FloorMod>(n)
            || !!as_type_ptr<opset1::Greater>(n)
            || !!as_type_ptr<opset1::GreaterEqual>(n)
            || !!as_type_ptr<opset1::Less>(n)
            || !!as_type_ptr<opset1::LessEqual>(n)
            || !!as_type_ptr<opset1::LogicalAnd>(n)
            || !!as_type_ptr<opset1::LogicalOr>(n)
            || !!as_type_ptr<opset1::LogicalXor>(n)
            || !!as_type_ptr<opset1::Maximum>(n)
            || !!as_type_ptr<opset1::Minimum>(n)
            || !!as_type_ptr<opset1::Mod>(n)
            || !!as_type_ptr<opset1::Multiply>(n)
            || !!as_type_ptr<opset1::NotEqual>(n)
            || !!as_type_ptr<opset1::PRelu>(n)
            || !!as_type_ptr<opset1::Power>(n)
            || !!as_type_ptr<opset1::SquaredDifference>(n)
            || !!as_type_ptr<opset1::Subtract>(n);
            // || !!as_type_ptr<opset1::Xor>(n);
    };

    auto is_lou = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::Abs>(n)
            // || !!as_type_ptr<opset1::Acos>(n)
            // || !!as_type_ptr<opset1::Asin>(n)
            // || !!as_type_ptr<opset1::Atan>(n)
            // || !!as_type_ptr<opset1::Ceiling>(n)
            || !!as_type_ptr<opset1::Clamp>(n)
            // || !!as_type_ptr<opset1::Cos>(n)
            // || !!as_type_ptr<opset1::Cosh>(n)
            || !!as_type_ptr<opset1::Elu>(n)
            || !!as_type_ptr<opset1::Erf>(n)
            || !!as_type_ptr<opset1::Exp>(n)
            // || !!as_type_ptr<opset1::Floor>(n)
            // || !!as_type_ptr<opset1::Log>(n)
            || !!as_type_ptr<opset1::LogicalNot>(n)
            || !!as_type_ptr<opset1::Negative>(n)
            || !!as_type_ptr<opset1::Relu>(n)
            // || !!as_type_ptr<opset1::Sign>(n)
            || !!as_type_ptr<opset1::Sigmoid>(n)
            // || !!as_type_ptr<opset1::Sin>(n)
            // || !!as_type_ptr<opset1::Sinh>(n)
            || !!as_type_ptr<opset1::Sqrt>(n)
            // || !!as_type_ptr<opset1::Tan>(n)
            || !!as_type_ptr<opset1::Tanh>(n);
    };

    auto is_lot = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::HardSigmoid>(n) // ternary with 2 constants
            || !!as_type_ptr<opset1::Selu>(n); // ternary with 2 constants / or DW
    };

    auto is_lo = [is_lou, is_lob](std::shared_ptr<Node> n) -> bool { return is_lou(n) || is_lob(n); };

    this->add_matcher(std::make_shared<pattern::Matcher>(
        std::make_shared<pattern::op::Label>(pattern::any_input(),
        [hasSomeSubgraphInput, is_lo, tokenize_by_node](std::shared_ptr<Node> n) {
            return is_lo(n) && (tokenize_by_node || !hasSomeSubgraphInput(n));
        }),
        "CollapseSubgraphNew"),
        [](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        remark(3) << "Match root "
                  << node->get_friendly_name()
                  << " " << node
                  << " Creating new snippet - no input subgraphs found" << std::endl;

        auto subgraph = op::Subgraph::wrap_node_as_subgraph(node);
        ngraph::replace_node(node, subgraph);

        remark(3) << "Replacement (new) done for: "
                  << subgraph->get_friendly_name()
                  << " with " << subgraph->inputs().size()
                  << " inputs and " << subgraph->outputs().size()
                  << " outputs" << "\n";

        return true;
    }, PassProperty::CHANGE_DYNAMIC_STATE);

    this->add_matcher(std::make_shared<pattern::Matcher>(
        std::make_shared<pattern::op::Label>(pattern::any_input(),
        [hasSomeSubgraphInput, is_lo](std::shared_ptr<Node> n) {
        return is_lo(n) && hasSomeSubgraphInput(n);
        }), "CollapseSubgraphPart"),
        continuation_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}