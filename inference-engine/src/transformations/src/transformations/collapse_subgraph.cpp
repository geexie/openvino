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
/// Scalar constants are placed as is into subgraph due to optimization purpose
///
/// FIXME: There is no check to test that multiple outputs can be actually scheduled together.
/// If outputs are not broadcastable to each other, subgraph should not be merged and new subgraph should be introduced,
/// equality of all outputs is on too strict the other hand
///
/// FIXME: other propertie except data dependencies are not transferred does it mean that if we don't do this it wont be an effect on original graph?
/// ====================================================================================================================================================

auto has_path_from_to(ngraph::Node* from, ngraph::Node* to) -> bool {
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
            remark(3) << "path found over " << visited.size() << " nodes" << std::endl;
            for (auto& n : visited) {
                remark(3) << "-> " << n << std::endl;
            }
            return true;
        }
    }

    return false;
}

auto is_scalar_constant(const std::shared_ptr<ngraph::Node>& source_output_node) -> bool {
    return !!ngraph::as_type_ptr<ngraph::opset1::Constant>(source_output_node) &&
        (source_output_node->get_shape() == ngraph::Shape() || ngraph::shape_size(source_output_node->get_shape()) == 1);
};

auto create_body(std::string name, const ngraph::ResultVector& results, const ngraph::ParameterVector& parameters) -> std::shared_ptr<ngraph::Function> {
    auto body = std::make_shared<ngraph::Function>(results, parameters);
    body->set_friendly_name(name);
    return body;
};

auto build_subgraph(const std::shared_ptr<ngraph::Node>& node, const ngraph::OutputVector& inputs, const std::shared_ptr<ngraph::Function>& body)
    -> std::shared_ptr<ngraph::op::Subgraph>{
    auto subgraph = std::make_shared<ngraph::op::Subgraph>(inputs, body);
    copy_runtime_info(node, subgraph);
    subgraph->set_friendly_name(node->get_friendly_name());
    return subgraph;
};

auto create_new_subgraph_from(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<ngraph::op::Subgraph> {
    ngraph::ParameterVector body_parameters;
    ngraph::OutputVector body_inputs;

    ngraph::OutputVector subgraph_inputs;

    for (auto input : node->inputs()) {
        auto source_output = input.get_source_output();
        if (is_scalar_constant(source_output.get_node_shared_ptr())) {
            body_inputs.push_back(source_output);
        } else {
            auto parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(parameter);
            body_inputs.push_back(parameter->output(0));

            subgraph_inputs.push_back(source_output);
        }
    }

    auto body_node = node->copy_with_new_inputs(body_inputs);

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    if (subgraph->get_output_size() != body->get_results().size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

auto visualize(const std::shared_ptr<ngraph::Function>& body) -> void {
    static std::atomic_int32_t qq {0};
    ngraph::pass::VisualizeTree(std::string("./snippets/subgraph") + std::to_string(qq++) + ".dot").run_on_function(body);
}

auto print_subgraph(const std::shared_ptr<ngraph::op::Subgraph> subgraph) -> void {
    remark(13) << "subgraph " << subgraph->get_friendly_name() << " "
        << subgraph->get_type_name()
        << " which contains " << subgraph->get_body()->get_ops().size() << " nodes" << std::endl;

    for (auto& node : subgraph->get_body()->get_ops()) {
        remark(13) << "  " << node->get_friendly_name() << " (" << node->get_type_name() << ")" << std::endl;
    }

    for (auto& in : subgraph->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
            << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : subgraph->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " "  << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

auto print_edge(const ngraph::Output<ngraph::Node>& output) -> void {
    remark(13) << "Looking for " << output.get_node_shared_ptr()->get_friendly_name() << " " << output.get_node_shared_ptr() << " " << output << std::endl;
}

ngraph::pass::CollapseSubgraph::CollapseSubgraph() {
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
                // print_subgraph(subgraph);

                if (!input_subgraphs.count(input_node)) {
                    input_subgraphs.insert(input_node);

                    auto f = clones[input_node];
                    const auto& input_body_parameters = f->get_parameters();

                    for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                        // print_edge(subgraph->input_value(i));

                        auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                        if (found != external_inputs.end()) {
                            auto current_input_index = get_input_index(*found);
                            remark(3) << "replacing " << *found << " " << current_input_index << " with " << body_parameters[current_input_index] << std::endl;
                            f->replace_parameter(i, body_parameters[current_input_index]);
                        } else if (is_recurrent(subgraph->input_value(i))) {
                            remark(3) << "recurrence is detected to be handled later " << subgraph->input_value(i).get_node_shared_ptr() << std::endl;

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
                if (is_scalar_constant(input_node)) {
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


        ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        for (auto subgraph : input_subgraphs) {
            for (auto output : subgraph->outputs()) {
                bool has_side_consumers = false;

                for (auto target_input : output.get_target_inputs()) {
                    // Check if target_input is in a list of considered nodes (all sub-graphs and the node)
                    // suppose there is a consumer TODO: need to worry?
                    auto target_node = target_input.get_node()->shared_from_this();

                    if (input_subgraphs.count(target_node) && (subgraph != target_node) && (target_node != node)) {
                        std::cout << "WE ARE FOUND RECURRENCE AGAIN "
                            << target_input.get_source_output().get_node_shared_ptr()
                            << " vs " << target_node << " vs " << subgraph << std::endl;
                    }

                    bool is_side_consumer = !input_subgraphs.count(target_node) && target_node != node;
                    if (is_side_consumer) {
                        if (!has_side_consumers) {
                            auto& input_subgraph_body = clones[subgraph];
                            // Create a new Result op node inside the body
                            // TODO: what about reuse the existing Result op nodes in subgraphs as it is done for Parameters?
                            body_results.push_back(std::make_shared<opset1::Result>(input_subgraph_body->get_results()[output.get_index()]->input_value(0)));
                            subgraph_result_inputs.push_back({}); // create empty set if inputs
                            has_side_consumers = true;
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
            remark(3) << "new subgraph is created. unable to schedule subgraph with "
                       << body_parameters.size() << " inputs and "
                       << body_results.size() << " outputs." << std::endl;

            auto single_node_subgraph = create_new_subgraph_from(node);
            ngraph::replace_node(node, single_node_subgraph);

            return true;
        }

        if (node->get_output_size() != body_node->get_output_size()) {
            throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
        }

        auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = build_subgraph(node, external_inputs, body);

        if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
            throw ngraph_error("newly create subgraph doesn't much number of results");
        }

        auto outputs_are_not_broadcastable = [](const std::shared_ptr<Node>& node) -> bool {
            auto outShape = node->output(0).get_shape();
            for (size_t i = 1; i < node->get_output_size(); ++i) {
                if (node->output(i).get_shape() != outShape) {
                    return false;
                }
            }
            return true;
        };

        if (outputs_are_not_broadcastable(subgraph)) {
            remark(3) << "New subgraph should be created due to outputs of a subgraph not proadcascable." << std::endl;
            // auto single_node_subgraph = create_new_subgraph_from(node);
            // single_node_subgraph->validate_and_infer_types();
            // ngraph::replace_node(node, single_node_subgraph);
            // return true;
        }

        auto has_cycles_of_dependencies = [](const std::unordered_set<std::shared_ptr<Node>>& input_subgraphs,
                                             const std::vector<std::set<Input<Node>>>& results, const std::shared_ptr<Node>& node) -> bool {
            bool containsLoop = false;
            for (auto& result : results) {
                for (auto& user : result) {
                    for (auto& input : node->inputs()) {
                        auto source = input.get_source_output().get_node();
                        containsLoop |= has_path_from_to(user.get_node(), source);

                        remark(3) <<  "checking path from "
                                << user.get_node()->get_friendly_name()
                                << " to " << source->get_friendly_name()
                                << " resulted in " << containsLoop << std::endl;

                        if (containsLoop) {
                            for (auto& input_subgraph : input_subgraphs) {
                                for (auto& subgraph_input : input_subgraph->inputs()) {
                                    if (subgraph_input.get_source_output().get_node_shared_ptr() == input.get_source_output().get_node_shared_ptr()) {
                                        remark(3) << "Found this node among the following subgraph "
                                                    << input_subgraph->get_friendly_name() << " "
                                                    << input_subgraph << std::endl;
                                    }
                                }
                            }
                            return true;
                        }
                    }
                }
            }
            return containsLoop;
        };

        if (has_cycles_of_dependencies(input_subgraphs, subgraph_result_inputs, subgraph)) {
            remark(3) << "New subgraph is created due to loop dependency introduced by one of input subgraphs." << std::endl;
            auto single_node_subgraph = create_new_subgraph_from(node);
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

        // print_subgraph(subgraph);
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
            || !!as_type_ptr<opset1::Subtract>(n)
            || !!as_type_ptr<opset1::Xor>(n);
    };

    auto is_lou = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::Abs>(n)
            || !!as_type_ptr<opset1::Acos>(n)
            || !!as_type_ptr<opset1::Asin>(n)
            || !!as_type_ptr<opset1::Atan>(n)
            || !!as_type_ptr<opset1::Ceiling>(n)
            || !!as_type_ptr<opset1::Clamp>(n)
            || !!as_type_ptr<opset1::Cos>(n)
            || !!as_type_ptr<opset1::Cosh>(n)
            || !!as_type_ptr<opset1::Elu>(n)
            || !!as_type_ptr<opset1::Erf>(n)
            || !!as_type_ptr<opset1::Exp>(n)
            || !!as_type_ptr<opset1::Floor>(n)
            || !!as_type_ptr<opset1::HardSigmoid>(n)
            || !!as_type_ptr<opset1::Log>(n)
            || !!as_type_ptr<opset1::LogicalNot>(n)
            || !!as_type_ptr<opset1::Negative>(n)
            || !!as_type_ptr<opset1::Relu>(n)
            || !!as_type_ptr<opset1::Selu>(n)
            || !!as_type_ptr<opset1::Sign>(n)
            || !!as_type_ptr<opset1::Sigmoid>(n)
            || !!as_type_ptr<opset1::Sin>(n)
            || !!as_type_ptr<opset1::Sinh>(n)
            || !!as_type_ptr<opset1::Sqrt>(n)
            || !!as_type_ptr<opset1::Tan>(n)
            || !!as_type_ptr<opset1::Tanh>(n);
    };

    auto is_lo = [is_lou, is_lob](std::shared_ptr<Node> n) -> bool { return is_lou(n) || is_lob(n); };

    this->add_matcher(std::make_shared<pattern::Matcher>(
        std::make_shared<pattern::op::Label>(pattern::any_input(),
        [hasSomeSubgraphInput, is_lo](std::shared_ptr<Node> n) {
            return is_lo(n) && !hasSomeSubgraphInput(n);
        }),
        "CollapseSubgraphNew"),
        [](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        remark(3) << "Match root "
                  << node->get_friendly_name()
                  << " " << node
                  << " Creating new snippet - no input subgraphs found" << std::endl;

        auto subgraph = create_new_subgraph_from(node);
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