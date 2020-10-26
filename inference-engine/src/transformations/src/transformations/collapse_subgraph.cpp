// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/collapse_subgraph.hpp"
#include "ngraph_ops/subgraph.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph/pass/visualize_tree.hpp>

#include "snippets/remarks.hpp"

template <typename T>
ngraph::OutputVector as_output_vector(const T& args) {
    ngraph::OutputVector output_vector;
    for (auto arg : args) {
        output_vector.push_back(arg);
    }
    return output_vector;
}

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

auto visualize(const std::shared_ptr<ngraph::Function>& body) -> void {
    static std::atomic_int32_t qq {0};
    ngraph::pass::VisualizeTree(std::string("./snippets/subgraph") + std::to_string(qq++) + ".dot").run_on_function(body);
}

auto create_body(std::string name, const ngraph::ResultVector& results, const ngraph::ParameterVector& parameters) -> std::shared_ptr<ngraph::Function> {
    auto body = std::make_shared<ngraph::Function>(results, parameters);
    body->set_friendly_name(name);
    visualize(body);
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
    ngraph::OutputVector external_inputs;
    ngraph::OutputVector internal_inputs;

    auto inputs = node->inputs();
    for (auto input : inputs) {
        auto source_output_node = input.get_source_output().get_node_shared_ptr();
        if (is_scalar_constant(source_output_node)) {
            internal_inputs.push_back(source_output_node->output(0));
        } else {
            external_inputs.push_back(input.get_source_output());
            auto new_parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(new_parameter);
            internal_inputs.push_back(new_parameter->output(0));
        }
    }

    auto body_node = node->copy_with_new_inputs(internal_inputs);

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted node outputs size doesn't much");
    }

    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, external_inputs, body);

    if (subgraph->get_output_size() != body_results.size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

ngraph::pass::CollapseSubgraph::CollapseSubgraph() {
    // Support ops are operations that forses snippet tokenization like complex tensor math (Convolutions) or data permutations
    std::vector<pattern::op::NodePredicate> support_ops {
            // pattern::op::NodePredicate([](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::Convolution>(n); }),
    };

    // for future experiments if we want to start only from specific op
    auto collect_support_ops = [support_ops](const std::shared_ptr<ngraph::Node>& node) -> std::vector<Input<Node>> {
        std::vector<Input<Node>> inputs_in_support_ops;
        auto inputs = node->inputs();
        for (auto input : inputs) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (std::any_of(support_ops.begin(), support_ops.end(), [parent](pattern::op::NodePredicate predicate) {
                return predicate(parent); })) {
                inputs_in_support_ops.push_back(input);
            }
        }
        return inputs_in_support_ops;
    };

    // auto subgraph = std::make_shared<op::Subgraph>(element::f32, Shape{});
    // auto eltwise = std::make_shared<op::Add>(subgraph, std::make_shared<pattern::op::Label>(element::f32, Shape{}));

    ngraph::graph_rewrite_callback new_snippet_callback = [/*subgraph,*/ collect_support_ops](ngraph::pattern::Matcher &m) {
        auto node = m.get_match_root();
        // auto map = m.get_pattern_value_map();
        // auto my_subgraph = map.at(subgraph);

        remark(3) << "Match root "
                   << node->get_friendly_name()
                   << " " << node
                   << " Creating new snippet - no input subgraphs found" << std::endl;

        std::vector<Input<Node>> inputs_in_support_ops = collect_support_ops(node);
        if (!inputs_in_support_ops.empty()) {
            throw ngraph_error("support ops are not implemented");
        }

        auto subgraph = create_new_subgraph_from(node);
        ngraph::replace_node(node, subgraph);

        remark(3) << "Replacement (new) done for: "
                   << subgraph->get_friendly_name()
                   << " with " << subgraph->inputs().size()
                   << " inputs and " << subgraph->outputs().size()
                   << " outputs" << "\n";

        return true;
    };

    ngraph::graph_rewrite_callback continuation_callback = [](ngraph::pattern::Matcher &m) {
        auto node = m.get_match_root();

        remark(3) << "Match root " << node->get_friendly_name() << " " << node << std::endl;

        // inputs that are already subgraphs
        std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
        // clone bodies because we need a rollback if loop is found
        std::map<std::shared_ptr<Node>, std::shared_ptr<ngraph::Function>> clones;
        // prevert from visiting the same subgraph twice if multiple outputs of the same subgraph comes to the current node as inputs
        std::set<ngraph::op::Subgraph*> visited;

        ParameterVector body_parameters;
        OutputVector external_inputs;
        OutputVector internal_inputs;

        // collect new inputs
        auto inputs = node->inputs();
        for (auto input : inputs) {
            auto input_node = input.get_source_output().get_node_shared_ptr();

            if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(input_node)) {
                remark(3) << "processing " << input_node->get_friendly_name() << " "
                    << input_node->get_type_name()
                    << " which contains " << subgraph->get_body()->get_ops().size() << " nodes" << std::endl;

                for (auto& node : subgraph->get_body()->get_ops()) {
                    remark(3) << "  " << node->get_friendly_name() << " (" << node->get_type_name() << ")" << std::endl;
                }

                if (!input_subgraphs.count(input_node)) {
                    input_subgraphs.insert(subgraph);
                    // how is to clone keeping original friendly names of nodes?
                    auto f = ngraph::clone_function(*subgraph->get_body().get());
                    f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                    clones[input_node] = f;
                }

                auto& input_body = clones[input_node];
                if (input_body->get_parameters().size() != subgraph->get_input_size()) {
                    throw ngraph_error("subgraph internal and external parameter lists' size doesn't much");
                }

                const auto& subgraph_parameters = input_body->get_parameters();

                if (visited.count(subgraph.get()) == 0) {
                    for (size_t i = 0; i < subgraph_parameters.size(); ++i) {
                        remark(3) << "Looking for " << subgraph->input_value(i) << std::endl;

                        auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                        if (found != external_inputs.end()) {
                            auto current_input_index = [found]() -> size_t{
                                for (auto& input : found->get_target_inputs()) {
                                    remark(3) << input.get_source_output() << " vs " << *found << std::endl;
                                    if (input.get_source_output() == *found) {
                                        return input.get_index();
                                    }
                                }
                                return 0;
                            }();

                            remark(3) << "replacing " << *found << " " << current_input_index
                            << " with " << body_parameters[current_input_index] << std::endl;
                            // Index supposed to be kept the same for internal and external parameters
                            input_body->replace_parameter(i, body_parameters[current_input_index]);
                        } else {
                            external_inputs.push_back(subgraph->input_value(i));
                            body_parameters.push_back(subgraph_parameters[i]);
                        }
                    }

                    visited.insert(subgraph.get());
                }

                // this is there stitching happens, get result of a copy of a body of currently processed input and put it to the new inputs
                // internal output index == external output index
                size_t source_output_index = input.get_source_output().get_index();
                auto source_result = input_body->get_results()[source_output_index];
                // Result op has a single input
                internal_inputs.push_back(source_result->input_value(0));

                for (auto output : subgraph->outputs()) {
                    remark(3) << "input subgraph output # " << output.get_index() << std::endl;
                    for (auto user : output.get_target_inputs()) {
                        remark(3) << "output user " << user.get_node()->get_friendly_name()
                                    << " (" << user.get_node()->get_type_name() << ")" << std::endl;
                    }
                }
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

        // clone node with new inputs
        auto body_node = node->copy_with_new_inputs(internal_inputs);
        remark(3) << "Original node outputs = " << node->get_output_size()
                   << " body node outputs = " << body_node->get_output_size() << std::endl;;

        // Then collect outputs of body_node
        // Collect outputs for the body. The set of outputs consists of two part: the first part is side consumers
        // of subgraphs that are consumed by the currently constructed subgraph op node; the second part is body_node
        // own outputs.
        ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;
        // Collect the side output of the subgraph op nodes first
        for (auto node_subgraph : input_subgraphs) {
            auto subgraph = as_type_ptr<ngraph::op::Subgraph>(node_subgraph);
            for (auto output : subgraph->outputs()) {
                bool has_side_consumers = false;
                for (auto target_input : output.get_target_inputs()) {
                    // Check if target_input is in a list of considered nodes (all sub-graphs and the node)
                    // suppose there is a consumer TODO: need to worry?
                    auto target_node = target_input.get_node()->shared_from_this();
                    bool is_side_consumer = !input_subgraphs.count(target_node) && target_node != node;
                    if (is_side_consumer) {
                        if (!has_side_consumers) {
                            auto& input_subgraph_body = clones[node_subgraph]; // subgraph->get_body();
                            // Create a new Result op node inside the body
                            // TODO: what about reuse the existing Result op nodes in subgraphs as it is done for Parameters?
                            body_results.push_back(std::make_shared<opset1::Result>(
                                input_subgraph_body->get_results()[output.get_index()]->input_value(0)));
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

        // Then collect outputs of body_node
        for (auto output : node->outputs()) {
            body_results.push_back(std::make_shared<opset1::Result>(body_node->output(output.get_index())));
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            throw ngraph_error("body results and node results size mismatch during subgraph collaps");
        }

        if (node->get_output_size() != body_node->get_output_size()) {
            throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
        }

        auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = build_subgraph(node, external_inputs, body);

        if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
            throw ngraph_error("newly create subgraph doesn't much number of results");
        }

        if (subgraph->inputs().size() + subgraph->get_output_size() > 7) {
            remark(3) << "new subgraph is created. unable to schedule subgraph with "
                       << subgraph->inputs().size() << " inputs and "
                       << subgraph->get_output_size() << " outputs." << std::endl;

            auto single_node_subgraph = create_new_subgraph_from(node);
            ngraph::replace_node(node, single_node_subgraph);

            return true;
        }

        // it should be a check if outputs are broadcastable to each other,
        // FIXME: if not new subgraph should not be introduced
        auto allOutputShapesAreEqual = [](const std::shared_ptr<Node>& node) -> bool {
            auto outShape = node->output(0).get_shape();
            for (size_t i = 1; i < node->get_output_size(); ++i) {
                if (node->output(i).get_shape() != outShape) {
                    return false;
                }
            }
            return true;
        };

        remark(3) << "output shapes are " << (allOutputShapesAreEqual(subgraph) ? "equal" : "not equal") << std::endl;

        // temporal debug output
        {
            int num = 0;
            for (auto& output : subgraph_result_inputs) {
                for (auto& user : output) {
                    remark(3) << "out #" << num << " user " << user.get_node() << std::endl;
                }
                num++;
            }

            for (auto& new_subgraph_input : subgraph->inputs()) {
                remark(3) << "in " <<  new_subgraph_input.get_source_output().get_node() << std::endl;
            }
        }

        // check for loops before replasing
        auto containsLoop = [input_subgraphs](const std::vector<std::set<Input<Node>>>& results, const std::shared_ptr<Node>& node) -> bool {
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

                        for (auto& input_subgraph : input_subgraphs) {
                            for (auto& subgraph_input : input_subgraph->inputs()) {
                                if (subgraph_input.get_source_output().get_node_shared_ptr() == input.get_source_output().get_node_shared_ptr()) {
                                    remark(3) << "Found this node among the following subgraph "
                                               << input_subgraph->get_friendly_name() << " "
                                               << input_subgraph << std::endl;
                                }
                            }
                        }

                        if (containsLoop) {
                        //     throw 1;
                            return true;
                        }
                    }
                }
            }
            return containsLoop;
        };

        if (containsLoop(subgraph_result_inputs, subgraph)) {
            remark(3) << "New subgraph is created due to loop dependency introduced by one of input subgraphs." << std::endl;
            // throw 1;
            auto single_node_subgraph = create_new_subgraph_from(node);
            single_node_subgraph->validate_and_infer_types();
            ngraph::replace_node_update_name(node, single_node_subgraph);

            return true;
        }

        // finally replace with subgraph. Cannot use replace_node because multiple nodes are replaced; make the replacement manually
        // FIXME: other propertie except data dependencies are not transferred does it mean that if we don't do this it wont be an effect on original graph?
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
                    << " outputs" << "\n";

        return true;
    };

    auto isPossible = [](std::shared_ptr<Node> n) -> bool {
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

    auto nodeIsSupported = [](std::shared_ptr<Node> n) -> bool {
        return  !!as_type_ptr<opset1::Add>(n)
             || !!as_type_ptr<opset1::Subtract>(n)
             || !!as_type_ptr<opset1::Negative>(n)
             || !!as_type_ptr<opset1::Multiply>(n)
             || !!as_type_ptr<opset1::Erf>(n)
             || !!as_type_ptr<opset1::Power>(n)
             || !!as_type_ptr<opset1::SquaredDifference>(n)
             || !!as_type_ptr<opset1::Clamp>(n)
             || !!as_type_ptr<opset1::Sigmoid>(n)
             || !!as_type_ptr<opset1::Relu>(n)
            //  || !!as_type_ptr<opset1::Concat>(n) // to add in snippet opset
            //  || !!as_type_ptr<opset1::Reshape>(n) // to add in snippet opset
             || !!as_type_ptr<opset1::Divide>(n);
    };

    std::vector<pattern::op::NodePredicate> continuation_ops {
        [isPossible, nodeIsSupported](std::shared_ptr<Node> n) {
            return nodeIsSupported(n) && isPossible(n);
        }
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{},
        [hasSomeSubgraphInput, nodeIsSupported](std::shared_ptr<Node> n) {
            return nodeIsSupported(n) && !hasSomeSubgraphInput(n);
        });
    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "CollapseSubgraphNew");
    this->add_matcher(m, new_snippet_callback, PassProperty::CHANGE_DYNAMIC_STATE);

    for (auto predicate : continuation_ops) {
        auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, predicate);
        auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "CollapseSubgraphPart");
        this->add_matcher(m, continuation_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
}