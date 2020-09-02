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
            remark(13) << "path found over " << visited.size() << " nodes" << std::endl;
            for (auto& n : visited) {
                remark(13) << "-> " << n << std::endl;
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

auto print_subgraph(const std::shared_ptr<ngraph::op::Subgraph> subgraph) -> void {
    remark(13) << "processing " << subgraph->get_friendly_name() << " "
        << subgraph->get_type_name()
        << " which contains " << subgraph->get_body()->get_ops().size() << " nodes" << std::endl;

    for (auto& node : subgraph->get_body()->get_ops()) {
        remark(13) << "  " << node->get_friendly_name() << " (" << node->get_type_name() << ")" << std::endl;
    }
}

auto print_edge(const ngraph::Output<ngraph::Node>& output) -> void {
    remark(13) << "Looking for " << output.get_node_shared_ptr()->get_friendly_name() << " " << output.get_node_shared_ptr() << " " << output << std::endl;
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

    std::cout << "create_new_subgraph_from !!!!!" << node << std::endl;

    auto inputs = node->inputs();
    for (auto input : inputs) {
        auto source_output_node = input.get_source_output().get_node_shared_ptr();
        std::cout << source_output_node << std::endl;
        if (is_scalar_constant(source_output_node)) {
            internal_inputs.push_back(source_output_node->output(0));
        } else {
            std::cout << "wrapping to parameter" << std::endl;
            external_inputs.push_back(input.get_source_output());
            auto new_parameter = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            body_parameters.push_back(new_parameter);
            internal_inputs.push_back(new_parameter->output(0));
        }
    }

    for (auto ii : internal_inputs) {
        std::cout << ii.get_node_shared_ptr() << std::endl;
    }
    auto body_node = node->copy_with_new_inputs(internal_inputs);

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted node outputs size doesn't much");
    }

    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }
    for (auto br : body_results) {
        std::cout << br << std::endl;
    }
    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);

    std::cout << "body contains " <<  body->get_ops().size() << " ops" << std::endl;

    auto subgraph = build_subgraph(node, external_inputs, body);

    if (subgraph->get_output_size() != body_results.size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

auto collapse_inputs(const std::shared_ptr<ngraph::Node>& node) -> bool {
    using namespace ngraph;
    remark(13) << "Match root " << node->get_friendly_name() << " " << node << std::endl;

    // inputs that are already subgraphs
    std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
    // clone bodies because we need a rollback if loop is found
    std::map<std::shared_ptr<Node>, std::shared_ptr<ngraph::Function>> clones;
    // prevert from visiting the same subgraph twice if multiple outputs of the same subgraph comes to the current node as inputs
    // std::set<ngraph::op::Subgraph*> visited;

    ParameterVector body_parameters;
    OutputVector external_inputs;
    OutputVector internal_inputs;

    // collect new inputs
    auto inputs = node->inputs();

    auto is_recurrent = [inputs](const ngraph::Output<ngraph::Node>& to_find) -> bool {
        for (auto in : inputs) {
            if (in.get_source_output().get_node_shared_ptr() == to_find.get_node_shared_ptr()) {
                std::cout << "WE ARE FOUND OUTSELF " << in.get_source_output().get_node_shared_ptr() << " vs " << to_find.get_node() << std::endl;
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

    // first walk over inputs to collect all parameters
    for (auto input : inputs) {
        auto input_node = input.get_source_output().get_node_shared_ptr();

        if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(input_node)) {
            print_subgraph(subgraph);

            if (!input_subgraphs.count(input_node)) {
                input_subgraphs.insert(input_node);

                // how is to clone keeping original friendly names of nodes?
                // auto f = ngraph::clone_function(*subgraph->get_body().get());
                // f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                // clones[input_node] = f;
                auto f = clones[input_node];

                const auto& input_body_parameters = f->get_parameters();

                for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                    print_edge(subgraph->input_value(i));

                    auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                    if (found != external_inputs.end()) {
                        auto current_input_index = get_input_index(*found);

                        remark(13) << "replacing " << *found << " " << current_input_index << " with " << body_parameters[current_input_index] << std::endl;
                        // Index supposed to be kept the same for internal and external parameters
                        f->replace_parameter(i, body_parameters[current_input_index]);
                    } else if (is_recurrent(subgraph->input_value(i))) {
                        // nothing to do
                        // we should remove this from parameters and link internally only
                        remark(13) << "recurrence is detected to be handled later " << subgraph->input_value(i).get_node_shared_ptr() << std::endl;

                        // const auto& params =  f->get_parameters()[i];
                        auto internal = input_body_parameters[i];
                        auto internal_consumers = internal->outputs();

                        auto node_to_replace = std::make_shared<op::Constant>(internal->get_element_type(), Shape{});

                        // auto qqq = inputs[0].get_source_output().get_node_shared_ptr();
                        for (auto in : inputs) {
                            if (in.get_source_output().get_node_shared_ptr() == subgraph->input_value(i).get_node_shared_ptr()) {
                                std::cout << "WE ARE FOUND OUTSELF " << in.get_source_output().get_node_shared_ptr() << " " << in << " "
                                << subgraph->input_value(i) << " "
                                          << in.get_index() << " " << i << " " << in.get_source_output().get_index() << std::endl;
                                // qqq = in;
                                // return true;
                            }
                        }

                        // for (auto out : subgraph->outputs()) {
                        //     std::cout << out.get_target_inputs() << std::endl
                        // }

                        for (auto output : internal->outputs()) {
                            for (auto consumer : output.get_target_inputs()) {
                                std::cout << consumer.get_node()->shared_from_this()->get_friendly_name()
                                          << consumer.get_node()->shared_from_this() << std::endl;

                                if (auto to_replace_with = as_type_ptr<ngraph::op::Subgraph>(subgraph->input_value(i).get_node_shared_ptr())) {
                                    auto other_body = clones[subgraph->input_value(i).get_node_shared_ptr()];

                                    std::cout << other_body->get_ops().size() << std::endl;
                                    auto other_body_result = other_body->get_results()[consumer.get_source_output().get_index()];
                                    auto result_producer = other_body_result->input(0).get_source_output();

                                    for (auto result : other_body->get_results()) {
                                        std::cout << result << std::endl;
                                    }

                                    std::cout << "HERE WE ARE " << result_producer.get_node_shared_ptr() << " " << other_body_result << std::endl;

                                // if (consumer.get_node()->shared_from_this() == internal) {
                                    // std::cout << "WE ARE WE ARE!! " << internal << std::endl;
                                    std::cout << "replaceing " << consumer.get_source_output().get_index()
                                    << " " << subgraph->input_value(i).get_node_shared_ptr() << subgraph << " !!!!!" << std::endl;
                                    consumer.replace_source_output(result_producer.get_node_shared_ptr());
                                }
                            }
                        //     std::cout << c.get_node_shared_ptr()->get_friendly_name() << c.get_node_shared_ptr() << std::endl;
                        //     std::cout << c.get_target_inputs()()->get_friendly_name() << c.get_node_shared_ptr() << std::endl;
                        }

                        for (auto& node : f->get_ordered_ops()) {
                            // if (node->get_input_node_shared_ptr() == internal)
                        //     // node->revalidate_and_infer_types();

                        //     // If we find a parameter make sure it is in the list of parameters of the function
                        //     if (op::is_parameter(node))
                        //     {
                        //         auto it = std::find(m_parameters.begin(), m_parameters.end(), node);
                        //         if (it == m_parameters.end())
                        //         {
                        //             ngraph::replace_node(node, std::make_shared<op::Constant>(node->get_element_type(), Shape{}));
                        //             //throw ngraph_error("Function references undeclared parameter");
                        //         }
                        //     }
                        }

                        // input_node->input_value(i).get_node_shared_ptr()

                        //

                        // 1 get external parameters to bind
                        // f->replace_parameter(i, other_subgraph_);

                        // here we can find a parameter
                        // auto param = f->get_parameters()[i];
                        // ngraph::replace_node(param,
                        //     std::make_shared<opset1::Constant>(param->get_element_type(), param->get_shape()));
                        // throw 1;
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

            for (auto output : subgraph->outputs()) {
                remark(13) << "input subgraph output # " << output.get_index() << std::endl;
                for (auto user : output.get_target_inputs()) {
                    remark(13) << "output user " << user.get_node()->get_friendly_name()
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
    remark(13) << "Original node outputs = " << node->get_output_size()
                << " body node outputs = " << body_node->get_output_size() << std::endl;


    // auto is_recurrent_out = [](const ngraph::Output<ngraph::Node>& to_find) -> bool {
    //     for (auto in : inputs) {
    //         if (in.get_source_output().get_node_shared_ptr() == to_find.get_node_shared_ptr()) {
    //             std::cout << "WE ARE FOUND OUTSELF " << in.get_source_output().get_node_shared_ptr() << " vs " << to_find.get_node() << std::endl;
    //             return true;
    //         }
    //     }
    //     return false;
    // };


    // 1. subgraph outputs consumed by some other node
    // 2. subgraph outputs consumed by the node to be merged
    // 3. subgraph outputs consumed by other subgraphs which are inputs to the merged node
    ResultVector body_results;
    std::vector<std::set<Input<Node>>> subgraph_result_inputs;

    // we can handle batterfly effect here
    for (auto subgraph : input_subgraphs) {
        for (auto output : subgraph->outputs()) {
            bool has_side_consumers = false;

            for (auto target_input : output.get_target_inputs()) {
                // Check if target_input is in a list of considered nodes (all sub-graphs and the node)
                // suppose there is a consumer TODO: need to worry?
                auto target_node = target_input.get_node()->shared_from_this();

                if (input_subgraphs.count(target_node) && (subgraph != target_node) && (target_node != node)) {
                    std::cout << "WE ARE FOUND RECURRENCE AGAIN "
                        << target_input.get_source_output().get_node_shared_ptr() << " vs " << target_node << " vs " << subgraph << std::endl;

                    // return collapse_inputs(target_node);
                    // if (res) {
                    //     std::cout << "INPUTS ARE COLLAPSED" << std::endl;
                    //     res = collapse_inputs(node);
                    // }
                    // throw 1;
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

    // Then collect outputs of body_node
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<opset1::Result>(body_node->output(output.get_index())));
        subgraph_result_inputs.push_back(output.get_target_inputs());
    }

    if (body_results.size() != subgraph_result_inputs.size()) {
        throw ngraph_error("body results and node results size mismatch during subgraph collaps");
    }

    // if (body_parameters.size() + body_results.size() > 7) {
    //     remark(13) << "new subgraph is created. unable to schedule subgraph with "
    //                << body_parameters.size() << " inputs and "
    //                << body_results.size() << " outputs." << std::endl;

    //     auto single_node_subgraph = create_new_subgraph_from(node);
    //     ngraph::replace_node(node, single_node_subgraph);

    //     return /*true*/true;
    // }

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
    }

    std::cout << "before create_body" << std::endl;
    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, external_inputs, body);

    if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
        throw ngraph_error("newly create subgraph doesn't much number of results");
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

    remark(13) << "output shapes are " << (allOutputShapesAreEqual(subgraph) ? "equal" : "not equal") << std::endl;

    // temporal debug output
    {
        int num = 0;
        for (auto& output : subgraph_result_inputs) {
            for (auto& user : output) {
                remark(13) << "out #" << num << " user " << user.get_node() << std::endl;
            }
            num++;
        }

        for (auto& new_subgraph_input : subgraph->inputs()) {
            remark(13) << "in " <<  new_subgraph_input.get_source_output().get_node() << std::endl;
        }
    }

    // check self recurrence

    // check for loops before replasing
    auto containsLoop = [input_subgraphs](const std::vector<std::set<Input<Node>>>& results, const std::shared_ptr<Node>& node) -> bool {
        bool containsLoop = false;
        for (auto& result : results) {
            for (auto& user : result) {
                for (auto& input : node->inputs()) {
                    auto source = input.get_source_output().get_node();
                    containsLoop |= has_path_from_to(user.get_node(), source);

                    remark(13) <<  "checking path from "
                            << user.get_node()->get_friendly_name()
                            << " to " << source->get_friendly_name()
                            << " resulted in " << containsLoop << std::endl;

                    for (auto& input_subgraph : input_subgraphs) {
                        for (auto& subgraph_input : input_subgraph->inputs()) {
                            if (subgraph_input.get_source_output().get_node_shared_ptr() == input.get_source_output().get_node_shared_ptr()) {
                                remark(13) << "Found this node among the following subgraph "
                                            << input_subgraph->get_friendly_name() << " "
                                            << input_subgraph << std::endl;
                            }
                        }
                    }

                    if (containsLoop) {
                        // throw 1;
                        return true;
                    }
                }
            }
        }
        return containsLoop;
    };

    if (containsLoop(subgraph_result_inputs, subgraph) || (body_parameters.size() + body_results.size() > 7)) {
        remark(13) << "New subgraph is created due to loop dependency introduced by one of input subgraphs." << std::endl;
        // // throw 1;
        auto single_node_subgraph = create_new_subgraph_from(node);
        single_node_subgraph->validate_and_infer_types();
        ngraph::replace_node(node, single_node_subgraph);
        return true;
        // return false;
    }

    // finally replace with subgraph. Cannot use replace_node because multiple nodes are replaced; make the replacement manually
    // FIXME: other propertie except data dependencies are not transferred does it mean that if we don't do this it wont be an effect on original graph?
    for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
        for (auto target_input : subgraph_result_inputs[i]) {
            target_input.replace_source_output(subgraph->output(i));
        }
    }

    subgraph->validate_and_infer_types();

    remark(13) << "Replacement (merge) done for: "
                << subgraph->get_friendly_name()
                << " with " << subgraph->inputs().size()
                << " inputs and " << subgraph->outputs().size()
                << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";

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
    return true;
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

        remark(13) << "Match root "
                   << node->get_friendly_name()
                   << " " << node
                   << " Creating new snippet - no input subgraphs found" << std::endl;

        std::vector<Input<Node>> inputs_in_support_ops = collect_support_ops(node);
        if (!inputs_in_support_ops.empty()) {
            throw ngraph_error("support ops are not implemented");
        }

        auto subgraph = create_new_subgraph_from(node);
        ngraph::replace_node(node, subgraph);

        remark(13) << "Replacement (new) done for: "
                   << subgraph->get_friendly_name()
                   << " with " << subgraph->inputs().size()
                   << " inputs and " << subgraph->outputs().size()
                   << " outputs" << "\n";

        return true;
    };

    ngraph::graph_rewrite_callback continuation_callback = [](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        return collapse_inputs(node);
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
             || !!as_type_ptr<opset1::Relu>(n)//;
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