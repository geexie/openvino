// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/subgraph.hpp"

#include <algorithm>
#include <memory>
#include <array>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>

#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/insert_explisit_fakebroadcast_pass.hpp"
#include "transformations/snippets/insert_explisit_fakebroadcast_pass.hpp"
#include "transformations/snippets/pull_up_fakebroadcast_pass.hpp"
#include "transformations/snippets/insert_explicit_loads_pass.hpp"
#include "transformations/snippets/merge_load_fakebroadcast_pass.hpp"

using namespace std;
using namespace ngraph;

static void visualize(const std::string& name, std::shared_ptr<ngraph::Function>& f) {
    ngraph::pass::VisualizeTree(name).run_on_function(f);
}

void op::Subgraph::set_generator(std::shared_ptr<Generator> generator) {
    m_generator = generator;
}

constexpr NodeTypeInfo op::Subgraph::type_info;

op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<Function> body)
    : Op(args), m_body(body), m_generator(nullptr) {
    constructor_validate_and_infer_types();
}

op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<Function> body)
    : Subgraph(as_output_vector(args), body) {}

std::shared_ptr<Node> op::Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    // FIXME: it should be clone for a body which keeps original operations friendly names
    return make_shared<Subgraph>(inputs, m_body);
}

void op::Subgraph::validate_and_infer_types() {
    // Go over all inputs in the node and replace parameters in m_body with new shape/type
    // FIXME: Check if shape/type is changed before replacement?
    for (size_t i = 0; i < get_input_size(); ++i) {
        m_body->replace_parameter(i, std::make_shared<Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    m_body->validate_nodes_and_infer_types();

    // Go over all outputs and update shape/type from m_body
    set_output_size(m_body->get_output_size());

    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, m_body->get_output_element_type(i), m_body->get_output_partial_shape(i));
    }

    // if (m_body->get_output_size() > 1) {
    //     std::cout << "subgraph " << this->get_friendly_name() << " " << *this << " has multiple outputs" << std::endl;

    //     for (auto op : this->m_body->get_ordered_ops()) {
    //         std::cout << "  " << op->get_friendly_name() << " " << op << std::endl;
    //     }
    // }

    // cleanup codegen state
    ptr = nullptr;
}

bool op::Subgraph::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

auto op::Subgraph::wrap_node_as_subgraph(const std::shared_ptr<ngraph::Node>& node) -> std::shared_ptr<ngraph::op::Subgraph> {
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

// We also can think of canonization as of pass to copy original subgraph and transforming it to canonical form suitable for code generation
// pass actual parameters and results shapes to generate for as well as channel mapping,
// we need to distinguish between 5d tensors that represents <N, C, H, W, c> and <N, C, D, H, W> somehow like locked dimensions
// ngraph::AxisVector to code
//
// Dunamic dimension like <N, C, H, W> = <?, ?, ?, ?> or <N, C, H, W> = <?, ?, ?, W> means that we can merge the consecutive and linearise
// <N, C, H, W> = <?> or <N, C, H, W> = <?, W> folding consecutive dimensions
bool op::Subgraph::generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes) {
    return false;
    // FIXME: check if types are compatible, as well for quantized topologies
    // NODE_VALIDATION_CHECK(this, input_shapes.size() != m_body->get_parameters().size(),
        // "number of parameters for snippet doesn't much passed to generate method: ", input_shapes.size(), ").");


    // canonization
    for (auto& shape : input_shapes) {
        remark(1) << shape << std::endl;
    }
    for (auto& shape : output_shapes) {
        remark(1) << shape << std::endl;
    }

    if (output_shapes.size() != m_body->get_results().size()) {
        throw ngraph::ngraph_error("number of results for snippet doesn't much passed to generate method");
    }

    if (input_shapes.size() != m_body->get_parameters().size()) {
        throw ngraph::ngraph_error("number of parameters for snippet doesn't much passed to generate method");
    }

    // FixMe: check that if input is blocked output is also blocked, if not we should map Result node with a type conversion
    // Is it better to modify storing rather than loading???

    // it should be in subgraph node to be aligned with internal and external parameter list, but adding this for testing
    for (int i = 0; i < m_body->get_parameters().size(); i++) {
        auto param = m_body->get_parameters()[i];

        // FixMe: it's better to collapse not required dimensions rather than introduce some artificial,
        // this will make linearization more natural
        if (param->get_shape().size() < 4) {
            std::vector<size_t> shape(4, 1);
            std::copy(param->get_shape().begin(), param->get_shape().end(), &shape.at(4 - param->get_shape().size()) );

            remark(1) << "parameter" << i << " shape " << param->get_shape() << " reshaping to " << ngraph::Shape(shape) << std::endl;

            m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(param->get_element_type(), ngraph::Shape(shape)));
        } else if (param->get_shape().size() >= 4) {
            if (param->get_element_type() != std::get<2>(input_shapes[i])) {
                std::cout << this->get_friendly_name() << " " << param->get_element_type() << " " << std::get<2>(input_shapes[i]) << std::endl;
                throw ngraph::ngraph_error("changes in presision. Is it legal??");
            }

            m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(std::get<2>(input_shapes[i]), std::get<0>(input_shapes[i])));
            remark(1) << "parameter" << i << " shape " << param->get_shape() << " reshaping to " << std::get<0>(input_shapes[i]) << std::endl;
        }
    }

    // reshape constants to 4d as well
    // FixMe: fix bloking reshape, may be it's better to pass them as parameters, if they are not Scalars????
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = as_type_ptr<opset1::Constant>(op)) {
            // scalars will be replaced after
            if (constant->get_shape().size() < 4 && constant->get_shape() != Shape()) {
                std::vector<size_t> shape(4, 1);
                std::copy(constant->get_shape().begin(), constant->get_shape().end(), &shape.at(4 - constant->get_shape().size()) );
                remark(1) << "constant" << " shape " << constant->get_shape() << " reshaping to " << Shape(shape) << std::endl;
                auto values = constant->get_data_ptr();
                auto new_constant = std::make_shared<opset1::Constant>(constant->get_element_type(), Shape(shape), values);
                ngraph::replace_node(constant, new_constant);
                new_constant->set_friendly_name(constant->get_friendly_name());
                ngraph::copy_runtime_info(constant, new_constant);
            }
        }
    }

    m_body->validate_nodes_and_infer_types();

    // FixMe: at least check that on putput we got that is expected
    // assume blocking is done only by C dimesion. It seems that we need to insert AxisVector to every tensor to support true blocking
    for (int i = 0; i < m_body->get_results().size(); i++) {
        auto result = m_body->get_results()[i];
        PartialShape partial(result->get_shape());

        remark(1) << "result" << i << " shape " << result->get_shape() << " while requested " << std::get<0>(output_shapes[i]) << std::endl;

        bool isCompatible = ngraph::PartialShape::broadcast_merge_into(partial, std::get<0>(output_shapes[i]), op::AutoBroadcastSpec::NUMPY);
        remark(1) << "result" << i << " isCompatible = " << isCompatible << " " << partial << std::endl;

        // indeed, in this case some layout transform should be placed
        if (!isCompatible) {
            auto axises = std::get<1>(output_shapes[i]);
            auto shape = std::get<0>(output_shapes[i]);

            // unblocking
            if (partial.rank() == shape.size()+1) {
                std::cout << "most likely this is blocking" << partial.rank() << " " << result->get_input_node_shared_ptr(0) << std::endl;

                auto newShape = std::vector<uint64_t>({0, 1, 4, 2, 3}/*{shape[0], shape[1]/8, 8, shape[2], shape[3]}*/);
                auto config = std::make_shared<opset1::Constant>(element::u64, Shape{5}, newShape);

                auto transpose = std::make_shared<opset1::Transpose>(result->get_input_node_shared_ptr(0), config);
                result->set_argument(0, transpose);
                result->validate_and_infer_types();
                PartialShape partial(result->get_shape());

                std::cout << transpose->get_output_shape(0) << " " << result->get_output_shape(0)
                << " " << ngraph::PartialShape::broadcast_merge_into(partial, std::get<0>(output_shapes[i]), op::AutoBroadcastSpec::NUMPY)
                << " " << partial << std::endl;
                // auto reshape = std::make_shared<opset1::Reshape>(result->get_input_node_shared_ptr(0), std::make_shared<opset1::Constant>(
                //     element::Type(element::f32), Shape(), {shape[0], shape[1]/8, 8, shape[2], shape[3]}
                // ));
            // blocking
            } else if (partial.rank() == shape.size()) {
                auto newShape = std::vector<uint64_t>({shape[0], shape[1], shape[4], shape[2], shape[3]});
                auto config = std::make_shared<opset1::Constant>(element::u64, Shape{5}, newShape);
                auto reshape = std::make_shared<opset1::Reshape>(result->get_input_node_shared_ptr(0), config, false);
                std::cout << "insert type conversion here " << axises << reshape->get_output_shape(0) << std::endl;

                newShape = std::vector<uint64_t>({0, 1, 3, 4, 2}/*{shape[0], shape[1]/8, 8, shape[2], shape[3]}*/);
                config = std::make_shared<opset1::Constant>(element::u64, Shape{5}, newShape);

                auto transpose = std::make_shared<opset1::Transpose>(reshape, config);
                result->set_argument(0, transpose);
                result->validate_and_infer_types();

                std::cout << transpose->get_output_shape(0) << " " << result->get_output_shape(0)
                << " " << ngraph::PartialShape::broadcast_merge_into(partial, std::get<0>(output_shapes[i]), op::AutoBroadcastSpec::NUMPY)
                << " " << partial << std::endl;
            } else {
                std::cout << "insert type conversion here " << axises << std::endl;
                throw ngraph::ngraph_error("resulting shape is not what is expected. Is it legal??");
            }
        }
    }

    // return true;

    // visualize("0_initial.dot", m_body);
    // adds explicit broadcasts if needed
    // ToDO: this indeed make model not reshapable, need to come up with more clever way to insert fake broadcast,
    // well on the other hand, if we replace scalar constant with Scalar op / or ShapeOf, we could have broadcasts that are reshapable
    ngraph::pass::InsertExplicitFakeBroadcastPass().run_on_function(m_body);

    // Legalization on correct canonical form, disable autobroadcast on operation itself
    for (auto op : m_body->get_ordered_ops()) {
        if (auto binary = std::dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(op)) {
            bool is_scalar = false;
            for (auto input : binary->inputs()) {
                if (input.get_shape() == Shape() || ngraph::shape_size(input.get_shape()) == 1) {
                    is_scalar = true;
                }
            }

            if (!is_scalar)
                binary->set_autob(op::AutoBroadcastSpec::NONE);
        } else if (auto binary = std::dynamic_pointer_cast<op::util::BinaryElementwiseComparison>(op)) {
            bool is_scalar = false;
            for (auto input : binary->inputs()) {
                if (input.get_shape() == Shape() || ngraph::shape_size(input.get_shape()) == 1) {
                    is_scalar = true;
                }
            }

            if (!is_scalar)
                binary->set_autob(op::AutoBroadcastSpec::NONE);
        } else if (auto binary = std::dynamic_pointer_cast<op::util::BinaryElementwiseLogical>(op)) {
            bool is_scalar = false;
            for (auto input : binary->inputs()) {
                if (input.get_shape() == Shape() || ngraph::shape_size(input.get_shape()) == 1) {
                    is_scalar = true;
                }
            }

            if (!is_scalar)
                binary->set_autob(op::AutoBroadcastSpec::NONE);
        }
    }


    // visualize("1_InsertExplicitFakeBroadcastPass.dot", f);
    // remark(2) << "InsertExplicitFakeBroadcastPass is done!" << std::endl;

    // FixMe: pull up as FakeBroadcast
    // propagate broadcast up if needed
    // Mostly unused and disabled for now, FixMe: reenable
    // ngraph::pass::PullUpBroadcastsPass().run_on_function(m_body);
    // visualize("2_PullUpBroadcastsPass.dot", m_body);
    // remark(2) << "PullUpBroadcastsPass is done!" << std::endl;

    // adds explicit loads after each marameter
    ngraph::pass::InsertExplicitLoadsPass().run_on_function(m_body);
    // visualize("3_InsertExplicitLoadsPass.dot", m_body);
    remark(2) << "InsertExplicitLoadsPass is done!" << std::endl;

    // merge Load followed with fake broadcast to broadcast load
    ngraph::pass::MergeLoadFakeBroadcastToBroadcastLoadPass().run_on_function(m_body);
    // visualize("4_MergeLoadFakeBroadcastToBroadcastLoadPass.dot", m_body);
    remark(2) << "MergeLoadFakeBroadcastToBroadcastLoadPass is done!" << std::endl;

    // 4th step merge consecutive load+broadcast+constants,replace broadcast with passtrought/nop

    // 5th step possibly eliminate duplicates

    remark(2) << "Done with transformations!!" << std::endl;

    // Old flow
    // ngraph::pass::DetectBroadcastPass().run_on_function(f);
    // visualize("DetectBroadcastPass.dot", f);

    // actual code mission
    if (m_generator != nullptr)
        ptr = m_generator->generate(m_body);

    // collect constants for scheduling
    m_constants.clear();
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = as_type_ptr<opset1::Constant>(op)) {
            if (ngraph::shape_size(constant->get_shape()) != 1 && constant->get_shape() != Shape()) {
                m_constants.push_back(constant);
            }
        }
    }
    remark(1) << "Found " << m_constants.size() << " constants" << std::endl;

    // check resulting shapes are broadcastable to each other so can be scheduled
    work_size = m_body->output(0).get_shape();
    for (int k = 0; k < m_body->get_output_size(); k++) {
        auto shape = m_body->output(k).get_shape();

        if (work_size.size() != shape.size()) {
            throw ngraph_error("number of channels for all outputs of a snippet should match");
        }

        for (int i = 0; i < work_size.size(); i++) {
            if (work_size[i] != shape[i]) {
                if (work_size[i] == 1) {
                    work_size[i] = shape[i];
                } else {
                    throw ngraph_error("incompatible shapes for output graphs");
                }
            }
        }
    }

    // here we should collapse dimensions by, do at least something
    canBeLinearized = true; // fix tile linearization if no broadcast load present

    if (m_body->get_results().size() <= 1) {
        for (auto& p : m_body->get_ops()) {
            if (!!as_type_ptr<ngraph::op::BroadcastLoad>(p) || as_type_ptr<ngraph::op::FakeBroadcast>(p)) {
                canBeLinearized = false;
                break;
            }
        }
    }
    // std::cout << "Remark: tile " << (canBeLinearized ? "can" : "can't") << " be linearized" << std::endl;
    return ptr != nullptr;
}

bool op::Subgraph::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    if (!m_generator) {
        std::cout << "We are evaluating " << inputs.size() << " -> " << outputs.size() << std::endl;
        return m_body->evaluate(outputs, inputs);
    }
    // return true;
    // std::cout << "We are evaluating " << inputs.size() << " -> " << outputs.size() << std::endl;

    // make codegen here just as an example;
    // if (ptr == nullptr) {
    //     std::cout << "Warning: generation is done during execution time" << std::endl;
    //     if (!generate()) {
    //         throw ngraph_error("Code generation failed!");
    //     }
    // }

    union param {
        float* ptr;
        size_t len;
    };

    std::array<param, 8> args;

    // if (inputs.size()+outputs.size()+m_constants.size() > args.size()-1)
    //     throw ngraph_error("Too much parameters for snippet. Up to 7 is expected");

    auto workSize = work_size;
    size_t in_size = inputs.size();
    size_t out_size = outputs.size();
    size_t const_size = m_constants.size();

    // FixMe: linearization conflicts with post increment generation logic for now...
    if (false && canBeLinearized) {
        for (int i = 0; i < in_size; i++) {
            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr());
        }

        for (int i = 0; i < out_size; i++) {
            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr());
        }

        args[in_size+out_size].len = ngraph::shape_size(work_size);

        for (int i = 0; i < const_size; i++) {
            args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>());
        }

        typedef void (*ker)(const void *);
        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(ptr));
        k(&args[0]);
    } else if (workSize.size() <= 4) {
        auto deduce_strides = [](const Shape& p, const Shape& w) -> std::array<size_t, 4> {
            size_t h = (p[2] != w[2] ? 0 : p[3]);
            size_t c = (p[1] != w[1] ? 0 : p[3]*p[2]);
            size_t n = (p[0] != w[0] ? 0 : p[3]*p[2]*p[1]);
            return std::array<size_t, 4> {1, n, c, h};
        };

        std::vector<std::array<size_t, 4>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        std::vector<std::array<size_t, 4>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        std::vector<std::array<size_t, 4>> constant_shapes;
        std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
            [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        for (int n = 0; n < work_size[0]; n++) {
            for (int c = 0; c < work_size[1]; c++) {
                for (int h = 0; h < work_size[2]; h++) {
                    // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                    for (int i = 0; i < in_size; i++) {
                        auto paramShape = in_shapes[i];
                        args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    for (int i = 0; i < out_size; i++) {
                        auto paramShape = out_shapes[i];
                        args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    args[in_size+out_size].len = work_size[3];

                    for (int i = 0; i < const_size; i++) {
                        auto paramShape = constant_shapes[i];
                        args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    typedef void (*ker)(const void *);
                    ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(ptr));
                    k(&args[0]);
                }
            }
        }
    } else if (workSize.size() == 5) {
        auto deduce_strides = [](const Shape& p, const Shape& ws) -> std::array<size_t, 5> {
            size_t w = (p[3] != ws[3] ? 0 : p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[4]*p[3]*p[2]*p[1]);

            // std::cout << ws << " " << p << std::endl;
            // std::cout << n << " " << c << " " << h << " " << w << std::endl;
            return std::array<size_t, 5> {1, n, c, h, w};
        };

        // std::cout << "in" << std::endl;
        std::vector<std::array<size_t, 5>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        // std::cout << "out" << std::endl;
        std::vector<std::array<size_t, 5>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        // std::cout << "const " << m_constants.size() << std::endl;
        std::vector<std::array<size_t, 5>> constant_shapes;
        std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
            [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        for (int n = 0; n < work_size[0]; n++) {
            for (int c = 0; c < work_size[1]; c++) {
                for (int h = 0; h < work_size[2]; h++) {
                    for (int w = 0; w < work_size[3]; w++) {
                        // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                        for (int i = 0; i < in_size; i++) {
                            auto paramShape = in_shapes[i];
                            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        for (int i = 0; i < out_size; i++) {
                            auto paramShape = out_shapes[i];
                            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        args[in_size+out_size].len = work_size[4];

                        for (int i = 0; i < const_size; i++) {
                            auto paramShape = constant_shapes[i];
                            args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        typedef void (*ker)(const void *);
                        ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(ptr));
                        k(&args[0]);
                    }
                }
            }
        }
    } else {
        // std::cout << "!!!!!!!!workSize = " << workSize << std::endl;
        auto deduce_strides = [](const Shape& p, const Shape& ws) -> std::array<size_t, 6> {
            size_t v = (p[4] != ws[4] ? 0 : p[5]);
            size_t w = (p[3] != ws[3] ? 0 : p[5]*p[4]);
            size_t h = (p[2] != ws[2] ? 0 : p[5]*p[4]*p[3]);
            size_t c = (p[1] != ws[1] ? 0 : p[5]*p[4]*p[3]*p[2]);
            size_t n = (p[0] != ws[0] ? 0 : p[5]*p[4]*p[3]*p[2]*p[1]);

            // std::cout << ws << " " << p << std::endl;
            // std::cout << n << " " << c << " " << h << " " << w << " " << v << std::endl;
            return std::array<size_t, 6> {1, n, c, h, w, v};
        };

        // std::cout << "in" << std::endl;
        std::vector<std::array<size_t, 6>> in_shapes;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        // std::cout << "out" << std::endl;
        std::vector<std::array<size_t, 6>> out_shapes;
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_shapes), [workSize, deduce_strides](const HostTensorPtr& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        // std::cout << "const " << m_constants.size() << std::endl;
        std::vector<std::array<size_t, 6>> constant_shapes;
        std::transform(m_constants.begin(), m_constants.end(), std::back_inserter(constant_shapes),
            [workSize, deduce_strides](const std::shared_ptr<opset1::Constant>& tensor){
            auto paramShape = tensor->get_shape();
            return deduce_strides(paramShape, workSize);
        });

        for (int n = 0; n < work_size[0]; n++) {
            for (int c = 0; c < work_size[1]; c++) {
                for (int h = 0; h < work_size[2]; h++) {
                    for (int w = 0; w < work_size[3]; w++) {
                        for (int v = 0; v < work_size[4]; v++) {
                            // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                            for (int i = 0; i < in_size; i++) {
                                auto paramShape = in_shapes[i];
                                args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            for (int i = 0; i < out_size; i++) {
                                auto paramShape = out_shapes[i];
                                args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            args[in_size+out_size].len = work_size[5];

                            for (int i = 0; i < const_size; i++) {
                                auto paramShape = constant_shapes[i];
                                args[in_size+out_size+1+i].ptr = const_cast<float*>(m_constants[i]->get_data_ptr<float>())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            typedef void (*ker)(const void *);
                            ker k = reinterpret_cast<ker>(const_cast<unsigned char*>(ptr));
                            k(&args[0]);
                        }
                    }
                }
            }
        }
    }

    return true;
}

void ngraph::op::Subgraph::print() const {
    remark(13) << "subgraph " << this->get_friendly_name() << " "
        << this->get_type_name()
        << " which contains " << this->get_body()->get_ops().size() << " nodes" << std::endl;

    int qqq = 0;
    for (auto op : this->get_body()->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    for (auto& in : this->inputs()) {
        remark(13) << "  -> " << in.get_source_output().get_node_shared_ptr()->get_friendly_name() << " "
            << in.get_source_output().get_node_shared_ptr() << std::endl;
    }

    for (auto& out : this->outputs()) {
        for (auto& user : out.get_target_inputs()) {
            remark(13) << " <- " << user.get_node()->get_friendly_name() << " "  << user.get_node() << std::endl;
        }
        remark(13) << std::endl;
    }
}

void ngraph::op::Subgraph::print_statistics(bool verbose) {
    auto getNodeInventory = [](std::shared_ptr<ngraph::Node> n) -> size_t {
        size_t total = 0;

        for (auto input : n->inputs()) {
            total += input.get_tensor().size();
        }

        for (auto output : n->outputs()) {
            total += output.get_tensor().size();
        }

        if (auto subgraph = ngraph::as_type_ptr<ngraph::op::Subgraph>(n)) {
            for (auto op : subgraph->get_body()->get_ordered_ops()) {
                if (ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                    total += op->output(0).get_tensor().size();
                }
            }
        }

        return total;
    };

    auto getFunctionInventory = [getNodeInventory](std::shared_ptr<ngraph::Function> f) -> size_t {
        size_t total = 0;
        for (auto op : f->get_ordered_ops()) {
            // Results and parameters are artificially introduced,
            // while Constants are already considered if they are inputs of other operation
            // this should lead to 1:1 inventory for single node operations
            if (!ngraph::as_type_ptr<ngraph::opset1::Parameter>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Result>(op)
             && !ngraph::as_type_ptr<ngraph::opset1::Constant>(op)) {
                total += getNodeInventory(op);
            }
        }
        return total;
    };

    auto countConstants = [](std::shared_ptr<ngraph::Function> f) -> size_t {
        size_t count = 0;
        for (auto op : f->get_ordered_ops()) {
            count += !!ngraph::as_type_ptr<ngraph::opset1::Constant>(op) ? 1 : 0;
        }
        return count;
    };

    auto body = this->get_body();

    std::cout << this->get_friendly_name()
                << ";" << this
                << ";" << body->get_ops().size()
                << ";" << body->get_parameters().size()
                << ";" << body->get_results().size()
                << ";" << countConstants(body)
                << ";" << getFunctionInventory(body)
                << ";" << getNodeInventory(this->shared_from_this()) << std::endl;

    if (verbose) {
        this->print();
    }
}
