// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/subgraph.hpp"

#include <algorithm>
#include <memory>
#include <array>

#include <ngraph/util.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/insert_explisit_fakebroadcast_pass.hpp"
#include "transformations/snippets/insert_explisit_fakebroadcast_pass.hpp"
#include "transformations/snippets/insert_explicit_loads_pass.hpp"
#include "transformations/snippets/merge_load_fakebroadcast_pass.hpp"
#include "transformations/snippets/assign_registers_pass.hpp"

using namespace std;
using namespace ngraph;

// static void visualize(const std::string& name, std::shared_ptr<ngraph::Function>& f) {
//     ngraph::pass::VisualizeTree(name).run_on_function(f);
// }

constexpr NodeTypeInfo op::Subgraph::type_info;

void op::Subgraph::set_generator(std::shared_ptr<Generator> generator) {
    m_generator = generator;
}

op::Subgraph::Subgraph(const OutputVector& args, std::shared_ptr<Function> body)
    : Op(args), m_body(body), m_generator(nullptr) {
    constructor_validate_and_infer_types();
}

op::Subgraph::Subgraph(const NodeVector& args, std::shared_ptr<Function> body)
    : Subgraph(as_output_vector(args), body) {}

std::shared_ptr<Node> op::Subgraph::clone_with_new_inputs(const OutputVector& inputs) const {
    return make_shared<Subgraph>(inputs, ngraph::clone_function(*m_body.get()));
}

void op::Subgraph::validate_and_infer_types() {
    ngraph::ParameterVector old_parameters;
    for (auto op : m_body->get_parameters()) {
        old_parameters.push_back(op);
    }

    // FIXME: Check if shape/type is changed before replacement?
    for (size_t i = 0; i < get_input_size(); ++i) {
        m_body->replace_parameter(i, std::make_shared<Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    m_body->validate_nodes_and_infer_types();

    for (size_t i = 0; i < m_body->get_parameters().size(); i++) {
        m_body->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(m_body->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, m_body->get_output_element_type(i), m_body->get_output_partial_shape(i));
    }

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
            body_parameters.back()->set_friendly_name(source_output.get_node()->get_friendly_name());
            body_inputs.push_back(parameter->output(0));

            subgraph_inputs.push_back(source_output);
        }
    }

    auto body_node = node->copy_with_new_inputs(body_inputs);
    body_node->set_friendly_name(node->get_friendly_name());

    if (node->get_output_size() != body_node->get_output_size()) {
        throw ngraph::ngraph_error("original node outputs size and extracted subgraph node outputs size doesn't much");
    }

    ngraph::ResultVector body_results;
    for (auto output : node->outputs()) {
        body_results.push_back(std::make_shared<ngraph::opset1::Result>(body_node->output(output.get_index())));
    }

    auto body = create_body(node->get_friendly_name(), body_results, body_parameters);
    auto subgraph = build_subgraph(node, subgraph_inputs, body);

    for (size_t i = 0; i < body->get_parameters().size(); i++) {
        body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
    }

    if (subgraph->get_output_size() != body->get_results().size()) {
        throw ngraph::ngraph_error("newly create subgraph doesn't much number of original node results");
    }

    return subgraph;
}

std::shared_ptr<op::Subgraph> op::Subgraph::make_canonical_from_this() {
    ngraph::OutputVector subgraph_node_inputs;
    for (auto input : this->input_values()) {
        subgraph_node_inputs.push_back(input);
    }
    auto new_body = ngraph::clone_function(*this->get_body().get());
    auto snippet = std::make_shared<ngraph::op::Subgraph>(subgraph_node_inputs, new_body);
    ngraph::copy_runtime_info(this->shared_from_this(), snippet);
    snippet->set_friendly_name(this->get_friendly_name());
    snippet->set_generator(this->m_generator);

    return snippet;
}

// We also can think of canonization as of pass to copy original subgraph and transforming it to canonical form suitable for code generation
// pass actual parameters and results shapes to generate for as well as channel mapping,
// we need to distinguish between 5d tensors that represents <N, C, H, W, c> and <N, C, D, H, W> somehow like locked dimensions
// ngraph::AxisVector to code
//
// Dunamic dimension like <N, C, H, W> = <?, ?, ?, ?> or <N, C, H, W> = <?, ?, ?, W> means that we can merge the consecutive and linearise
// <N, C, H, W> = <?> or <N, C, H, W> = <?, W> folding consecutive dimensions
// FIXME: check that if input is blocked output is also blocked, if not we should map Result node with a type conversion
// Is it better to modify storing rather than loading???
// FIXME: it's better to collapse not required dimensions rather than introduce some artificial,
// this will make linearization more natural
// FIXME: at least check that on putput we got that is expected
// assume blocking is done only by C dimesion. It seems that we need to insert AxisVector to every tensor to support true blocking
// this actually true only if input and putput shapes are not the same.
// FIXME: check if types are compatible, as well for quantized topologies
void op::Subgraph::canonicalize(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes) {
    NODE_VALIDATION_CHECK(this, input_shapes.size() == m_body->get_parameters().size(),
        "Number of parameters for snippet doesn't much passed to generate method: ", input_shapes.size(), " vs ", m_body->get_parameters().size(), ".");

    NODE_VALIDATION_CHECK(this, output_shapes.size() == m_body->get_results().size(),
        "number of results for snippet doesn't much passed to generate method: ", output_shapes.size(), " vs ", m_body->get_results().size(), ".");

    for (auto& shape : input_shapes) {
        remark(11) << shape << std::endl;
    }
    for (auto& shape : output_shapes) {
        remark(11) << shape << std::endl;
    }

    // FIXME: replace only constants which are actually should be represented as scalars during code generation and probably move this step a bit later
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = ngraph::as_type_ptr<opset1::Constant>(op)) {
            std::cout << constant << std::endl;
            auto scalar = std::make_shared<op::Scalar>(*constant);
            scalar->set_friendly_name(constant->get_friendly_name());
            ngraph::copy_runtime_info(constant, scalar);
            ngraph::replace_node(constant, scalar);
        }
    }

    // repalace power with power static
    for (auto op : m_body->get_ordered_ops()) {
        if (auto power = ngraph::as_type_ptr<opset1::Power>(op)) {
            std::cout << power << std::endl;
            if (ngraph::as_type_ptr<op::Scalar>(power->input(1).get_node()->shared_from_this())) {
                std::cout << power << " is to be replaced with static variant" << std::endl;
                auto power_static = std::make_shared<op::PowerStatic>(
                    power->input(0).get_source_output(), power->input(1).get_source_output(), power->get_autob());
                power_static->set_friendly_name(power->get_friendly_name());
                ngraph::copy_runtime_info(power, power_static);
                ngraph::replace_node(power, power_static);
            }
        }
    }


    // it should be in subgraph node to be aligned with internal and external parameter list, but adding this for testing
    // FIXME: store blocking into to Parameter's rt_info for future propagation
    for (size_t i = 0; i < m_body->get_parameters().size(); i++) {
        auto param = m_body->get_parameters()[i];
        if (param->get_shape().size() < 4) {
            remark(11) << "param->get_shape().size() = " << param->get_shape().size() <<
             " " << 4 - (param->get_shape().size() == 0 ? 1 : param->get_shape().size()) << std::endl;
            std::vector<size_t> shape(4, 1);
            std::copy(param->get_shape().begin(), param->get_shape().end(), &shape.at(4 - (param->get_shape().size() == 0 ? 1 : param->get_shape().size())) );

            remark(11) << "parameter " << i << " shape " << param->get_shape() << " reshaping to " << ngraph::Shape(shape) << std::endl;

            m_body->replace_parameter(i, std::make_shared<opset1::Parameter>(param->get_element_type(), ngraph::Shape(shape)));
        } else if (param->get_shape().size() >= 4) {
            if (param->get_element_type() != std::get<2>(input_shapes[i])) {
                std::cout << this->get_friendly_name() << " " << param->get_element_type() << " " << std::get<2>(input_shapes[i]) << std::endl;
                throw ngraph::ngraph_error("changes in presision. Is it legal??");
            }

            if (param->get_shape().size() != std::get<0>(input_shapes[i]).size()) {
                m_body->replace_parameter(i, std::make_shared<op::/*Blocked*/Parameter>(std::get<2>(input_shapes[i]), std::get<0>(input_shapes[i])));
                remark(11) << "parameter" << i << " shape " << param->get_shape()
                           << " reshaping to blocked parameter with shape " << std::get<0>(input_shapes[i]) << std::endl;
            }
        }
    }

    m_body->validate_nodes_and_infer_types();

    // FIXME: add `AxisVector` propagation pass.
    for (auto op : m_body->get_ordered_ops()) {
        // propagate rt_info from parents to current node & check compatibility
    }


    for (size_t i = 0; i < m_body->get_results().size(); i++) {
        auto result = m_body->get_results()[i];
        PartialShape partial(result->get_shape());
        remark(11) << "result" << i << " shape " << result->get_shape() << " while requested " << std::get<0>(output_shapes[i]) << std::endl;

        bool isCompatible = ngraph::PartialShape::broadcast_merge_into(partial, std::get<0>(output_shapes[i]), op::AutoBroadcastSpec::NUMPY);
        remark(11) << "result" << i << " isCompatible = " << isCompatible << " " << partial << std::endl;

        // equality check won't pass since we reshape without changes on external snippet edges
        NODE_VALIDATION_CHECK(this, /*result->get_shape() == std::get<0>(output_shapes[i])*/ isCompatible,
            "Inferend and passed results shapes are difference for snippet : ", result->get_shape(), " vs ", std::get<0>(output_shapes[i]), ".");
    }

    #if 0
    // FIXME: it seems that we jit the rock bottom with hackish approach of representing blocking, we cannot distinguish here between
    auto ops = m_body->get_ordered_ops();
    for (auto op : ops) {
        if (ngraph::op::supports_auto_broadcast(op)) {
            auto shape = op->input(0).get_shape();
            bool vector_broadcast = false;
            for (auto input : op->inputs()) {
                if (input.get_shape().size() > 3 && shape[1] != input.get_shape()[1] && ngraph::shape_size(input.get_shape()) != 1) {
                    std::cout << shape << " " << input.get_shape() << std::endl;
                    vector_broadcast = true;
                }
            }

            if (vector_broadcast) {
                NODE_VALIDATION_CHECK(op.get(), op->inputs().size() == 2, "only binary operation for implicit broadcast is supported");

                auto left = op->input(0).get_shape()[1] == 1 ? op->input(0) : op->input(1);
                auto shape = Shape(left.get_shape());
                shape[shape.size()-1] = 1;

                // FIXME: use ShapeOf to keep it dynamic
                auto begin = opset1::Constant::create(element::i64, Shape{shape.size()}, std::vector<int64_t>(shape.size(), 0));
                auto end   = opset1::Constant::create(element::i64, Shape{shape.size()}, shape);
                auto slice = std::make_shared<opset1::StridedSlice>(left.get_source_output(), begin, end, std::vector<int64_t>{0}, std::vector<int64_t>{0});

                auto node = left.get_source_output().get_node_shared_ptr();
                slice->set_friendly_name(node->get_friendly_name()+"_stride");
                ngraph::copy_runtime_info(node, {slice, begin, end});
                left.replace_source_output(slice->output(0));
            }
        }
    }
    m_body->validate_nodes_and_infer_types();
    #endif

    remark(10) << "after canonicalization" << std::endl;
    print();
}

void op::Subgraph::convert_to_snippet_dialect() {
    ngraph::pass::Manager manager;
    // FIXME: add support
    // manager.register_pass<ngraph::pass::DecomposeFakeQuantizePass>();
    // manager.register_pass<ngraph::pass::InsertExplicitLeaPass>();
    manager.register_pass<ngraph::pass::InsertExplicitLoadsPass>();
    // FIXME:
    manager.register_pass<ngraph::pass::InsertExplicitFakeBroadcastPass>();
    manager.register_pass<ngraph::pass::MergeLoadFakeBroadcastToBroadcastLoadPass>();
    manager.run_passes(m_body);
    remark(10) << "after dialect transform" << std::endl;
    print();
}

bool op::Subgraph::generate(const BlockedShapeVector& output_shapes, const BlockedShapeVector& input_shapes) {
    canonicalize(output_shapes, input_shapes);
    convert_to_snippet_dialect();

    // part 2: generation flow
    ngraph::pass::AssignRegistersPass().run_on_function(m_body);

    // actual code emission
    if (m_generator != nullptr)
        ptr = m_generator->generate(m_body);

    // collect constants for scheduling
    m_constants.clear();
    for (auto op : m_body->get_ordered_ops()) {
        if (auto constant = as_type_ptr<opset1::Constant>(op)) {
            if (ngraph::shape_size(constant->get_shape()) != 1 && constant->get_shape() != Shape()) {
                std::cout << constant << std::endl;
                m_constants.push_back(constant);
            }
        }
    }
    remark(11) << "Found " << m_constants.size() << " constants" << std::endl;
    if (m_constants.size() != 0) {
        throw ngraph_error("constant is founded during code generation flow... aborted!");
    }

    // check resulting shapes are broadcastable to each other so can be scheduled
    work_size = m_body->output(0).get_shape();
    for (size_t k = 0; k < m_body->get_output_size(); k++) {
        auto shape = m_body->output(k).get_shape();

        if (work_size.size() != shape.size()) {
            throw ngraph_error("rank for all outputs of a snippet should match");
        }

        for (size_t i = 0; i < work_size.size(); i++) {
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
        for (size_t i = 0; i < in_size; i++) {
            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr());
        }

        for (size_t i = 0; i < out_size; i++) {
            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr());
        }

        args[in_size+out_size].len = ngraph::shape_size(work_size);

        for (size_t i = 0; i < const_size; i++) {
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

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                    for (size_t i = 0; i < in_size; i++) {
                        auto paramShape = in_shapes[i];
                        args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    for (size_t i = 0; i < out_size; i++) {
                        auto paramShape = out_shapes[i];
                        args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                            + h*paramShape[3]
                            + c*paramShape[2]
                            + n*paramShape[1];
                    }

                    args[in_size+out_size].len = work_size[3];

                    for (size_t i = 0; i < const_size; i++) {
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

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                        for (size_t i = 0; i < in_size; i++) {
                            auto paramShape = in_shapes[i];
                            args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        for (size_t i = 0; i < out_size; i++) {
                            auto paramShape = out_shapes[i];
                            args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                + w*paramShape[4]
                                + h*paramShape[3]
                                + c*paramShape[2]
                                + n*paramShape[1];
                        }

                        args[in_size+out_size].len = work_size[4];

                        for (size_t i = 0; i < const_size; i++) {
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

        for (size_t n = 0; n < work_size[0]; n++) {
            for (size_t c = 0; c < work_size[1]; c++) {
                for (size_t h = 0; h < work_size[2]; h++) {
                    for (size_t w = 0; w < work_size[3]; w++) {
                        for (size_t v = 0; v < work_size[4]; v++) {
                            // ToDo generate right increment, so it shouldn't be this complicated compute and most important execute multiple tiles
                            for (size_t i = 0; i < in_size; i++) {
                                auto paramShape = in_shapes[i];
                                args[i].ptr = reinterpret_cast<float*>(inputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            for (size_t i = 0; i < out_size; i++) {
                                auto paramShape = out_shapes[i];
                                args[in_size+i].ptr = reinterpret_cast<float*>(outputs[i]->get_data_ptr())
                                    + v*paramShape[5]
                                    + w*paramShape[4]
                                    + h*paramShape[3]
                                    + c*paramShape[2]
                                    + n*paramShape[1];
                            }

                            args[in_size+out_size].len = work_size[5];

                            for (size_t i = 0; i < const_size; i++) {
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
