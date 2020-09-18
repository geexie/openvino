// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generator.hpp"
#include "transformations/snippets/generate_pass.hpp"
#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/generator.hpp"
#include "transformations/rt_info/register_info.hpp"

#include "ngraph_ops/scalar.hpp"
#include "ngraph_ops/nop.hpp"

#include <ngraph/pass/visualize_tree.hpp>

bool ngraph::pass::GenerateCodePass::run_on_function(std::shared_ptr<Function> func) {
    for (auto n : func->get_ordered_ops()) {
        auto regs = ngraph::snippet::getRegisters(n);

        if (m_generator->jitters.find(n->get_type_info()) != m_generator->jitters.end()) {
            m_generator->jitters[n->get_type_info()](n)->emit(regs.first, regs.second);
            continue;
        }

        remark(12) << "code generation for " << n->get_friendly_name() << "register precure " << regs.first.size() << " -> " << regs.second.size() << std::endl;

        if (auto op = std::dynamic_pointer_cast<op::BroadcastLoad>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Result>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Parameter>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::ScalarStore>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::Store>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::ScalarLoad>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::Load>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::Scalar>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Add>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Subtract>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Negative>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Multiply>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Divide>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Erf>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::PRelu>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Tanh>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::FakeBroadcast>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Power>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::SquaredDifference>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Clamp>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Relu>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Sigmoid>(n)) {
            m_generator->emit(op, regs);
        } else {
            std::cout << "Warning " << std::string("unknown operation ") << n->get_type_info().name << std::endl;
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

    return false;
}
