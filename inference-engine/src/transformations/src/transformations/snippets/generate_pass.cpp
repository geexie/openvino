// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generator.hpp"
#include "transformations/snippets/generate_pass.hpp"
#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/rt_info/register_info.hpp"

#include "ngraph_ops/scalar.hpp"
#include "ngraph_ops/nop.hpp"

#include <ngraph/pass/visualize_tree.hpp>

auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> std::pair<std::vector<size_t>, std::vector<size_t>> {
    auto rt = n->get_rt_info();

    std::vector<size_t> rout;
    if (auto rinfo = rt["reginfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
        for (auto reg : reginfo) {
            rout.push_back(reg);
        }
    }

    std::vector<size_t> rin;
    for (auto input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        if (auto rinfo = rt["reginfo"]) {
            auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
            for (auto reg : reginfo) {
                rin.push_back(reg);
            }
        }
    }
    return std::make_pair(rin, rout);
}

bool ngraph::pass::GenerateCodePass::run_on_function(std::shared_ptr<Function> func) {
#if 0
    // usage
    std::vector<std::shared_ptr<Emitter>> emitters;
    // generation pass
    for (auto n : f->get_ordered_ops()) {
        if (jitters.find(n->get_type_info()) != jitters.end()) { // it shouldn't be here if we have complete
            auto e = jitters[n->get_type_info()](n);
            auto regs = getRegisters(n);
            e->emit(regs.first, regs.second, {});

            emitters.push_back(e);
        } else {
            //throw ngraph_error(std::string("Dear library writer, would you mind to implement another one jitter for ") + n->get_type_info().name);
        }
    }

    // table setup pass
    for (auto& e : emitters) {
        e->emit_table();
    }

    // doesn't go any futher for now...
    throw std::exception();
#endif

    for (auto n : func->get_ordered_ops()) {
        auto regs = getRegisters(n);

        remark(2) << (m_shouldLoadVectors ? "vector " : "scalar ")
            << "code generation for " << n->get_friendly_name() << " of type " << n->get_type_info().name << std::endl;
        remark(2) << "register precure " << regs.first.size() << " -> " << regs.second.size() << std::endl;

        if (auto op = std::dynamic_pointer_cast<opset1::Parameter>(n)) {
            m_generator->emit(op, regs, m_shouldLoadVectors);
        } else if (auto op = std::dynamic_pointer_cast<op::Load>(n)) {
            m_generator->emit(op, regs, m_shouldLoadVectors);
        } else if (auto op = std::dynamic_pointer_cast<op::BroadcastLoad>(n)) {
            m_generator->emit(op, regs, m_shouldLoadVectors);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Result>(n)) {
            m_generator->emit(op, regs, m_shouldLoadVectors);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Constant>(n)) {
            m_generator->emit(op, regs, m_shouldLoadVectors);
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
        } else if (auto op = std::dynamic_pointer_cast<op::FakeBroadcast>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<op::Scalar>(n)) {
            // m_generator->emit(op, regs);
            throw ngraph::ngraph_error(std::string("not implemented operation ") + n->get_type_info().name);
        } else if (auto op = std::dynamic_pointer_cast<op::Nop>(n)) {
            // m_generator->emit(op, regs);
            throw ngraph::ngraph_error(std::string("not implemented operation ") + n->get_type_info().name);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Power>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::SquaredDifference>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Clamp>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Relu>(n)) {
            m_generator->emit(op, regs);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Concat>(n)) { // some limitation on control flow rather than actual code
            // m_generator->emit(op, regs);
            throw ngraph::ngraph_error(std::string("not implemented operation ") + n->get_type_info().name);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Sigmoid>(n)) {
            m_generator->emit(op, regs);
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

    return false;
}

#if 0
// std::map<ngraph::NodeTypeInfo, std::function<void(std::vector<size_t>, std::vector<size_t>, std::vector<size_t>)>> jitters;

bool ngraph::pass::GenerateCodePass::run_on_function(std::shared_ptr<Function> func) {
    for (auto n : func->get_ordered_ops()) {
        auto regs = getRegisters(n);
        this->jitters[n->get_type_info()](registers.in, registers.out, registers.pool);
    }

    return false;
}
#endif