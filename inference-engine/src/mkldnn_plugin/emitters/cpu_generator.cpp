// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/setup_stack_pass.hpp"
#include "transformations/snippets/generate_constant_tables.hpp"
#include "transformations/snippets/generate_pass.hpp"
#include "transformations/rt_info/register_info.hpp"

#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>

#include <string>
#include <iostream>
#include <array>

#include "transformations/snippets/remarks.hpp"

#include "jitters.hpp"

using namespace std;
using namespace ngraph;

#define CREATE_EMITTER(e_type) [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {return std::make_shared<e_type>(h.get(), isa, n);};

CPUGenerator::CPUGenerator() : h(new jit_snippet()), isa(mkldnn::impl::cpu::avx2) {
    reg64_tmp_start = h->r8.getIdx();

    jitters[ngraph::opset1::Parameter().get_type_info()] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::opset1::Result().get_type_info()] = CREATE_EMITTER(NopEmitter);

    jitters[ngraph::op::Load().get_type_info()] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::op::ScalarLoad().get_type_info()] = CREATE_EMITTER(ScalarLoadEmitter);
    jitters[ngraph::op::Store().get_type_info()] = CREATE_EMITTER(StoreEmitter);
    jitters[ngraph::op::ScalarStore().get_type_info()] = CREATE_EMITTER(ScalarStoreEmitter);
    jitters[ngraph::op::BroadcastLoad().get_type_info()] = CREATE_EMITTER(BroadcastLoadEmitter);
    jitters[ngraph::op::Scalar().get_type_info()] = CREATE_EMITTER(ScalarEmitter);
    jitters[ngraph::op::FakeBroadcast().get_type_info()] = CREATE_EMITTER(FakeBroadcastEmitter);

    jitters[ngraph::opset1::Add().get_type_info()] = CREATE_EMITTER(AddEmitter);
    jitters[ngraph::opset1::Subtract().get_type_info()] = CREATE_EMITTER(SubtractEmitter);
    jitters[ngraph::opset1::Erf().get_type_info()] = CREATE_EMITTER(ErfEmitter);
    jitters[ngraph::opset1::Multiply().get_type_info()] = CREATE_EMITTER(MultiplyEmitter);
    jitters[ngraph::opset1::Negative().get_type_info()] = CREATE_EMITTER(NegativeEmitter);
    jitters[ngraph::opset1::Divide().get_type_info()] = CREATE_EMITTER(DivideEmitter);
    jitters[ngraph::opset1::Clamp().get_type_info()] = CREATE_EMITTER(ClampEmitter);
    jitters[ngraph::opset1::Relu().get_type_info()] = CREATE_EMITTER(ReluEmitter);
    jitters[ngraph::op::Sigmoid().get_type_info()] = CREATE_EMITTER(SigmoidEmitter);
    jitters[ngraph::opset1::SquaredDifference().get_type_info()] = CREATE_EMITTER(SquaredDifferenceEmitter);
    jitters[ngraph::op::PRelu().get_type_info()] = CREATE_EMITTER(PReluEmitter);
    jitters[ngraph::opset1::Power().get_type_info()] = CREATE_EMITTER(PowerEmitter);
}

code CPUGenerator::generate(std::shared_ptr<ngraph::Function>& f) const {
    if (mkldnn::impl::cpu::mayiuse(isa)) {
        remark(10) << "generating for AVX2 ISA" << std::endl;
    } else {
        throw ngraph::ngraph_error("unsupported architecture for code genration");
    }

    // sets up offsets to constant and temporals
    // FIXME: is not needed for string based constants
    ngraph::pass::SetupStackTemporalsOffsetPass().run_on_function(f);

    generate_propotype(f);
    generate_tile(f);
    generate_return(f);
    return h->getCode();
}

void CPUGenerator::generate_propotype(std::shared_ptr<ngraph::Function>& f) const {
    // Note: should it be also a pass?
    h->preamble();

    auto params = f->get_parameters();
    auto results = f->get_results();

    std::vector<Xbyak::Reg64> regs(params.size()+results.size());
    for (auto i = 0; i < regs.size(); i++) {
        regs[i] = Xbyak::Reg64(reg64_tmp_start+i);
    }

    for (auto i = 0; i < params.size(); i++) {
        h->mov(regs[i], h->ptr[param + i*sizeof(size_t)]);
    }

    for (auto i = 0; i < results.size(); i++) {
        h->mov(regs[params.size()+i], h->ptr[param + (params.size()+i)*sizeof(size_t)]);
    }

    size_t nConstants = 0;
    for (auto op : f->get_ordered_ops()) {
        if (auto constant = as_type_ptr<ngraph::opset1::Constant>(op)) {
            // Setup non-scalar constant for load
            if (constant->output(0).get_tensor().get_shape() != Shape() && ngraph::shape_size(constant->output(0).get_tensor().get_shape()) > 1) {
                remark(12) << "setting constant to " << regs.size()+1+nConstants << " "
                          << std::hex << reinterpret_cast<size_t>(&constant->get_data_ptr<float>()[0]) << " "
                          << std::dec << constant->get_data_ptr<float>()[0]<< std::endl;
                h->mov(Xbyak::Reg64(reg64_tmp_start+regs.size()+1+nConstants),
                    /*reinterpret_cast<size_t>(&constant->get_data_ptr<float>()[0])*/
                    h->ptr[param + (params.size()+results.size()+1+nConstants)*sizeof(size_t)]);
                nConstants++;
            }
        }
    }

    if (params.size()+results.size()+nConstants+1 > 8) {
        throw ngraph_error(std::string("snippet signature should not exceed 7 arguments. got") + std::to_string(params.size()+results.size()+nConstants));
    }

    h->mov(p_table, l_table);
}

void CPUGenerator::generate_tile(std::shared_ptr<ngraph::Function>& f) const {
    auto f_scalar = ngraph::clone_function(*f.get());
    ngraph::pass::ReplaceLoadsWithScalarLoads().run_on_function(f_scalar);
    int qqq = 0;
    for (auto op : f_scalar->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }
    qqq = 0;
    for (auto op : f->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

#if 1
    // configure tile variants
    size_t vlen = mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen;
    std::array<size_t, 2> nloads   = {vlen / sizeof(float), 1};
    std::array<std::shared_ptr<ngraph::Function>, 2> bodies   = {f, f_scalar};
    std::array<Xbyak::Label, nloads.size() + 1> for_body;

    // obtain work amount
    Xbyak::Reg64 param = mkldnn::impl::cpu::abi_param1; // RCX
    int reg_start = h->r8.getIdx();
    auto nparams = f->get_results().size() + f->get_parameters().size();
    Xbyak::Reg64 amount = Xbyak::Reg64(reg_start+nparams);
    h->mov(amount, h->ptr[param + sizeof(size_t)*nparams]);

    // generate both vector and scalar loops
    for (int loopId = 0; loopId < nloads.size(); loopId++) {
        // loop_entry()
        h->cmp(amount, nloads[loopId]);
        h->jl(for_body[loopId+1], jit_snippet::T_NEAR);

        // loop_body()
        h->L(for_body[loopId]); {
            for (auto n : bodies[loopId]->get_ordered_ops()) {
                auto regs = ngraph::snippet::getRegisters(n);
                if (jitters.find(n->get_type_info()) != jitters.end()) {
                    jitters[n->get_type_info()](n)->emit(regs.first, regs.second);
                } else {
                    throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
                }
            }

            // loop_advance()
            h->sub(amount, nloads[loopId]);
            h->cmp(amount, nloads[loopId]);
            h->jge(for_body[loopId], jit_snippet::T_NEAR);
        }
    }

    h->L(for_body[nloads.size()]);
#endif
}

void CPUGenerator::generate_return(std::shared_ptr<ngraph::Function>& f) const {
    h->postamble();
#if 1
    h->align(64);
    h->L(l_table);

    for (auto n : f->get_ordered_ops()) {
        if (jitters.find(n->get_type_info()) != jitters.end()) {
            jitters[n->get_type_info()](n)->emit_table();
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }
#endif
}
