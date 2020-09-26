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
#include "jit_eltwise_emitters.hpp"
#include "jit_mkldnn_emitters.hpp"

using namespace std;
using namespace ngraph;

#define CREATE_EMITTER(e_type) [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {return std::make_shared<e_type>(h.get(), isa, n);};

CPUGenerator::CPUGenerator() : h(new jit_snippet()), isa(mkldnn::impl::cpu::avx2) {
    reg64_tmp_start = h->r8.getIdx();

    // data movement
    jitters[ngraph::opset1::Parameter().get_type_info()] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::op::BlockedParameter().get_type_info()] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::opset1::Result().get_type_info()] = CREATE_EMITTER(NopEmitter);
    // jitters[ngraph::opset1::Constant().get_type_info()] = CREATE_EMITTER(); // Not supported

    jitters[ngraph::op::Load().get_type_info()] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::op::VectorLoad().get_type_info()] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::op::ScalarLoad().get_type_info()] = CREATE_EMITTER(ScalarLoadEmitter);
    jitters[ngraph::op::BroadcastLoad().get_type_info()] = CREATE_EMITTER(BroadcastLoadEmitter);
    // jitters[ngraph::op::BlockedLoad().get_type_info()] = CREATE_EMITTER(); // Not supported

    jitters[ngraph::op::Store().get_type_info()] = CREATE_EMITTER(StoreEmitter);
    jitters[ngraph::op::VectorStore().get_type_info()] = CREATE_EMITTER(StoreEmitter);
    jitters[ngraph::op::ScalarStore().get_type_info()] = CREATE_EMITTER(ScalarStoreEmitter);

    jitters[ngraph::op::Scalar().get_type_info()] = CREATE_EMITTER(ScalarEmitter);
    jitters[ngraph::op::FakeBroadcast().get_type_info()] = CREATE_EMITTER(FakeBroadcastEmitter);
    // jitters[ngraph::op::Nop().get_type_info()] = CREATE_EMITTER(NopEmitter); // Not supported
    // jitters[ngraph::opset1::Broadcast().get_type_info()] = CREATE_EMITTER(); // Not supported

    // binary
    jitters[ngraph::opset1::Add().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_add_emitter);
    jitters[ngraph::opset1::Divide().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_divide_emitter);
    jitters[ngraph::opset1::Equal().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_equal_emitter);
    jitters[ngraph::opset1::FloorMod().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_floor_mod_emitter);
    jitters[ngraph::opset1::Greater().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_greater_emitter);
    jitters[ngraph::opset1::GreaterEqual().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_greater_equal_emitter);
    jitters[ngraph::opset1::Less().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_less_emitter);
    jitters[ngraph::opset1::LessEqual().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_less_equal_emitter);
    jitters[ngraph::opset1::LogicalAnd().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_and_emitter);
    jitters[ngraph::opset1::LogicalOr().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_or_emitter);
    jitters[ngraph::opset1::LogicalXor().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_xor_emitter);
    jitters[ngraph::opset1::Maximum().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_maximum_emitter);
    jitters[ngraph::opset1::Minimum().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_minimum_emitter);
    jitters[ngraph::opset1::Mod().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mod_emitter);
    jitters[ngraph::opset1::Multiply().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_multiply_emitter);
    jitters[ngraph::opset1::NotEqual().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_not_equal_emitter);
    jitters[ngraph::opset1::Power().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_power_static_emitter);
    jitters[ngraph::opset1::PRelu().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_prelu_emitter);
    jitters[ngraph::opset1::SquaredDifference().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_squared_difference_emitter);
    jitters[ngraph::opset1::Subtract().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_subtract_emitter);
    // jitters[ngraph::opset1::Xor().get_type_info()] = CREATE_EMITTER(); // not supported

    // unary
    jitters[ngraph::opset1::Abs().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_abs_emitter);
    // jitters[ngraph::opset1::Acos().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Asin().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Atan().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Ceiling().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Clamp().get_type_info()] = CREATE_EMITTER(ClampEmitter);
    // jitters[ngraph::opset1::Cos().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Cosh().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Elu().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_elu_emitter);
    // jitters[ngraph::opset1::Erf().get_type_info()] = CREATE_EMITTER(ErfEmitter);
    jitters[ngraph::opset1::Exp().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_exp_emitter);
    // jitters[ngraph::opset1::Floor().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::HardSigmoid().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Log().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::LogicalNot().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_not_emitter);
    jitters[ngraph::opset1::Negative().get_type_info()] = CREATE_EMITTER(NegativeEmitter);
    jitters[ngraph::opset1::Relu().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_relu_emitter);
    // jitters[ngraph::opset1::Selu().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Sign().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sigmoid().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_sigmoid_emitter);
    // jitters[ngraph::opset1::Sin().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Sinh().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sqrt().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_sqrt_emitter);
    // jitters[ngraph::opset1::Tan().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Tanh().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_mkldnn_tanh_emitter);
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

    // h->mov(p_table, l_table);
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

    std::vector<std::shared_ptr<Emitter>> lowered;
    std::vector<std::shared_ptr<Emitter>> scalar_lowered;
    std::vector<RegInfo> reginfo;
    std::vector<RegInfo> scalar_reginfo;

    for (auto n : f->get_ordered_ops()) {
        reginfo.push_back(ngraph::snippet::getRegisters(n));
        if (jitters.find(n->get_type_info()) != jitters.end()) {
            lowered.push_back(jitters[n->get_type_info()](n));
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

    for (auto n : f_scalar->get_ordered_ops()) {
        scalar_reginfo.push_back(ngraph::snippet::getRegisters(n));
        if (jitters.find(n->get_type_info()) != jitters.end()) {
            scalar_lowered.push_back(jitters[n->get_type_info()](n));
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

#if 1
    // configure tile variants
    size_t vlen = mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen;
    std::array<size_t, 2> nloads   = {vlen / sizeof(float), 1};
    std::array<std::shared_ptr<ngraph::Function>, 2> bodies   = {f, f_scalar};
    std::array<Xbyak::Label, nloads.size() + 1> for_body;

    std::array<std::vector<std::shared_ptr<Emitter>>, 2> lowered_ops = {lowered, scalar_lowered};
    std::array<std::vector<RegInfo>, 2> reginfos   = {reginfo, scalar_reginfo};

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
            for (size_t i = 0; i < lowered_ops[loopId].size(); i++) {
                auto regs = reginfos[loopId][i];
                lowered_ops[loopId][i]->emit(regs.first, regs.second);
            }

            // loop_advance()
            h->sub(amount, nloads[loopId]);
            h->cmp(amount, nloads[loopId]);
            h->jge(for_body[loopId], jit_snippet::T_NEAR);
        }
    }

    h->L(for_body[nloads.size()]);
#endif

    h->postamble();
    for (int loopId = 0; loopId < nloads.size(); loopId++) {
        for (size_t i = 0; i < lowered_ops[loopId].size(); i++) {
            lowered_ops[loopId][i]->emit_table();
        }
    }
}

void CPUGenerator::generate_return(std::shared_ptr<ngraph::Function>& f) const {
//     h->postamble();
// #if 1
//     // h->align(64);
//     // h->L(l_table);

//     for (auto n : f->get_ordered_ops()) {
//         if (jitters.find(n->get_type_info()) != jitters.end()) {
//             jitters[n->get_type_info()](n)->emit_table();
//         } else {
//             throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
//         }
//     }
// #endif
}
