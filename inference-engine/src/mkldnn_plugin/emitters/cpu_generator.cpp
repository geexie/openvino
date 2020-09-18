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

CPUGenerator::CPUGenerator() : h(new jit_snippet()) {
    reg64_tmp_start = h->r8.getIdx();

    // FIXME: it's going to be created with `NGRAPH_OP` macro or even pass manager like add mather, but cannot deside for now
    jitters[ngraph::opset1::Add().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new AddEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::opset1::Subtract().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new SubtractEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::opset1::Erf().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ErfEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::opset1::Parameter().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new NopEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::opset1::Result().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new NopEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Load().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new LoadEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::ScalarLoad().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ScalarLoadEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Store().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new StoreEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::ScalarStore().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ScalarStoreEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::BroadcastLoad().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new BroadcastLoadEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::v1::Multiply().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new MultiplyEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Negative().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new NegativeEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::v1::Divide().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new DivideEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Clamp().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ClampEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Relu().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ReluEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Sigmoid().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new SigmoidEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::SquaredDifference().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new SquaredDifferenceEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::Scalar().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ScalarEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };

    jitters[ngraph::op::PRelu().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new PReluEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };
    jitters[ngraph::op::FakeBroadcast().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new FakeBroadcastEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };
    jitters[ngraph::op::v1::Power().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new PowerEmitter(h.get(), mkldnn::impl::cpu::avx2, n));
    };
    std::cout << jitters.size() << " @@@@@@@@ !!!!!!!!!" << std::endl;
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

code CPUGenerator::generate(std::shared_ptr<ngraph::Function>& f) const {
    if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::avx2)) {
        remark(10) << "generating for AVX2 ISA" << std::endl;
    } else {
        throw ngraph::ngraph_error("unsupported architecture for code genration");
    }

    // part 2: generation flow
    // ngraph::pass::AssignRegistersPass().run_on_function(f_scalar);
    // sets up offsets to constant and temporals
    ngraph::pass::SetupStackTemporalsOffsetPass().run_on_function(f);

    auto f_scalar = ngraph::clone_function(*f.get());

    generate_propotype(f);

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
            ngraph::pass::GenerateCodePass(this).run_on_function(bodies[loopId]);

            // loop_advance()
            h->sub(amount, nloads[loopId]);
            h->cmp(amount, nloads[loopId]);
            h->jge(for_body[loopId], jit_snippet::T_NEAR);
        }
    }

    h->L(for_body[nloads.size()]);
#endif

    generate_return(f);

    return h->getCode();
}

void CPUGenerator::generate_tile(std::shared_ptr<ngraph::Function>& f) const {
}

void CPUGenerator::generate_return(std::shared_ptr<ngraph::Function>& f) const {
    h->postamble();
#if 1
    h->align(64);
    h->L(l_table);

    ngraph::pass::GenerateConstntTables(this).run_on_function(f);
#endif
}

void CPUGenerator::emit_table(const std::shared_ptr<op::Scalar>& op) const {
    auto out_shape = op->output(0).get_tensor().get_shape();
    if (out_shape == Shape() || ngraph::shape_size(out_shape) == 1) {
        remark(11) << "pugging constant " << op->cast_vector<float>()[0] << " to the stack" << std::endl;
        h->dd(mkldnn::impl::cpu::float2int(op->cast_vector<float>()[0]));
    }
}

void CPUGenerator::emit_table(const std::shared_ptr<opset1::Erf>& op) const {
    h->dd(mkldnn::impl::cpu::float2int(2.86f));                   // 0
    h->dd(mkldnn::impl::cpu::float2int(1.00f));// 0x3f800000      // 1

    h->dd(mkldnn::impl::cpu::float2int(90.0260162353515625f));    // 2
    h->dd(mkldnn::impl::cpu::float2int(2232.00537109375f));       // 3
    h->dd(mkldnn::impl::cpu::float2int(7003.3251953125f));        // 4
    h->dd(mkldnn::impl::cpu::float2int(55592.30078125f));         // 5

    h->dd(mkldnn::impl::cpu::float2int(33.56171417236328125f));   // 6
    h->dd(mkldnn::impl::cpu::float2int(521.35797119140625f));     // 7
    h->dd(mkldnn::impl::cpu::float2int(4594.32373046875f));       // 8
    h->dd(mkldnn::impl::cpu::float2int(22629.0f));                // 9
    h->dd(mkldnn::impl::cpu::float2int(49267.39453125f));         // 10

    h->dd(mkldnn::impl::cpu::float2int(9.60497379302978515625f)); // 11
    h->dd(0x80000000);                                            // 12
    h->dd(0x7fffffff);                                            // 13
}

void CPUGenerator::emit_table(const std::shared_ptr<opset1::Clamp>& op) const {
    remark(11) << "pugging Clamp min " << op->get_min() << " to the stack" << std::endl;
    h->dd(mkldnn::impl::cpu::float2int(static_cast<float>(op->get_min())));
    remark(11) << "pugging Clamp max " << op->get_max() << " to the stack" << std::endl;
    h->dd(mkldnn::impl::cpu::float2int(static_cast<float>(op->get_max())));
}

void CPUGenerator::emit_table(const std::shared_ptr<opset1::Sigmoid>& op) const {
    size_t vlen = 8*sizeof(float);
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b17218, //[10] logf(FLT_MAX)
            0xc2aeac50, //[11] logf(FLT_MIN)
            // tanh(x) constants,
            0x80000000, //[12] mask to extract sign
            0x39ddb3d7, //[13] arg below which tanh(x) = x
            0x3f0c9f54, //[14] arg below which pol approx is valid
            0x41102cb4, //[15] arg after which tanh(x) = 1
            0xc0000000, //[16] -2.0f
            0x7fffffff, //[17] mask to make positive
            // tanh pol approx
            0x3f7fffff, //[18] p0
            0xbeaaa9cf, //[19] p1
            0x3e085f1f, //[20] p2
            0xbd572bda, //[21] p3
            0x3c84fd08, //[22] p4
            // gelu approx constants
            0x3d372713, //[23] 0.044715
            0x3f4c4229, //[24] sqrt(2/pi)
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(cvals[i]);
    }
}