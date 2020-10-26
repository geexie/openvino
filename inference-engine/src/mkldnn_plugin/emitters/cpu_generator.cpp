// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/vector_to_scalar_pass.hpp"
#include "transformations/rt_info/register_info.hpp"
#include "transformations/snippets/remarks.hpp"

#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>

#include <string>
#include <iostream>
#include <array>

#include "cpu_generator.hpp"
#include "jitters.hpp"
#include "jit_eltwise_emitters.hpp"
#include "jit_mkldnn_emitters.hpp"

using namespace std;
using namespace ngraph;

#define CREATE_EMITTER(e_type) [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {return std::make_shared<e_type>(h.get(), isa, n);};

CPUGenerator::CPUGenerator() : h(new jit_snippet()), isa(mkldnn::impl::cpu::avx2) {
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

    // jitters[ngraph::opset1::Convert().get_type_info()] = CREATE_EMITTER(); // Not supported
    // it might be better to decompose it, but standart do_decompose algorithm use Quantize which is deprecated
    // jitters[ngraph::opset1::FakeQuantize().get_type_info()] = CREATE_EMITTER(); // not supported

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
    jitters[ngraph::opset1::Xor().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_xor_emitter);

    // unary
    jitters[ngraph::opset1::Abs().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_abs_emitter);
    // jitters[ngraph::opset1::Acos().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Asin().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Atan().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Ceiling().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Clamp().get_type_info()] = CREATE_EMITTER(ClampEmitter);
    // jitters[ngraph::opset1::Cos().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Cosh().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Elu().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_elu_emitter);
    jitters[ngraph::opset1::Erf().get_type_info()] = CREATE_EMITTER(ErfEmitter);
    jitters[ngraph::opset1::Exp().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_exp_emitter);
    // jitters[ngraph::opset1::Floor().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Log().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::LogicalNot().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_not_emitter);
    jitters[ngraph::opset1::Negative().get_type_info()] = CREATE_EMITTER(NegativeEmitter);
    jitters[ngraph::opset1::Relu().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_relu_emitter);
    // jitters[ngraph::opset1::Sign().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sigmoid().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_sigmoid_emitter);
    // jitters[ngraph::opset1::Sin().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Sinh().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sqrt().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_sqrt_emitter);
    // jitters[ngraph::opset1::Tan().get_type_info()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Tanh().get_type_info()] = CREATE_EMITTER(MKLDNNPlugin::jit_tanh_emitter);

    // jitters[ngraph::opset1::HardSigmoid().get_type_info()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Selu().get_type_info()] = CREATE_EMITTER(); // not supported

    // Also set tile jitter here.
    // jitters[ngraph::op::Subgraph().get_type_info()] = CREATE_EMITTER(KernelEmitter);
    // jitters[ngraph::op::Tile().get_type_info()] = CREATE_EMITTER(TileEmitter);
}

// template <> struct cpu_isa_traits<avx> {
//     typedef Xbyak::Ymm Vmm;
//     static constexpr int vlen_shift = 5;
//     static constexpr int vlen = 32;
//     static constexpr int n_vregs = 16;
// };

auto get_lanes(mkldnn::impl::cpu::cpu_isa_t isa) -> size_t {
    switch (isa) {
        case mkldnn::impl::cpu::avx2 : return mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen / sizeof(float);
        case mkldnn::impl::cpu::sse42 : return mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::sse42>::vlen / sizeof(float);
        case mkldnn::impl::cpu::avx512_common : return mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx512_common>::vlen / sizeof(float);
        default : throw ngraph::ngraph_error(std::string("unknown isa"));
    }
}

code CPUGenerator::generate(std::shared_ptr<ngraph::Function>& f) const {
    if (mkldnn::impl::cpu::mayiuse(isa)) {
        remark(10) << "generating for AVX2 ISA" << std::endl;
    } else {
        throw ngraph::ngraph_error("unsupported architecture for code genration");
    }

    int qqq = 0;
    for (auto op : f->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    auto params = f->get_parameters();
    auto results = f->get_results();

    auto nparams = f->get_results().size() + f->get_parameters().size();

    if (nparams > 7) {
        throw ngraph_error(std::string("snippet signature should not exceed 7 arguments. got") + std::to_string(nparams));
    }

    // vector tile
    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> lowered;

    for (auto n : f->get_ordered_ops()) {
        if (jitters.find(n->get_type_info()) != jitters.end()) {
            lowered.push_back(std::make_pair(jitters[n->get_type_info()](n), ngraph::snippet::getRegisters(n)));
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

    // scalar tile
    auto f_scalar = ngraph::clone_function(*f.get());
    ngraph::pass::ReplaceLoadsWithScalarLoads().run_on_function(f_scalar);

    qqq = 0;
    for (auto op : f_scalar->get_ordered_ops()) {
        remark(13) << "op " << qqq++ << " " << op->get_friendly_name() << " (" << op->get_type_name() << ") " << op << std::endl;
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> scalar_lowered;

    for (auto n : f_scalar->get_ordered_ops()) {
        if (jitters.find(n->get_type_info()) != jitters.end()) {
            scalar_lowered.push_back(make_pair(jitters[n->get_type_info()](n), ngraph::snippet::getRegisters(n)));
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, RegInfo>> tiles;
    tiles.push_back(std::make_pair(std::make_shared<TileEmitter>(h.get(), isa, lowered),
                    std::make_pair(std::vector<size_t>({get_lanes(isa), nparams}), std::vector<size_t>{})));
    tiles.push_back(std::make_pair(std::make_shared<TileEmitter>(h.get(), isa, scalar_lowered),
                    std::make_pair(std::vector<size_t>{1, nparams}, std::vector<size_t>{})));

    //// emission
    std::shared_ptr<Emitter> kernel = std::make_shared<KernelEmitter>(h.get(), isa, tiles);
    kernel->emit({params.size(), results.size()}, {});

    lowered.insert(lowered.end(), scalar_lowered.begin(), scalar_lowered.end());
    for (auto& op : lowered) {
        op.first->emit_table();
    }

    return h->getCode();
}
