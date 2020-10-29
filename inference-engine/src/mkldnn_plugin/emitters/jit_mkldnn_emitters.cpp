// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mkldnn_emitters.hpp"
#include "nodes/mkldnn_eltwise_node.h"

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu;
using namespace Xbyak;

namespace MKLDNNPlugin {

jit_mkldnn_emitter::jit_mkldnn_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& node, InferenceEngine::Precision exec_prc)
    : jit_emitter(host, host_isa, node, exec_prc) {

    kind = mkldnn_eltwise_tanh;//static_cast<mkldnn_alg_kind_t>(eltwiseNode.getAlgorithm());
    alpha = 0.f;
    beta = 0.f;

    set_injector();
}

jit_mkldnn_emitter::jit_mkldnn_emitter(jit_generator *host, cpu_isa_t host_isa, const MKLDNNNode& node, InferenceEngine::Precision exec_prc)
    : jit_emitter(host, host_isa, node, exec_prc) {
    auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(node);
    kind = static_cast<mkldnn_alg_kind_t>(eltwiseNode.getAlgorithm());
    alpha = 0.f;
    beta = 0.f;

    set_injector();
}

void jit_mkldnn_emitter::set_injector() {
    if (host_isa_ == cpu::sse42) {
        eltwise_injector_sse42 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::sse42>>(h, kind, alpha, beta);
    } else if (host_isa_ == cpu::avx2) {
        eltwise_injector_avx2 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx2>>(h, kind, alpha, beta);
    } else if (host_isa_ == cpu::avx512_common) {
        eltwise_injector_avx512_common = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx512_common>>(h, kind, alpha, beta);
    } else {
        assert(!"unsupported isa");
    }
}

size_t jit_mkldnn_emitter::get_inputs_num() { return 1; }

void jit_mkldnn_emitter::emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                              const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    assert(kind != mkldnn_alg_kind_undef);
    if (host_isa_ == cpu::sse42) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Xmm(out_vec_idxs[0]), Xmm(in_vec_idxs[0]));
        eltwise_injector_sse42->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::avx2) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Ymm(out_vec_idxs[0]), Ymm(in_vec_idxs[0]));
        eltwise_injector_avx2->compute_vector(out_vec_idxs[0]);
    } else if (host_isa_ == cpu::avx512_common) {
        if (out_vec_idxs[0] != in_vec_idxs[0])
            h->uni_vmovups(Zmm(out_vec_idxs[0]), Zmm(in_vec_idxs[0]));
        eltwise_injector_avx512_common->compute_vector(out_vec_idxs[0]);
    } else {
        assert(!"unsupported isa");
    }
}

void jit_mkldnn_emitter::emit_table() {
    assert(kind != mkldnn_alg_kind_undef);
    if (host_isa_ == cpu::sse42) {
        eltwise_injector_sse42->prepare_table();
    } else if (host_isa_ == cpu::avx2) {
        eltwise_injector_avx2->prepare_table();
    } else if (host_isa_ == cpu::avx512_common) {
        eltwise_injector_avx512_common->prepare_table();
    } else {
        assert(!"unsupported isa");
    }
}

jit_mkldnn_aux_emitter::jit_mkldnn_aux_emitter(jit_generator *host, cpu_isa_t host_isa, const MKLDNNNode& node, InferenceEngine::Precision exec_prc)
    : jit_mkldnn_emitter(host, host_isa, node, exec_prc) {
    // auto& eltwiseNode = dynamic_cast<const MKLDNNEltwiseNode&>(n);

    // auto alg = static_cast<mkldnn_alg_kind_t>(eltwiseNode.getAlgorithm());

    // if (host_isa_ == cpu::sse42) {
    //     eltwise_injector_sse42 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::sse42>>(
    //             host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    // } else if (host_isa_ == cpu::avx2) {
    //     eltwise_injector_avx2 = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx2>>(
    //             host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    // } else if (host_isa_ == cpu::avx512_common) {
    //     eltwise_injector_avx512_common = std::make_shared<jit_uni_eltwise_injector_f32<cpu::avx512_common>>(
    //             host, alg, eltwiseNode.getAlpha(), eltwiseNode.getBeta());
    // } else {
    //     assert(!"unsupported isa");
    // }
}

} // namespace MKLDNNPlugin
