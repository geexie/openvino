// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitter.hpp"
#include "jit_generator.hpp"
#include "mkldnn_node.h"
#include "jit_uni_eltwise.hpp"

// wrapped
// mkldnn_eltwise_relu = 0x1f, +
// mkldnn_eltwise_tanh = 0x2f, +
// mkldnn_eltwise_logistic = 0xaf, +
// mkldnn_eltwise_elu = 0x3f, +
// mkldnn_eltwise_exp = 0xbf, +
// mkldnn_eltwise_abs = 0x5f,+
// mkldnn_eltwise_sqrt = 0x6f,+

// to do
// mkldnn_eltwise_bounded_relu = 0x8f,
// mkldnn_eltwise_soft_relu = 0x9f,

// not in opset 1
// mkldnn_eltwise_linear = 0x7f, ??
// mkldnn_eltwise_gelu = 0xcf,
// mkldnn_eltwise_swish = 0xff,
// mkldnn_eltwise_mish = 0x1f0,
// mkldnn_eltwise_square = 0x4f,

// implemented manually
// mkldnn_eltwise_clamp = 0xdf,
// mkldnn_eltwise_not = 0xef,

namespace MKLDNNPlugin {

class jit_mkldnn_emitter : public jit_emitter {
public:
    size_t get_inputs_num() override;

    void emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
              const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;

    void emit_table() override;

protected:
    jit_mkldnn_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_mkldnn_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode& node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    void set_injector();

    mkldnn_alg_kind_t kind {mkldnn_alg_kind_undef};
    float alpha {0.f};
    float beta {0.f};

private:
    std::shared_ptr<mkldnn::impl::cpu::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::sse42>> eltwise_injector_sse42;
    std::shared_ptr<mkldnn::impl::cpu::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::avx2>> eltwise_injector_avx2;
    std::shared_ptr<mkldnn::impl::cpu::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::avx512_common>> eltwise_injector_avx512_common;
};

class jit_mkldnn_aux_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_aux_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const MKLDNNNode& node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

private:
};

class jit_mkldnn_relu_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_relu_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_relu;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_sigmoid_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_sigmoid_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_logistic;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_tanh_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_tanh_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_tanh;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_elu_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_elu_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_elu;
            alpha = ngraph::as_type_ptr<ngraph::opset1::Elu>(n)->get_alpha();
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_exp_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_exp_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_exp;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_abs_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_abs_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_abs;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_mkldnn_sqrt_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_sqrt_emitter(mkldnn::impl::cpu::jit_generator *host, mkldnn::impl::cpu::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_sqrt;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};


} // namespace MKLDNNPlugin