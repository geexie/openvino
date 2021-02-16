// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include "jit_emitter.hpp"
#include "mkldnn_node.h"

namespace MKLDNNPlugin {

class jit_add_emitter : public jit_emitter {
public:
    jit_add_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_add_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_mul_add_emitter : public jit_emitter {
public:
    jit_mul_add_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_mul_add_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};


class jit_subtract_emitter : public jit_emitter {
public:
    jit_subtract_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_subtract_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_multiply_emitter : public jit_emitter {
public:
    jit_multiply_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_multiply_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                         InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_divide_emitter : public jit_emitter {
public:
    jit_divide_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_divide_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_floor_mod_emitter : public jit_emitter {
public:
    jit_floor_mod_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_floor_mod_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_mod_emitter : public jit_emitter {
public:
    jit_mod_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_mod_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    size_t aux_vecs_count() const override;
};


class jit_maximum_emitter : public jit_emitter {
public:
    jit_maximum_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_maximum_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_minimum_emitter : public jit_emitter {
public:
    jit_minimum_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_minimum_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;
    static std::set<InferenceEngine::Precision> get_supported_precisions();

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_squared_difference_emitter : public jit_emitter {
public:
    jit_squared_difference_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                                   InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_squared_difference_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                                   InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_power_dynamic_emitter : public jit_emitter {
public:
    jit_power_dynamic_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_power_dynamic_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};


class jit_equal_emitter : public jit_emitter {
public:
    jit_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_not_equal_emitter : public jit_emitter {
public:
    jit_not_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_not_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                          InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_emitter : public jit_emitter {
public:
    jit_greater_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_greater_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_greater_equal_emitter : public jit_emitter {
public:
    jit_greater_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_greater_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                              InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_emitter : public jit_emitter {
public:
    jit_less_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                     InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_less_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                     InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_less_equal_emitter : public jit_emitter {
public:
    jit_less_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    jit_less_equal_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_and_emitter : public jit_emitter {
public:
    jit_logical_and_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_logical_and_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_or_emitter : public jit_emitter {
public:
    jit_logical_or_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_logical_or_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};


class jit_logical_xor_emitter : public jit_emitter {
public:
    jit_logical_xor_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_logical_xor_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_logical_not_emitter : public jit_emitter {
public:
    jit_logical_not_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_logical_not_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;
};

class jit_power_static_emitter : public jit_emitter {
public:
    jit_power_static_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                            InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_power_static_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                             InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void register_table_entries() override;
    size_t aux_vecs_count() const override;

    float power;
    float scale;
    float shift;
};

class jit_prelu_emitter : public jit_emitter {
public:
    jit_prelu_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_prelu_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                      InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    size_t aux_vecs_count() const override;
};

class jit_sqrt_emitter : public jit_emitter {
public:
    jit_sqrt_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_sqrt_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                  const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                  const emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class jit_negative_emitter : public jit_emitter {
public:
    jit_negative_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                    InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out,
                  const std::vector<size_t>& pool, const std::vector<size_t>& gpr,
                  const MKLDNNPlugin::emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
};

class ErfEmitter : public jit_emitter {
public:
    ErfEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        push_arg_entry_of("max_val", mkldnn::impl::cpu::x64::float2int(2.86f), false);
        push_arg_entry_of("one", 0x3f800000, false);

        push_arg_entry_of("p1", mkldnn::impl::cpu::x64::float2int(90.0260162353515625f), false);
        push_arg_entry_of("p2", mkldnn::impl::cpu::x64::float2int(2232.00537109375000f), false);
        push_arg_entry_of("p3", mkldnn::impl::cpu::x64::float2int(7003.32519531250000f), false);
        push_arg_entry_of("p4", mkldnn::impl::cpu::x64::float2int(55592.3007812500000f), false);

        push_arg_entry_of("rp0", mkldnn::impl::cpu::x64::float2int(33.56171417236328125f), false);
        push_arg_entry_of("rp1", mkldnn::impl::cpu::x64::float2int(521.35797119140625f), false);
        push_arg_entry_of("rp2", mkldnn::impl::cpu::x64::float2int(4594.32373046875f), false);
        push_arg_entry_of("rp3", mkldnn::impl::cpu::x64::float2int(22629.0f), false);
        push_arg_entry_of("rp4", mkldnn::impl::cpu::x64::float2int(49267.39453125f), false);

        push_arg_entry_of("p0", mkldnn::impl::cpu::x64::float2int(9.60497379302978515625f), false);

        push_arg_entry_of("exp", 0x80000000, false);
        push_arg_entry_of("sign", 0x7fffffff, false);

        prepare_table();
    }

    size_t get_inputs_num() const override {return 1;}

protected:
    size_t aux_gprs_count() const override {return 1;}
    size_t aux_vecs_count() const override {return 4;}

    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Ymm x = Xbyak::Ymm(in[0]);
        Xbyak::Ymm polynom = Xbyak::Ymm(out[0]);

        h->uni_vbroadcastss(polynom, table_val("p0"));

        Xbyak::Ymm x2 = Xbyak::Ymm(aux_vec_idxs[0]);
        Xbyak::Ymm c0 = Xbyak::Ymm(aux_vec_idxs[1]);
        Xbyak::Ymm val = Xbyak::Ymm(aux_vec_idxs[2]);
        Xbyak::Ymm sign = Xbyak::Ymm(aux_vec_idxs[3]);

        h->uni_vbroadcastss(c0, table_val("exp"));
        h->uni_vandps(sign, x, c0);

        h->uni_vbroadcastss(c0, table_val("sign"));
        h->uni_vandps(val, x, c0);

        h->uni_vmulps(x2, x, x);

        h->uni_vbroadcastss(c0, table_val("p1"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("p2"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("p3"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("p4"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        // x *= polynom;
        h->uni_vmulps(x, x, polynom);

        h->uni_vbroadcastss(polynom, table_val("one"));

        h->uni_vbroadcastss(c0, table_val("rp0"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("rp1"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("rp2"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("rp3"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, table_val("rp4"));
        h->uni_vfmadd213ps(polynom, x2, c0);

        // return x / polynom;
        h->uni_vdivps(polynom, x, polynom);

        h->uni_vbroadcastss(c0, table_val("max_val"));
        h->uni_vcmpgtps(val, val, c0);

        h->uni_vbroadcastss(c0, table_val("one"));
        h->uni_vpxor(sign, sign, c0);
        h->uni_vblendvps(polynom, polynom, sign, val);
    }
};

} // namespace MKLDNNPlugin
