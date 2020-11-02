// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include "emitter.hpp"

static inline auto getTableOffset(const std::shared_ptr<ngraph::Node>& n) -> size_t {
    auto rt = n->get_rt_info();

    size_t rout = 0;;
    if (auto rinfo = rt["stackinfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
        rout = static_cast<size_t>(reginfo);
    }

    return rout;
}

static inline auto getEA(const std::shared_ptr<ngraph::Node>& n) -> size_t {
    auto& rt = n->get_rt_info();
    size_t ea = 0;
    if (auto rinfo = rt["effectiveAddress"]) {
        ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
    } else {
        throw ngraph::ngraph_error("effective address for Load generation cannot be determined");
    }
    return ea;
}

class KernelEmitter : public MKLDNNPlugin::jit_emitter {
public:
    KernelEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa,
    const std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippet::RegInfo>> c)
    : jit_emitter(h, isa, nullptr) {
        code = c;
        remark(10) << "KernelEmitter: " << std::endl;
    }

    size_t get_inputs_num() override {return 0;}

    void emit(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool = {},
                   const std::vector<size_t>& gpr  = {}) const override {
        auto ins = in[0];
        auto outs = in[1];

        h->preamble();
        Xbyak::Reg64 amount = Xbyak::Reg64(reg64_tmp_start+ins+outs);

        std::vector<Xbyak::Reg64> regs(ins+outs);
        for (auto i = 0; i < regs.size(); i++) {
            regs[i] = Xbyak::Reg64(reg64_tmp_start+i);
        }

        for (auto i = 0; i < ins; i++) {
            h->mov(regs[i], h->ptr[param + i*sizeof(size_t)]);
        }

        for (auto i = 0; i < outs; i++) {
            h->mov(regs[ins+i], h->ptr[param + (ins+i)*sizeof(size_t)]);
        }

        h->mov(amount, h->ptr[param + sizeof(size_t)*(ins+outs)]);

        for (auto& c : code) {
            c.first->emit(c.second.first, c.second.second);
        }

        h->postamble();
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippet::RegInfo>> code;
};

class TileEmitter : public MKLDNNPlugin::jit_emitter {
public:
    TileEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa,
    const std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippet::RegInfo>> c)
    : jit_emitter(h, isa, nullptr) {
        code = c;
        remark(10) << "TileEmitter: " << std::endl;
    }

    size_t get_inputs_num() override {return 0;}

    void emit(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool = {},
                   const std::vector<size_t>& gpr  = {}) const override {
        auto nparams = in[1];
        Xbyak::Reg64 amount = Xbyak::Reg64(reg64_tmp_start+nparams);
        size_t nloads   = in[0];

        std::array<Xbyak::Label, 2> for_body;
        // loop_entry()
        h->cmp(amount, nloads);
        h->jl(for_body[1], Xbyak::CodeGenerator::T_NEAR);

        // loop_body()
        h->L(for_body[0]); {
            for (auto& c : code) {
                c.first->emit(c.second.first, c.second.second);
            }
            // loop_advance()
            h->sub(amount, nloads);
            h->cmp(amount, nloads);
            h->jge(for_body[0], Xbyak::CodeGenerator::T_NEAR);
        }

        h->L(for_body[1]);
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippet::RegInfo>> code;
};

class NopEmitter : public MKLDNNPlugin::jit_emitter {
public:
    NopEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        remark(10) << "NopEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    size_t get_inputs_num() override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool = {},
                   const std::vector<size_t>& gpr  = {}) const override {
    }
};

class StoreEmitter : public MKLDNNPlugin::jit_emitter  {
public:
    StoreEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "StoreEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    size_t get_inputs_num() override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> node is a store " << std::endl;

        Xbyak::Reg64 out_reg(reg64_tmp_start + ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        h->uni_vmovups(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen);

        remark(11) << "    -> store (" << in[0] << ")" << std::endl;
    }

private:
    size_t ea;
};

class ScalarStoreEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ScalarStoreEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "ScalarStoreEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> node is a scalar_store " << std::endl;

        Xbyak::Reg64 out_reg(reg64_tmp_start + ea);
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(in[0]);
        h->movss(h->ptr[out_reg], xmm_src0);
        h->add(out_reg, sizeof(float));

        remark(11) << "    -> scalar_store (" << in[0] << ")" << std::endl;
    }

private:
    size_t ea;
};

/// Assumption that every parameter loaded from memory only one should be correct
/// Every parameter loaded from memory once. Broadcast doesn’t need post increment.
/// For scalar loads we can use different tiles. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class LoadEmitter : public MKLDNNPlugin::jit_emitter {
public:
    LoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
        remark(10) << "LoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> node is a load " << ea << " " << shouldPostIncrement << std::endl;

        Xbyak::Reg64 in_reg(reg64_tmp_start + ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(out[0]);
        h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

        if (shouldPostIncrement) {
            remark(11) << "adding post increment" << std::endl;
            h->add(in_reg, mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen);
        }

        remark(11) << " -> load (" << (out[0]) << ") " << std::endl;
    }

private:
    size_t ea;
    bool shouldPostIncrement;
};

class BroadcastLoadEmitter : public MKLDNNPlugin::jit_emitter {
public:
    BroadcastLoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "BroadcastLoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> node is a broadcast load " << ea << std::endl;

        Xbyak::Reg64 in_reg(reg64_tmp_start + ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(out[0]);
        // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
        // key point here is not to add post-increment, it might be fixed by some other approach in future
        h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);

        remark(11) << "    -> broadcast (" << (out[0]) << ")" << std::endl;
    }

private:
    size_t ea;
};

class ScalarLoadEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ScalarLoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
        remark(10) << "ScalarLoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> node is a scalar_load " << ea << " " << shouldPostIncrement << std::endl;

        Xbyak::Reg64 in_reg(reg64_tmp_start + ea);
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(out[0]);
        h->movss(xmm_src0, h->ptr[in_reg]);

        // FIXME: something fundamentally wrong with this condition, it addresses the case if
        if (shouldPostIncrement) {
            remark(11) << "adding post increment" << std::endl;
            h->add(in_reg, sizeof(float));
        }

        remark(11) << " -> scalar_load (" << (out[0]) << ") " << std::endl;
    }

private:
    size_t ea;
    bool shouldPostIncrement;
};

class FakeBroadcastEmitter : public MKLDNNPlugin::jit_emitter {
public:
    FakeBroadcastEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), use_broadcast(*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin()) {
        remark(10) << "FakeBroadcastEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        // Fix me: make it bypass nop
        remark(11) << "fake broadcast "<< use_broadcast << std::endl;

        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);

        if (use_broadcast) {
            h->uni_vbroadcastss(vmm_dst, Xbyak::Xmm(in[0]));
        } else {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }

        remark(11) << "    -> " << out[0] << " = broadcast (" << in[0] << ")" << std::endl;
    }

private:
    bool use_broadcast;
};

class ScalarEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ScalarEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), offset(getTableOffset(n))  {
        auto out_shape = n->output(0).get_tensor().get_shape();
        remark(10) << "ScalarEmitter: " << n->get_friendly_name() << " " << n->get_type_info().name << " " << out_shape << std::endl;
        if (out_shape == ngraph::Shape() || ngraph::shape_size(out_shape) == 1) {
            remark(11) << "pugging constant " << ngraph::as_type_ptr<ngraph::op::Scalar>(n)->cast_vector<float>()[0] << " to the stack" << std::endl;
            value = mkldnn::impl::cpu::float2int(ngraph::as_type_ptr<ngraph::op::Scalar>(n)->cast_vector<float>()[0]);
        }

        push_arg_entry_of("scalar", value, false);
        prepare_table();
    }
    size_t get_inputs_num() override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << " scalar constant" << std::endl;
        Xbyak::Ymm vmm = Xbyak::Ymm(out[0]);
        h->uni_vbroadcastss(vmm, table_val("scalar"));
        remark(11) << "    -> " << out[0] << " = const (" << ")" << std::endl;
    }

private:
    size_t offset;
    int32_t value;
};

class ConvertEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ConvertEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), offset(getTableOffset(n))  {
        // auto out_shape = n->output(0).get_tensor().get_shape();
        // remark(10) << "ScalarEmitter: " << n->get_friendly_name() << " " << n->get_type_info().name << " " << out_shape << std::endl;
        // if (out_shape == ngraph::Shape() || ngraph::shape_size(out_shape) == 1) {
        //     remark(11) << "pugging constant " << ngraph::as_type_ptr<ngraph::op::Scalar>(n)->cast_vector<float>()[0] << " to the stack" << std::endl;
        //     value = mkldnn::impl::cpu::float2int(ngraph::as_type_ptr<ngraph::op::Scalar>(n)->cast_vector<float>()[0]);
        // }

        // push_arg_entry_of("scalar", value, false);
        // prepare_table();
    }
    size_t get_inputs_num() override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << " scalar constant" << std::endl;
        Xbyak::Ymm vmm = Xbyak::Ymm(out[0]);
        h->uni_vbroadcastss(vmm, table_val("scalar"));
        remark(11) << "    -> " << out[0] << " = const (" << ")" << std::endl;
    }

private:
    size_t offset;
    int32_t value;
};

class NegativeEmitter : public MKLDNNPlugin::jit_emitter {
public:
    NegativeEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        remark(10) << "NegativeEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }
    size_t get_inputs_num() override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> neg" << std::endl;
        Xbyak::Ymm vmm_src = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
        h->uni_vsubps(vmm_dst, vmm_dst, vmm_src);
        remark(11) << "    -> " << out[0] << " = neg (" << in[0] << ")" << std::endl;
    }
};

class ClampEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ClampEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), offset(getTableOffset(n)) {
        auto op = ngraph::as_type_ptr<ngraph::opset1::Clamp>(n);
        remark(11) << "pugging Clamp min " << op->get_min() << " to the stack" << std::endl;
        vmin = mkldnn::impl::cpu::float2int(static_cast<float>(op->get_min()));
        remark(11) << "pugging Clamp max " << op->get_max() << " to the stack" << std::endl;
        vmax = mkldnn::impl::cpu::float2int(static_cast<float>(op->get_max()));
        remark(10) << "ClampEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;

        push_arg_entry_of("vmin", vmin, false);
        push_arg_entry_of("vmax", vmax, false);
        prepare_table();
    }
    size_t get_inputs_num() override {return 1;}

protected:
    size_t aux_gprs_count() const override {return 1;}
    size_t aux_vecs_count() const override {return 1;}

    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> clamp" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(out[0]);
        Xbyak::Ymm vmm_max = Xbyak::Ymm(aux_vec_idxs[0]);

        h->uni_vbroadcastss(vmm_dst0, table_val("vmin"));
        h->uni_vbroadcastss(vmm_max, table_val("vmax"));
        h->uni_vmaxps(vmm_dst0, vmm_src0, vmm_dst0);
        h->uni_vminps(vmm_dst0, vmm_dst0, vmm_max);

        remark(1) << "    -> " << out[0] << " = clamp (" << in[0] << ")" << std::endl;
    }

private:
    size_t offset;
    int32_t vmin;
    int32_t vmax;
};

class ErfEmitter : public MKLDNNPlugin::jit_emitter {
public:
    ErfEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), offset(getTableOffset(n)) {
        remark(10) << "ErfEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;

        push_arg_entry_of("max_val", mkldnn::impl::cpu::float2int(2.86f), false);
        push_arg_entry_of("one", 0x3f800000, false);

        push_arg_entry_of("p1", mkldnn::impl::cpu::float2int(90.0260162353515625f), false);
        push_arg_entry_of("p2", mkldnn::impl::cpu::float2int(2232.00537109375000f), false);
        push_arg_entry_of("p3", mkldnn::impl::cpu::float2int(7003.32519531250000f), false);
        push_arg_entry_of("p4", mkldnn::impl::cpu::float2int(55592.3007812500000f), false);

        push_arg_entry_of("rp0", mkldnn::impl::cpu::float2int(33.56171417236328125f), false);
        push_arg_entry_of("rp1", mkldnn::impl::cpu::float2int(521.35797119140625f), false);
        push_arg_entry_of("rp2", mkldnn::impl::cpu::float2int(4594.32373046875f), false);
        push_arg_entry_of("rp3", mkldnn::impl::cpu::float2int(22629.0f), false);
        push_arg_entry_of("rp4", mkldnn::impl::cpu::float2int(49267.39453125f), false);

        push_arg_entry_of("p0", mkldnn::impl::cpu::float2int(9.60497379302978515625f), false);

        push_arg_entry_of("exp", 0x80000000, false);
        push_arg_entry_of("sign", 0x7fffffff, false);

        prepare_table();
    }

    size_t get_inputs_num() override {return 1;}

protected:
    size_t aux_gprs_count() const override {return 1;}
    size_t aux_vecs_count() const override {return 4;}

    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(10) << "  -> erf " << offset << std::endl;

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

        remark(10) << "    -> " << out[0] << " = erf (" << in[0] << ")" << std::endl;
    }

private:
    size_t offset;
};
