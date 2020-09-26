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

class NopEmitter : public MKLDNNPlugin::jit_emitter {
public:
    NopEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        remark(10) << "NopEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
        // prepare_table();
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
        // h->uni_vbroadcastss(vmm, h->ptr[p_table + offset*sizeof(float)]);
        h->uni_vbroadcastss(vmm, table_val("scalar"));
        remark(11) << "    -> " << out[0] << " = const (" << ")" << std::endl;
    }

    // void emit_table() override {
    //     h->dd(value);
    // }

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ClampEmitter : public ngraph::snippet::JitEmitter {
public:
    ClampEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), offset(getTableOffset(n)) {
        auto op = ngraph::as_type_ptr<ngraph::opset1::Clamp>(n);
        remark(11) << "pugging Clamp min " << op->get_min() << " to the stack" << std::endl;
        vmin = mkldnn::impl::cpu::float2int(static_cast<float>(op->get_min()));
        remark(11) << "pugging Clamp max " << op->get_max() << " to the stack" << std::endl;
        vmax = mkldnn::impl::cpu::float2int(static_cast<float>(op->get_max()));
        remark(10) << "ClampEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> clamp" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(out[0]);
        Xbyak::Ymm vmm_max = Xbyak::Ymm(out[0]+1);

        h->uni_vbroadcastss(vmm_dst0, h->ptr[p_table + (offset+0)*sizeof(float)]);
        h->uni_vbroadcastss(vmm_max, h->ptr[p_table + (offset+1)*sizeof(float)]);

        h->uni_vmaxps(vmm_dst0, vmm_src0, vmm_dst0);
        h->uni_vminps(vmm_dst0, vmm_dst0, vmm_max);

        remark(1) << "    -> " << out[0] << " = clamp (" << in[0] << ")" << std::endl;
    }

    void emit_table() override {
        h->dd(vmin);
        h->dd(vmax);
    }

private:
    size_t offset;
    int32_t vmin;
    int32_t vmax;
};

class ErfEmitter : public ngraph::snippet::JitEmitter {
public:
    ErfEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), offset(getTableOffset(n)) {
        remark(10) << "ErfEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        // FIXME: get from target machine conefig in future
        Xbyak::Reg64 p_table { Xbyak::util::rax };
        remark(10) << "  -> erf " << offset << std::endl;

        auto regIDx = out[0];
        decltype(regIDx) latest = 15;

        Xbyak::Ymm polynom = Xbyak::Ymm(regIDx);
        h->uni_vbroadcastss(polynom, h->ptr[p_table + (offset + 11)*sizeof(float)]);

        Xbyak::Ymm x = Xbyak::Ymm(in[0]);
        Xbyak::Ymm x2 = Xbyak::Ymm(regIDx+1);
        Xbyak::Ymm c0 = Xbyak::Ymm(regIDx+2);
        Xbyak::Ymm val = Xbyak::Ymm(regIDx+3);
        Xbyak::Ymm sign = Xbyak::Ymm(regIDx+4);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 12)*sizeof(float)]);
        h->uni_vandps(sign, x, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 13)*sizeof(float)]);
        h->uni_vandps(val, x, c0);

        h->uni_vmulps(x2, x, x);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 2)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 3)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 4)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 5)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        // x *= polynom;
        h->uni_vmulps(x, x, polynom);

        h->uni_vbroadcastss(polynom, h->ptr[p_table + (offset + 1)*sizeof(float)]);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 6)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 7)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 8)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 9)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 10)*sizeof(float)]);
        h->uni_vfmadd213ps(polynom, x2, c0);

        // return x / polynom;
        h->uni_vdivps(polynom, x, polynom);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 0)*sizeof(float)]);
        h->uni_vcmpgtps(val, val, c0);

        h->uni_vbroadcastss(c0, h->ptr[p_table + (offset + 1)*sizeof(float)]);
        h->uni_vpxor(sign, sign, c0);
        h->uni_vblendvps(polynom, polynom, sign, val);

        remark(10) << "Free registers " << latest - regIDx + 1 << std::endl;
        remark(10) << "    -> " << out[0] << " = erf (" << in[0] << ")" << std::endl;
    }

    void emit_table() override {
        remark(10) << "generating table for Erf" << std::endl;
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

private:
    size_t offset;
};
