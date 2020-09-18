// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>

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

class NopEmitter : public ngraph::snippet::JitEmitter {
public:
    NopEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "NopEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
    }
};

class StoreEmitter : public ngraph::snippet::JitEmitter {
public:
    StoreEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "StoreEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
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

class ScalarStoreEmitter : public ngraph::snippet::JitEmitter {
public:
    ScalarStoreEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "ScalarStoreEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
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
class LoadEmitter : public ngraph::snippet::JitEmitter {
public:
    LoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), ea(getEA(n)), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
        remark(10) << "LoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        // auto ea = getEA(op);
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

class BroadcastLoadEmitter : public ngraph::snippet::JitEmitter {
public:
    BroadcastLoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), ea(getEA(n)) {
        remark(10) << "BroadcastLoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
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

class ScalarLoadEmitter : public ngraph::snippet::JitEmitter {
public:
    ScalarLoadEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), ea(getEA(n)), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
        remark(10) << "ScalarLoadEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
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

class AddEmitter : public ngraph::snippet::JitEmitter {
public:
    AddEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "AddEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(10) << "  -> add" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vaddps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << out[0] << " = add (" << in[0] << ", " << in[1] << ")" << std::endl;
    }
};

class SubtractEmitter : public ngraph::snippet::JitEmitter{
public:
    SubtractEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "SubtractEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(10) << "  -> subtract" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << out[0] << " = subtract (" << in[0] << ", " << in[1] << ")" << std::endl;
    }
};

class MultiplyEmitter : public ngraph::snippet::JitEmitter{
public:
    MultiplyEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "MultiplyEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> mul" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vmulps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << out[0] << " = mul (" << in[0] << ", " << in[1] << ")" << std::endl;
    }
};

class DivideEmitter : public ngraph::snippet::JitEmitter{
public:
    DivideEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "DivideEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> div" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vdivps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << out[0] << " = div (" << in[0] << ", " << in[1] << ")" << std::endl;
    }
};

class NegativeEmitter : public ngraph::snippet::JitEmitter{
public:
    NegativeEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "NegativeEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
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

class ReluEmitter : public ngraph::snippet::JitEmitter{
public:
    ReluEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "ReluEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> relu" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(out[0]);
        Xbyak::Ymm vmm_mask = Xbyak::Ymm(out[0]+1);

        h->uni_vpxor(vmm_dst0, vmm_dst0, vmm_dst0);
        h->vcmpgtps(vmm_mask, vmm_src0, vmm_dst0);
        h->vblendvps(vmm_dst0, vmm_dst0, vmm_src0, vmm_mask);

        remark(11) << "    -> " << out[0] << " = relu (" << in[0] << ")" << std::endl;
    }
};

class PReluEmitter : public ngraph::snippet::JitEmitter{
public:
    PReluEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "PReluEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> prelu " << std::endl;

        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        Xbyak::Ymm vmm_tmp  = Xbyak::Ymm(out[0]+1);

        h->vmulps(vmm_dst, vmm_src0, vmm_src1);
        h->vxorps(vmm_tmp, vmm_tmp, vmm_tmp);
        h->vcmpgtps(vmm_tmp, vmm_src0, vmm_tmp);
        h->vblendvps(vmm_dst, vmm_dst, vmm_src0, vmm_tmp);

        remark(11) << "    -> " << out[0] << " = prelu (" << in[0] << "," << in[1] << ")" << std::endl;
    }
};

class FakeBroadcastEmitter : public ngraph::snippet::JitEmitter{
public:
    FakeBroadcastEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), use_broadcast(*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin()) {
        remark(10) << "FakeBroadcastEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        // Fix me: make it bypass nop
        remark(11) << "fake broadcast "<< std::endl;

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

class SquaredDifferenceEmitter : public ngraph::snippet::JitEmitter{
public:
    SquaredDifferenceEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "SquaredDifferenceEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> diff" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst = Xbyak::Ymm(out[0]);

        h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);
        h->uni_vmulps(vmm_dst, vmm_dst, vmm_dst);

        remark(11) << "    -> " << out[0] << " = diff (" << in[0] << "," << in[1] << ")" << std::endl;
    }
};

class PowerEmitter : public ngraph::snippet::JitEmitter{
public:
    PowerEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n) {
        remark(10) << "PowerEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
        auto parent = n->input(1).get_source_output().get_node_shared_ptr();
        if (!std::dynamic_pointer_cast<ngraph::op::Scalar>(parent)) {
            throw ngraph::ngraph_error("unsupported non constant power");
        }

        if (!(n->input(1).get_shape() == ngraph::Shape() || ngraph::shape_size(n->input(1).get_shape()) == 1)) {
            throw ngraph::ngraph_error("unsupported non scalar power");
        }
        order = ngraph::as_type_ptr<ngraph::op::Scalar>(parent)->get_data_ptr<float>()[0];
        std::cout << "order = " << order << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> pow" << std::endl;
        Xbyak::Ymm vmm_src = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst = Xbyak::Ymm(out[0]);

        if (order == -0.5f) {
            h->uni_vsqrtps(vmm_src, vmm_src);
            h->uni_vpcmpeqd(vmm_dst, vmm_dst, vmm_dst);
            h->uni_vpslld(vmm_dst, vmm_dst, 25);
            h->uni_vpsrld(vmm_dst, vmm_dst, 2);
            h->uni_vdivps(vmm_dst, vmm_dst, vmm_src);
        } else if (order == 2.0f) {
            h->uni_vmulps(vmm_dst, vmm_src, vmm_src);
        } else if (order == 0.5f) {
            h->uni_vsqrtps(vmm_dst, vmm_src);
        } else if (order == -1.0f) {
            h->uni_vpcmpeqd(vmm_dst, vmm_dst, vmm_dst);
            h->uni_vpslld(vmm_dst, vmm_dst, 25);
            h->uni_vpsrld(vmm_dst, vmm_dst, 2);
            h->uni_vdivps(vmm_dst, vmm_dst, vmm_src);
        } else {
            throw ngraph::ngraph_error("unsupported power value " + std::to_string(order));
        }

        remark(11) << "    -> " << out[0] << " = pow (" << in[0] << "," << in[1] << ")" << std::endl;
    }

private:
    float order;
};

class ScalarEmitter : public ngraph::snippet::JitEmitter{
public:
    ScalarEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), offset(getTableOffset(n))  {
        remark(10) << "ScalarEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << " scalar constant" << std::endl;
        Xbyak::Ymm vmm = Xbyak::Ymm(out[0]);
        h->uni_vbroadcastss(vmm, h->ptr[p_table + offset*sizeof(float)]);
        remark(11) << "    -> " << out[0] << " = const (" << ")" << std::endl;
    }

private:
    size_t offset;
};

class SigmoidEmitter : public ngraph::snippet::JitEmitter{
public:
    SigmoidEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), offset(getTableOffset(n)) {
        remark(10) << "SigmoidEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool = {},
              const std::vector<size_t>& gpr  = {}) const override {
        remark(11) << "  -> sigmoid" << std::endl;

        Xbyak::Ymm vmm_src = Xbyak::Ymm(in[0]);

        Xbyak::Ymm vmm_mask = Xbyak::Ymm(out[0]);
        Xbyak::Ymm vmm_aux1 = Xbyak::Ymm(out[0]+1);
        Xbyak::Ymm vmm_aux2 = Xbyak::Ymm(out[0]+2);
        Xbyak::Ymm vmm_aux3 = Xbyak::Ymm(out[0]+3);

        h->uni_vmovups(vmm_aux3, vmm_src);
        h->uni_vandps(vmm_aux3, vmm_aux3, h->ptr[p_table + (offset+12*8)*sizeof(float)]);
        h->uni_vorps(vmm_src, vmm_src, h->ptr[p_table + (offset+12*8)*sizeof(float)]);

        // exp

        // get mask of values lower than log(FLT_MIN) to zero them in the output
        h->vcmpltps(vmm_mask, vmm_src, h->ptr[p_table + (offset+11*8)*sizeof(float)]);

        h->uni_vminps(vmm_src, vmm_src, h->ptr[p_table + (offset+10*8)*sizeof(float)]);
        h->uni_vmaxps(vmm_src, vmm_src, h->ptr[p_table + (offset+11*8)*sizeof(float)]);
        h->uni_vmovups(vmm_aux1, vmm_src);
        //calculate exp(x)
        // fx = x * log2ef + 0.5
        h->uni_vmulps(vmm_src, vmm_src, h->ptr[p_table + (offset+2*8)*sizeof(float)]);
        h->uni_vaddps(vmm_src, vmm_src, h->ptr[p_table + (offset+1*8)*sizeof(float)]);

        // tmp = floorf(fx)
        h->uni_vroundps(vmm_aux2, vmm_src, mkldnn::impl::cpu::jit_generator::_op_floor);

        //keep fx for further computations
        h->uni_vmovups(vmm_src, vmm_aux2); //vmm_src = fx

        //x = x - fx * ln2
        h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, h->ptr[p_table + (offset+3*8)*sizeof(float)]);

        // compute 2^n
        h->uni_vcvtps2dq(vmm_aux2, vmm_src);
        h->uni_vpaddd(vmm_aux2, vmm_aux2, h->ptr[p_table + (offset+4*8)*sizeof(float)]);
        h->uni_vpslld(vmm_aux2, vmm_aux2, 23); //Vmm(6) = 2^-fx

        // use vmm_src as tmp vmm_zero when applying mask
        h->uni_vpxor(vmm_src, vmm_src, vmm_src);
        // set zeroes according to the mask
        h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);

        // y = p5
        h->uni_vmovups(vmm_src, h->ptr[p_table + (offset+9*8)*sizeof(float)]);
        // y = y * x + p4
        h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (offset+8*8)*sizeof(float)]);
        // y = y * x + p3
        h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (offset+7*8)*sizeof(float)]);
        // y = y * x + p2
        h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (offset+6*8)*sizeof(float)]);
        // y = y * x + p1
        h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (offset+0*8)*sizeof(float)]);
        // y = y * x + p0
        h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (offset+5*8)*sizeof(float)]);  //exp(q)
        // y = y * 2^n
        h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

        ////

        // dup exp(x)
        h->uni_vmovups(vmm_aux1, vmm_src);
        // (exp(x) + 1)
        h->uni_vaddps(vmm_aux1, vmm_aux1, h->ptr[p_table + (offset+0*8)*sizeof(float)]);
        // y = exp(x) / (exp(x) + 1)
        h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

        // Now we have to apply the "symmetry" based on original sign
        h->uni_vmovups(vmm_aux2, h->ptr[p_table + (offset+0*8)*sizeof(float)]);
        h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
        h->uni_vmovups(vmm_mask, vmm_aux3);// The mask should be xmm0 for sse4.1
        h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);
        h->uni_vmovups(vmm_mask, vmm_aux2);

        remark(11) << "    -> " << out[0] << " = sigmoid (" << in[0] << ")" << std::endl;
    }

private:
    size_t offset;
};

class ClampEmitter : public ngraph::snippet::JitEmitter {
public:
    ClampEmitter(mkldnn::impl::cpu::jit_generator* h, mkldnn::impl::cpu::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : JitEmitter(h, isa, n), offset(getTableOffset(n)) {
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

private:
    size_t offset;
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
