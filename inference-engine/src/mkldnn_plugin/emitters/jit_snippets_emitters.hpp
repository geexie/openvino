// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_emitter.hpp"

namespace MKLDNNPlugin {

class KernelEmitter : public jit_emitter {
public:
    KernelEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    const std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> c)
    : jit_emitter(h, isa, nullptr) {
        code = c;
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
        auto ins = in[0];
        auto outs = in[1];
        int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        Xbyak::Reg64 param  { dnnl::impl::cpu::x64::abi_param1 }; // RDI
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
            c.first->emit_code(c.second.first, c.second.second);
        }

        h->postamble();
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};

class TileEmitter : public jit_emitter {
public:
    TileEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa,
    const std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> c)
    : jit_emitter(h, isa, nullptr) {
        code = c;
    }

    size_t get_inputs_num() const override {return 0;}

    void emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
              const std::vector<size_t> &pool = {}, const std::vector<size_t> &gpr = {}) const override {
        emit_impl(in, out, pool, gpr, nullptr);
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
        auto nparams = in[1];
        int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
        Xbyak::Reg64 amount = Xbyak::Reg64(reg64_tmp_start+nparams);
        size_t nloads   = in[0];

        std::array<Xbyak::Label, 2> for_body;
        // loop_entry()
        h->cmp(amount, nloads);
        h->jl(for_body[1], Xbyak::CodeGenerator::T_NEAR);

        // loop_body()
        h->L(for_body[0]); {
            for (auto& c : code) {
                c.first->emit_code(c.second.first, c.second.second);
            }
            // loop_advance()
            h->sub(amount, nloads);
            h->cmp(amount, nloads);
            h->jge(for_body[0], Xbyak::CodeGenerator::T_NEAR);
        }

        h->L(for_body[1]);
    }

    std::vector<std::pair<std::shared_ptr<Emitter>, ngraph::snippets::RegInfo>> code;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const MKLDNNPlugin::emitter_context *emit_context) const override {
    }
};

class FakeBroadcastEmitter : public jit_emitter {
public:
    FakeBroadcastEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), use_broadcast(*n->get_input_shape(0).rbegin() != *n->get_output_shape(0).rbegin()) {
    }
    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        // Fix me: make it bypass nop
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);

        if (use_broadcast) {
            h->uni_vbroadcastss(vmm_dst, Xbyak::Xmm(in[0]));
        } else {
            h->uni_vmovups(vmm_dst, vmm_src0);
        }
    }

private:
    bool use_broadcast;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n) {
        auto out_shape = n->output(0).get_tensor().get_shape();
        if (out_shape == ngraph::Shape() || ngraph::shape_size(out_shape) == 1) {
            value = mkldnn::impl::cpu::x64::float2int(ngraph::as_type_ptr<ngraph::op::Scalar>(n)->cast_vector<float>()[0]);
        }

        push_arg_entry_of("scalar", value, false);
        prepare_table();
    }

    size_t get_inputs_num() const override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Ymm vmm = Xbyak::Ymm(out[0]);
        h->uni_vbroadcastss(vmm, table_val("scalar"));
    }

private:
    int32_t value;
};

///
/// Memory emitters:
///
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different tiles. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : jit_emitter(h, isa, n), ea(getEA(n)) {
    }

    size_t get_inputs_num() const override {return 1;}

protected:
    static auto getEA(const std::shared_ptr<ngraph::Node>& n) -> size_t {
        auto& rt = n->get_rt_info();
        size_t ea = 0;
        if (auto rinfo = rt["effectiveAddress"]) {
            ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
        } else {
            throw ngraph::ngraph_error("effective address for Load generation cannot be determined");
        }
        return ea;
    }

    size_t ea;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Reg64 out_reg(ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        h->uni_vmovups(h->ptr[out_reg], vmm_src0);
        h->add(out_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::avx2>::vlen);
    }
};

class ScalarStoreEmitter : public MemoryEmitter {
public:
    ScalarStoreEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Reg64 out_reg(ea);
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(in[0]);
        h->movss(h->ptr[out_reg], xmm_src0);
        h->add(out_reg, sizeof(float));
    }
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Reg64 in_reg(ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(out[0]);
        h->uni_vmovups(vmm_src0, h->ptr[in_reg]);

        if (shouldPostIncrement) {
            h->add(in_reg, mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::avx2>::vlen);
        }
    }

private:
    bool shouldPostIncrement;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Reg64 in_reg(ea);
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(out[0]);
        // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
        // key point here is not to add post-increment, it might be fixed by some other approach in future
        h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
    }
};

class ScalarLoadEmitter : public MemoryEmitter {
public:
    ScalarLoadEmitter(mkldnn::impl::cpu::x64::jit_generator* h, mkldnn::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ngraph::Node>& n)
    : MemoryEmitter(h, isa, n), shouldPostIncrement(*n->get_input_shape(0).rbegin() != 1) {
    }
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const MKLDNNPlugin::emitter_context *emit_context) const override {
        Xbyak::Reg64 in_reg(ea);
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(out[0]);
        h->movss(xmm_src0, h->ptr[in_reg]);

        // FIXME: something fundamentally wrong with this condition, it addresses the case if
        if (shouldPostIncrement) {
            h->add(in_reg, sizeof(float));
        }
    }

private:
    bool shouldPostIncrement;
};

} // namespace MKLDNNPlugin {