// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

class AddEmitter : public ngraph::snippet::Emitter {
public:
    AddEmitter(mkldnn::impl::cpu::jit_generator* h, const std::shared_ptr<ngraph::Node>& n)
    : Emitter(h, n) {
        remark(10) << "AddEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(std::vector<size_t> in, std::vector<size_t> out, std::vector<size_t> pool) const override {
        remark(10) << "  -> add" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vaddps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << in[0] << " = add (" << in[0] << ", " << out[1] << ")" << std::endl;
    }
};

class SubtractEmitter : public ngraph::snippet::Emitter{
public:
    SubtractEmitter(mkldnn::impl::cpu::jit_generator* h, const std::shared_ptr<ngraph::Node>& n)
    : Emitter(h, n) {
        remark(10) << "SubtractEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(std::vector<size_t> in, std::vector<size_t> out, std::vector<size_t> pool) const override {
        remark(10) << "  -> subtract" << std::endl;
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(in[0]);
        Xbyak::Ymm vmm_src1 = Xbyak::Ymm(in[1]);
        Xbyak::Ymm vmm_dst  = Xbyak::Ymm(out[0]);
        h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);
        remark(10) <<"    -> " << in[0] << " = subtract (" << in[0] << ", " << out[1] << ")" << std::endl;
    }
};

class ErfEmitter : public ngraph::snippet::Emitter {
public:
    ErfEmitter(mkldnn::impl::cpu::jit_generator* h, const std::shared_ptr<ngraph::Node>& n)
    : Emitter(h, n) {
        remark(10) << "ErfEmitter: " << n->get_friendly_name() << n->get_type_info().name << std::endl;
    }

    void emit(std::vector<size_t> in, std::vector<size_t> out, std::vector<size_t> pool) const override {
        h->mov(p_table, table);
        // auto n = as_type_ptr<Node>(op);
        auto offset = p_table.getIdx();// getTableOffset(n);
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
        h->align(64);
        h->L(table); // L disallow this function be const qualified

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
    Xbyak::Label table;
    const Xbyak::Reg64 p_table { Xbyak::util::rax }; // get from somewhere
};
