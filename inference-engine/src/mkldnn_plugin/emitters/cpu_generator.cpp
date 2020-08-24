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

auto getTableOffset(std::shared_ptr<ngraph::Node>& n) -> size_t {
    auto rt = n->get_rt_info();

    size_t rout = 0;;
    if (auto rinfo = rt["stackinfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
        rout = static_cast<size_t>(reginfo);
    }

    return rout;
}

CPUGenerator::CPUGenerator() : h(new jit_snippet()) {
    reg64_tmp_start = h->r8.getIdx();

    jitters[ngraph::opset1::Add().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new AddEmitter(h.get(), n));
    };

    jitters[ngraph::opset1::Subtract().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new SubtractEmitter(h.get(), n));
    };

    jitters[ngraph::opset1::Erf().get_type_info()]
    = [this](const std::shared_ptr<ngraph::Node>& n) -> std::shared_ptr<Emitter> {
        return std::shared_ptr<Emitter>(new ErfEmitter(h.get(), n));
    };
}

void CPUGenerator::emit_module_enter() {
    h->preamble();
}
void CPUGenerator::emit_module_exit() {
    h->postamble();
}

void CPUGenerator::emit(std::shared_ptr<opset1::Parameter>& param, RegInfo& registers, bool vec) const {
    remark(1) << "  -> node is a parameter with " << param->outputs().size() << " outputs"  << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<op::Load>& load, RegInfo& registers, bool vec) const {
    auto& rt = load->get_rt_info();
    size_t ea = 0;
    if (auto rinfo = rt["effectiveAddress"]) {
        ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
    } else {
        throw ngraph_error("effective address for Load generation cannot be determined");
    }

    remark(1) << "  -> node is a load " << ea << " " << *load->get_input_shape(0).rbegin() << std::endl;

    Xbyak::Reg64 in_reg(reg64_tmp_start + ea);

    if (vec /*&& (*load->get_input_shape(0).rbegin() != 1)*/) {
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.second[0]);
        h->uni_vmovups(vmm_src0, h->ptr[in_reg]);
    } else {
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(registers.second[0]);
        h->movss(xmm_src0, h->ptr[in_reg]);
    }

    if (*load->get_input_shape(0).rbegin() != 1)
        h->add(in_reg, vec ? mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen : sizeof(float));

    remark(1) << " -> load (" << (registers.second[0]) << ") " << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<op::BroadcastLoad>& load, RegInfo& registers, bool vec) const {
    auto& rt = load->get_rt_info();
    size_t ea = 0;
    if (auto rinfo = rt["effectiveAddress"]) {
        ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
    } else {
        throw ngraph_error("effective address for Load generation cannot be determined");
    }

    remark(1) << "  -> node is a broadcast load " << ea << " "
     << !!load->is_broadcast(load->get_output_shape(0).size() - 1) << " !!!!!" << std::endl;
    Xbyak::Reg64 in_reg(reg64_tmp_start + ea);

    auto should_broadcast_w = !!load->is_broadcast(load->get_output_shape(0).size() - 1);

    if (vec) {
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.second[0]);
        if (should_broadcast_w) {
            h->uni_vbroadcastss(vmm_src0, h->ptr[in_reg]);
        } else {
            h->uni_vmovups(vmm_src0, h->ptr[in_reg]);
        }
    } else {
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(registers.second[0]);
        h->movss(xmm_src0, h->ptr[in_reg]);
    }

    // Note: increment is needed if we cross line boundary over linearized tiles (currently disabled)
    if (!should_broadcast_w && *load->get_input_shape(0).rbegin() != 1) {
        remark(1) << "adding post increment" << std::endl;
        h->add(in_reg, vec ? mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen : sizeof(float));
    }

    remark(1) << "    -> broadcast (" << (registers.second[0]) << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Result>& result, RegInfo& registers, bool vec) const {
    auto& rt = result->get_rt_info();
    size_t ea = 0;
    if (auto rinfo = rt["effectiveAddress"]) {
        ea = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo)->get();
    } else {
        throw ngraph_error("effective address for Load generation cannot be determined");
    }

    remark(1) << "  -> node is a result " << std::endl;
    Xbyak::Reg64 out_reg(reg64_tmp_start + ea);
    if (vec) {
        Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
        h->uni_vmovups(h->ptr[out_reg], vmm_src0);
    } else {
        Xbyak::Xmm xmm_src0 = Xbyak::Xmm(registers.first[0]);
        h->movss(h->ptr[out_reg], xmm_src0);
    }
    h->add(out_reg, vec ? mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen : sizeof(float));

    remark(1) << "    -> store (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Add>& add, RegInfo& registers) const {
    remark(1) << "  -> add" << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_src1 = Xbyak::Ymm(registers.first[1]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    h->uni_vaddps(vmm_dst, vmm_src0, vmm_src1);

    remark(1) << "    -> " << registers.second[0] << " = add (" << registers.first[0] << ", " << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Subtract>& sub, RegInfo& registers) const {
    remark(1) << "  -> sub" << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_src1 = Xbyak::Ymm(registers.first[1]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);

    remark(1) << "    -> " << registers.second[0] << " = sub (" << registers.first[0] << ", " << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Multiply>& mul, RegInfo& registers) const {
    remark(1) << "  -> mul" << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_src1 = Xbyak::Ymm(registers.first[1]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    h->uni_vmulps(vmm_dst, vmm_src0, vmm_src1);

    remark(1) << "    -> " << registers.second[0] << " = mul (" << registers.first[0] << ", " << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Negative>& neg, RegInfo& registers) const {
    remark(1) << "  -> neg" << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(registers.second[0]);

    h->uni_vpxor(vmm_dst0, vmm_dst0, vmm_dst0);
    h->uni_vsubps(vmm_dst0, vmm_dst0, vmm_src0);

    remark(1) << "    -> " << registers.second[0] << " = neg (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Divide>& op, RegInfo& registers) const {
    remark(1) << "  -> div" << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_src1 = Xbyak::Ymm(registers.first[1]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    h->uni_vdivps(vmm_dst, vmm_src0, vmm_src1);

    remark(1) << "    -> " << registers.second[0] << " = div (" << registers.first[0] << ", " << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Clamp>& op, RegInfo& registers) const {
    remark(1) << "  -> clamp" << std::endl;
    // auto x = as_type_ptr<Node>(op);
    auto x = std::dynamic_pointer_cast<Node>(op);
    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(registers.second[0]);
    Xbyak::Ymm vmm_max = Xbyak::Ymm(registers.second[0]+1);

    h->uni_vbroadcastss(vmm_dst0, h->ptr[p_table + (getTableOffset(x)+0)*sizeof(float)]);
    h->uni_vbroadcastss(vmm_max, h->ptr[p_table + (getTableOffset(x)+1)*sizeof(float)]);

    h->uni_vmaxps(vmm_dst0, vmm_src0, vmm_dst0);
    h->uni_vminps(vmm_dst0, vmm_dst0, vmm_max);

    remark(1) << "    -> " << registers.second[0] << " = clamp (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Relu>& op, RegInfo& registers) const {
    remark(1) << "  -> relu" << std::endl;
    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst0 = Xbyak::Ymm(registers.second[0]);
    Xbyak::Ymm vmm_mask = Xbyak::Ymm(registers.second[0]+1);

    h->uni_vpxor(vmm_dst0, vmm_dst0, vmm_dst0);
    h->vcmpgtps(vmm_mask, vmm_src0, vmm_dst0);
    h->vblendvps(vmm_dst0, vmm_dst0, vmm_src0, vmm_mask);

    remark(1) << "    -> " << registers.second[0] << " = relu (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::SquaredDifference>& op, RegInfo& registers) const {
    remark(1) << "  -> diff" << std::endl;
    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_src1 = Xbyak::Ymm(registers.first[1]);
    Xbyak::Ymm vmm_dst = Xbyak::Ymm(registers.second[0]);

    h->uni_vsubps(vmm_dst, vmm_src0, vmm_src1);
    h->uni_vmulps(vmm_dst, vmm_dst, vmm_dst);

    remark(1) << "    -> " << registers.second[0] << " = diff (" << registers.first[0] << "," << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Power>& op, RegInfo& registers) const {
    remark(1) << "  -> pow" << std::endl;
    Xbyak::Ymm vmm_src = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst = Xbyak::Ymm(registers.second[0]);

    auto parent = op->input(1).get_source_output().get_node_shared_ptr();
    if (!std::dynamic_pointer_cast<opset1::Constant>(parent)) {
        throw ngraph_error("unsupported non constant power");
    }

    if (op->input(1).get_shape() == Shape() || ngraph::shape_size(op->input(1).get_shape()) == 1) {
        auto order = as_type_ptr<opset1::Constant>(parent)->get_data_ptr<float>()[0];
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
            throw ngraph_error("unsupported power value " + std::to_string(order));
        }
    } else {
        throw ngraph_error("unsupported non scalar power");
    }

    remark(1) << "    -> " << registers.second[0] << " = pow (" << registers.first[0] << "," << registers.first[1] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Sigmoid>& op, RegInfo& registers) const {
    remark(11) << "  -> sigmoid" << std::endl;
    // auto x = as_type_ptr<Node>(op);
    auto x = std::dynamic_pointer_cast<Node>(op);
    auto tbl_offset = getTableOffset(x);

    Xbyak::Ymm vmm_src = Xbyak::Ymm(registers.first[0]);

    Xbyak::Ymm vmm_mask = Xbyak::Ymm(registers.second[0]);
    Xbyak::Ymm vmm_aux1 = Xbyak::Ymm(registers.second[0]+1);
    Xbyak::Ymm vmm_aux2 = Xbyak::Ymm(registers.second[0]+2);
    Xbyak::Ymm vmm_aux3 = Xbyak::Ymm(registers.second[0]+3);

    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vandps(vmm_aux3, vmm_aux3, h->ptr[p_table + (tbl_offset+12*8)*sizeof(float)]);
    h->uni_vorps(vmm_src, vmm_src, h->ptr[p_table + (tbl_offset+12*8)*sizeof(float)]);

    // exp

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->vcmpltps(vmm_mask, vmm_src, h->ptr[p_table + (tbl_offset+11*8)*sizeof(float)]);

    h->uni_vminps(vmm_src, vmm_src, h->ptr[p_table + (tbl_offset+10*8)*sizeof(float)]);
    h->uni_vmaxps(vmm_src, vmm_src, h->ptr[p_table + (tbl_offset+11*8)*sizeof(float)]);
    h->uni_vmovups(vmm_aux1, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, h->ptr[p_table + (tbl_offset+2*8)*sizeof(float)]);
    h->uni_vaddps(vmm_src, vmm_src, h->ptr[p_table + (tbl_offset+1*8)*sizeof(float)]);

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, mkldnn::impl::cpu::jit_generator::_op_floor);

    //keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2); //vmm_src = fx

    //x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, h->ptr[p_table + (tbl_offset+3*8)*sizeof(float)]);

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    h->uni_vpaddd(vmm_aux2, vmm_aux2, h->ptr[p_table + (tbl_offset+4*8)*sizeof(float)]);
    h->uni_vpslld(vmm_aux2, vmm_aux2, 23); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_src, vmm_src, vmm_src);
    // set zeroes according to the mask
    h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);

    // y = p5
    h->uni_vmovups(vmm_src, h->ptr[p_table + (tbl_offset+9*8)*sizeof(float)]);
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (tbl_offset+8*8)*sizeof(float)]);
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (tbl_offset+7*8)*sizeof(float)]);
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (tbl_offset+6*8)*sizeof(float)]);
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (tbl_offset+0*8)*sizeof(float)]);
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, h->ptr[p_table + (tbl_offset+5*8)*sizeof(float)]);  //exp(q)
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    ////

    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, h->ptr[p_table + (tbl_offset+0*8)*sizeof(float)]);
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, h->ptr[p_table + (tbl_offset+0*8)*sizeof(float)]);
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    h->uni_vmovups(vmm_mask, vmm_aux3);// The mask should be xmm0 for sse4.1
    h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);
    h->uni_vmovups(vmm_mask, vmm_aux2);

    remark(11) << "    -> " << registers.second[0] << " = sigmoid (" << registers.first[0] << ")" << std::endl;
}

// FixMe: It should be Scalar instead!!
void CPUGenerator::emit(std::shared_ptr<opset1::Constant>& constant, RegInfo& registers, bool vec) const {
    if (constant->outputs().size() != 1) {
        throw ngraph_error("constant with more than 1 output is not supported");
    }

    auto out_shape = constant->output(0).get_tensor().get_shape();
    if (out_shape == Shape() || ngraph::shape_size(out_shape) == 1) {
        // auto x = as_type_ptr<Node>(constant);
        auto x = std::dynamic_pointer_cast<Node>(constant);
        remark(1) << " scalar constant -> " << constant->cast_vector<float>()[0] << " " << getTableOffset(x) << std::endl;

        Xbyak::Ymm vmm = Xbyak::Ymm(registers.second[0]);
        h->uni_vbroadcastss(vmm, h->ptr[p_table + getTableOffset(x)*sizeof(float)]);
    } else {
        // throw ngraph_error("non scalar constant support is not implemented");
    }

    // fix me don't allocate register for non scalar consts
    remark(1) << "    -> " << registers.second[0] << " = const (" << ")" << std::endl;
}

// Erf function approximation without tanh
void CPUGenerator::emit(std::shared_ptr<opset1::Erf>& op, RegInfo& registers) const {
    // auto n = as_type_ptr<Node>(op);
    auto n = std::dynamic_pointer_cast<Node>(op);
    auto offset = getTableOffset(n);
    remark(1) << "  -> erf " << offset << std::endl;

    auto regIDx = registers.second[0];
    decltype(regIDx) latest = 15;

    Xbyak::Ymm polynom = Xbyak::Ymm(regIDx);
    h->uni_vbroadcastss(polynom, h->ptr[p_table + (offset + 11)*sizeof(float)]);

    Xbyak::Ymm x = Xbyak::Ymm(registers.first[0]);
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

    // to replace with bypass
    // h->uni_vmovups(polynom, x);

    remark(1) << "Free registers " << latest - regIDx + 1 << std::endl;
    remark(1) << "    -> " << registers.second[0] << " = erf (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<op::FakeBroadcast>& broadcast, RegInfo& registers) const {
    // Fix me: make it bypass nop
    remark(1) << "broadcast " << broadcast->get_input_shape(0) << " -> " << broadcast->get_output_shape(0)
    << (*broadcast->get_input_shape(0).rbegin() != *broadcast->get_output_shape(0).rbegin() ? "BROADCAST X" : "BROADCAST YZW") << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    if (*broadcast->get_input_shape(0).rbegin() != *broadcast->get_output_shape(0).rbegin()) {
        h->uni_vbroadcastss(vmm_dst, Xbyak::Xmm(registers.first[0]));
    } else {
        h->uni_vmovups(vmm_dst, vmm_src0);
    }

    remark(1) << "    -> " << registers.second[0] << " = broadcast (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::emit(std::shared_ptr<opset1::Broadcast>& broadcast, RegInfo& registers) const {
    // Fix me: make it bypass nop
    remark(1) << "broadcast " << broadcast->get_input_shape(0) << " -> " << broadcast->get_output_shape(0) << std::endl;

    Xbyak::Ymm vmm_src0 = Xbyak::Ymm(registers.first[0]);
    Xbyak::Ymm vmm_dst  = Xbyak::Ymm(registers.second[0]);

    h->uni_vmovups(vmm_dst, vmm_src0);

    remark(1) << "    -> " << registers.second[0] << " = broadcast (" << registers.first[0] << ")" << std::endl;
}

void CPUGenerator::generate_propotype(std::shared_ptr<ngraph::Function>& f) const {
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
                remark(2) << "setting constant to " << regs.size()+1+nConstants << " "
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
}

static auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> std::pair<std::vector<size_t>, std::vector<size_t>> {
    auto rt = n->get_rt_info();

    std::vector<size_t> rout;
    if (auto rinfo = rt["reginfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
        for (auto reg : reginfo) {
            rout.push_back(reg);
        }
    }

    std::vector<size_t> rin;
    for (auto input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        if (auto rinfo = rt["reginfo"]) {
            auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
            for (auto reg : reginfo) {
                rin.push_back(reg);
            }
        }
    }
    return std::make_pair(rin, rout);
}

code CPUGenerator::generate(std::shared_ptr<ngraph::Function>& f) const {
    if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::avx2)) {
        remark(0) << "generating for AVX2 ISA" << std::endl;
    } else {
        throw ngraph::ngraph_error("unsupported architecture for code genration");
    }

    // part 2: generation flow
    ngraph::pass::AssignRegistersPass().run_on_function(f);

    // sets up offsets to constant and temporals
    ngraph::pass::SetupStackTemporalsOffsetPass().run_on_function(f);

    // Note: should it be also a pass?
    h->preamble();
#if 1
    generate_propotype(f);

    Xbyak::Label l_table;
    h->mov(p_table, l_table);

    // configure tile variants
    size_t vlen = mkldnn::impl::cpu::cpu_isa_traits<mkldnn::impl::cpu::avx2>::vlen;
    std::array<size_t, 2> nloads   = {vlen / sizeof(float), 1};
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
            ngraph::pass::GenerateCodePass(this, loopId == 0).run_on_function(f);

            // loop_advance()
            h->sub(amount, nloads[loopId]);
            h->cmp(amount, nloads[loopId]);
            h->jge(for_body[loopId], jit_snippet::T_NEAR);
        }
    }

    h->L(for_body[nloads.size()]);
#endif
    h->postamble();
#if 1
    h->align(64);
    h->L(l_table);

    ngraph::pass::GenerateConstntTables(this).run_on_function(f);
#endif
    return h->getCode();
}

void CPUGenerator::emit_table(std::shared_ptr<opset1::Constant>& op) const {
    auto out_shape = op->output(0).get_tensor().get_shape();
    if (out_shape == Shape() || ngraph::shape_size(out_shape) == 1) {
        remark(1) << "pugging constant " << op->cast_vector<float>()[0] << " to the stack" << std::endl;
        h->dd(mkldnn::impl::cpu::float2int(op->cast_vector<float>()[0]));
    }
}

void CPUGenerator::emit_table(std::shared_ptr<opset1::Erf>& op) const {
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

void CPUGenerator::emit_table(std::shared_ptr<opset1::Clamp>& op) const {
    remark(1) << "pugging Clamp min " << op->get_min() << " to the stack" << std::endl;
    h->dd(mkldnn::impl::cpu::float2int(static_cast<float>(op->get_min())));
    remark(1) << "pugging Clamp max " << op->get_max() << " to the stack" << std::endl;
    h->dd(mkldnn::impl::cpu::float2int(static_cast<float>(op->get_max())));
}

void CPUGenerator::emit_table(std::shared_ptr<opset1::Sigmoid>& op) const {
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

    // for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(mkldnn::impl::cpu::float2int(alpha_));
    // for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}