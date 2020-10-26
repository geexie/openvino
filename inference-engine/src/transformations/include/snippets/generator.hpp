// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "snippets/isa/load.hpp"
#include "snippets/isa/broadcastload.hpp"
#include "snippets/isa/fakebroadcast.hpp"
#include "snippets/register_info.hpp"

#include "jit_generator.hpp"

namespace ngraph {
namespace snippet {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

// Codegen opset:
//
//  1. op::Load
//  2. op::BroadcastLoad
//  3. op::FakeBroadcast
//
//  4. opset1::Parameter
//  5. opset1::Result
//  6. opset1::Constant
//   indeed only scalar constant is supported not scalar goes over load. it was an attempt to save registers
//   it might be better to introduce scalar Op which should encapsulate such things
//
//  7. opset1::Add
//  8. opset1::Subtract
//  9. opset1::Multiply
// 10. opset1::Negative
// 11. opset1::Erf
// 12. opset1::Divide

// Generator interface
class TRANSFORMATIONS_API Generator {
public:
    Generator() = default;
    virtual ~Generator() = default;

    virtual code generate(std::shared_ptr<Function>& f) const = 0;

    // FixMe: make prototype & module peramble/postamble to be a part of opset as well as more auxary things like function signature generation
    virtual void generate_propotype(std::shared_ptr<ngraph::Function>& f) const = 0;
    virtual void emit_module_enter() = 0;
    virtual void emit_module_exit() = 0;

    virtual void emit(std::shared_ptr<op::Load>& op, RegInfo& registers, bool vec) const = 0;
    virtual void emit(std::shared_ptr<op::BroadcastLoad>& op, RegInfo& registers, bool vec) const = 0;
    virtual void emit(std::shared_ptr<op::FakeBroadcast>& op, RegInfo& registers) const = 0;

    // FixMe: vec shouldn't be here in such an explicit way, but be need to generate tables once for a pass so cannot duplicate body
    // generate it like normal compilers usually do in future
    virtual void emit(std::shared_ptr<opset1::Parameter>& op, RegInfo& registers, bool vec) const = 0;
    virtual void emit(std::shared_ptr<opset1::Result>& op, RegInfo& registers, bool vec) const = 0;
    virtual void emit(std::shared_ptr<opset1::Constant>& op, RegInfo& registers, bool vec) const = 0;

    virtual void emit(std::shared_ptr<opset1::Add>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Subtract>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Multiply>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Negative>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Erf>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Divide>& op, RegInfo& registers) const = 0;

    virtual void emit(std::shared_ptr<opset1::Clamp>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Relu>& op, RegInfo& registers) const = 0;

    virtual void emit(std::shared_ptr<opset1::Sigmoid>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::SquaredDifference>& op, RegInfo& registers) const = 0;
    virtual void emit(std::shared_ptr<opset1::Power>& op, RegInfo& registers) const = 0;

    // FixMe: exclude from opset
    virtual void emit(std::shared_ptr<opset1::Broadcast>& broadcast, RegInfo& registers) const = 0;

    virtual void emit_table(std::shared_ptr<opset1::Constant>& constant) const = 0;
    virtual void emit_table(std::shared_ptr<opset1::Erf>& op) const = 0;
    virtual void emit_table(std::shared_ptr<opset1::Clamp>& op) const = 0;
    virtual void emit_table(std::shared_ptr<opset1::Sigmoid>& op) const = 0;
};

class jit_snippet : public mkldnn::impl::cpu::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator() {
        auto ker_ = this->getCode();
    }
};

// alternative interface
class Emitter {
public:
    // jit_generator shouldnot be here, but have no idea for now
    Emitter(mkldnn::impl::cpu::jit_generator* h_, const std::shared_ptr<ngraph::Node>& n) : h(h_) {
    }

    virtual void emit(std::vector<size_t> in, std::vector<size_t> out, std::vector<size_t> pool) const = 0;
    virtual void emit_table() {
    }

protected:
    mkldnn::impl::cpu::jit_generator* h;
};

class TRANSFORMATIONS_API CPUGenerator : public Generator {
public:
    CPUGenerator();
    ~CPUGenerator() = default;

    code generate(std::shared_ptr<Function>& f) const override;
    void generate_propotype(std::shared_ptr<ngraph::Function>& f) const override;
    void emit_module_enter() override;
    void emit_module_exit() override;

    void emit(std::shared_ptr<op::Load>& op, RegInfo& registers, bool vec) const override;
    void emit(std::shared_ptr<op::BroadcastLoad>& op, RegInfo& registers, bool vec) const override;
    void emit(std::shared_ptr<op::FakeBroadcast>& op, RegInfo& registers) const override;

    void emit(std::shared_ptr<opset1::Parameter>& op, RegInfo& registers, bool vec) const override;
    void emit(std::shared_ptr<opset1::Result>& op, RegInfo& registers, bool vec) const override;
    void emit(std::shared_ptr<opset1::Constant>& op, RegInfo& registers, bool vec) const override;

    void emit(std::shared_ptr<opset1::Add>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Subtract>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Multiply>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Negative>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Erf>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Divide>& op, RegInfo& registers) const override;

    void emit(std::shared_ptr<opset1::Clamp>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Relu>& op, RegInfo& registers) const override;

    void emit(std::shared_ptr<opset1::Sigmoid>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::SquaredDifference>& op, RegInfo& registers) const override;
    void emit(std::shared_ptr<opset1::Power>& op, RegInfo& registers) const override;

    void emit(std::shared_ptr<opset1::Broadcast>& broadcast, RegInfo& registers) const override;

    void emit_table(std::shared_ptr<opset1::Constant>& constant) const override;
    void emit_table(std::shared_ptr<opset1::Erf>& op) const override;
    void emit_table(std::shared_ptr<opset1::Clamp>& op) const override;
    void emit_table(std::shared_ptr<opset1::Sigmoid>& op) const override;

private:
    std::unique_ptr<mkldnn::impl::cpu::jit_generator> h;

    int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    Xbyak::Reg64 param  { mkldnn::impl::cpu::abi_param1 }; // RDI
    Xbyak::Reg64 p_table { Xbyak::util::rax }; // get from somewhere

    std::map<ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>> jitters;
};

} // namespace snippet
using snippet::Generator;
using snippet::CPUGenerator;
using snippet::code;
} // namespace ngraph