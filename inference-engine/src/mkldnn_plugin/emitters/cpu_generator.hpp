// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph_ops/snippets_isa.hpp"
#include "transformations/snippets/generator.hpp"

#include "jit_generator.hpp"

namespace ngraph {
namespace snippet {

class jit_snippet : public mkldnn::impl::cpu::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator() {
        auto ker_ = this->getCode();
    }
};

class CPUTargetMachine : public TargetMachine {
public:
    CPUTargetMachine(mkldnn::impl::cpu::jit_generator* h_, mkldnn::impl::cpu::cpu_isa_t host_isa_) : TargetMachine(), h(h_), host_isa(host_isa_) {
    }

    mkldnn::impl::cpu::jit_generator* h;
    mkldnn::impl::cpu::cpu_isa_t host_isa;
};

class JitEmitter : public Emitter {
public:
    // jit_generator shouldnot be here, but have no idea for now, something like TargetMachine
    JitEmitter(mkldnn::impl::cpu::jit_generator* h_, mkldnn::impl::cpu::cpu_isa_t host_isa_, const std::shared_ptr<ngraph::Node>& n)
        : Emitter(n), h(h_), host_isa(host_isa_) {
    }

    virtual void emit(const std::vector<size_t>& in,
                      const std::vector<size_t>& out,
                      const std::vector<size_t>& pool = {},
                      const std::vector<size_t>& gpr  = {}) const = 0;

    virtual void emit_table() {
    }

protected:
    mkldnn::impl::cpu::jit_generator* h;
    mkldnn::impl::cpu::cpu_isa_t host_isa;

    int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    Xbyak::Reg64 param  { mkldnn::impl::cpu::abi_param1 }; // RDI
    Xbyak::Reg64 p_table { Xbyak::util::rax }; // get from somewhere
};

class TRANSFORMATIONS_API CPUGenerator : public Generator {
public:
    CPUGenerator();
    ~CPUGenerator() = default;

    code generate(std::shared_ptr<Function>& f) const override;

protected:
    void generate_propotype(std::shared_ptr<ngraph::Function>& f) const override;
    void generate_tile(std::shared_ptr<ngraph::Function>& f) const override;
    void generate_return(std::shared_ptr<ngraph::Function>& f) const override;

private:
    std::unique_ptr<mkldnn::impl::cpu::jit_generator> h;
    mkldnn::impl::cpu::cpu_isa_t isa;

    int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    Xbyak::Reg64 param  { mkldnn::impl::cpu::abi_param1 }; // RDI
    Xbyak::Reg64 p_table { Xbyak::util::rax }; // get from somewhere

    mutable Xbyak::Label l_table;
};

} // namespace snippet
using snippet::CPUGenerator;
} // namespace ngraph