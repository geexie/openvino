// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "transformations/snippets/generator.hpp"

namespace MKLDNNPlugin {

class CPUTargetMachine : public ngraph::snippets::TargetMachine {
public:
    CPUTargetMachine(dnnl::impl::cpu::x64::jit_generator* h_, dnnl::impl::cpu::x64::cpu_isa_t host_isa_)
    : TargetMachine(), h(h_), host_isa(host_isa_) {
    }

    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
};

class CPUGenerator : public ngraph::snippets::Generator {
public:
    CPUGenerator();
    ~CPUGenerator() = default;

    ngraph::snippets::code generate(std::shared_ptr<ngraph::Function>& f) const override;

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
};

} // namespace MKLDNNPlugin {