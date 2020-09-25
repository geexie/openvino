// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include "ngraph_ops/snippets_isa.hpp"

namespace ngraph {
namespace snippet {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;
TRANSFORMATIONS_API auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippet::RegInfo;

class TargetMachine {
};

class Emitter {
public:
    Emitter(const std::shared_ptr<ngraph::Node>& n) {
    }

    virtual void emit(const std::vector<size_t>& in,
                      const std::vector<size_t>& out,
                      const std::vector<size_t>& pool = {},
                      const std::vector<size_t>& gpr  = {}) const = 0;

    // FIXME: remove in future
    virtual void emit_table() {
    }
};

// Generator interface
class TRANSFORMATIONS_API Generator {
public:
    Generator() = default;
    virtual ~Generator() = default;

    // FIXME: generates code for a specific tile decided by whom?
    virtual code generate(std::shared_ptr<Function>& f) const = 0;

protected:
    // those might be too platform specific
    // FIXME: make prototype & module peramble/postamble to be a part of opset as well as more auxary things like function signature generation
    // How is to make it before parameters? should it be part of the module? like module is a functions + signature + return or what...
    virtual void generate_propotype(std::shared_ptr<ngraph::Function>& f) const = 0;

    // FIXME: move main loop generation somewhere here
    virtual void generate_tile(std::shared_ptr<ngraph::Function>& f) const = 0;

    // FIXME: used to generate mable but can be anything make a part of opset as well
    // How is to make it before parameters? should it be part of the module?
    virtual void generate_return(std::shared_ptr<ngraph::Function>& f) const = 0;

    mutable std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>> jitters;
};

} // namespace snippet
using snippet::Generator;
using snippet::code;
} // namespace ngraph