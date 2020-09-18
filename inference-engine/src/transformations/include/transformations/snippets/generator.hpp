// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph_ops/snippets_isa.hpp"

// Raplace with jitter table

namespace ngraph {
namespace snippet {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

class TargetMachine {
};

// Generator interface
class TRANSFORMATIONS_API Generator {
public:
    Generator() = default;
    virtual ~Generator() = default;

    virtual code generate(std::shared_ptr<Function>& f) const = 0;

    // FIXME: make prototype & module peramble/postamble to be a part of opset as well as more auxary things like function signature generation
    // How is to make it before parameters? should it be part of the module? like module is a functions + signature + return or what...
    virtual void generate_propotype(std::shared_ptr<ngraph::Function>& f) const = 0;

    // FIXME: move main loop generation somewhere here
    virtual void generate_tile(std::shared_ptr<ngraph::Function>& f) const = 0;

    // FIXME: used to generate mable but can be anything make a part of opset as well
    // How is to make it before parameters? should it be part of the module?
    virtual void generate_return(std::shared_ptr<ngraph::Function>& f) const = 0;

    // FIXME: exclude extension form opset to use Standart broadcast
    // FIXME: vec shouldn't be here in such an explicit way, but be need to generate tables once for a pass so cannot duplicate body
    // generate it like normal compilers usually do in future
    virtual void emit(const std::shared_ptr<opset1::Broadcast>& broadcast, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<op::FakeBroadcast>& op, RegInfo& registers) const = 0;

    // FIXME: to jitters
    virtual void emit(const std::shared_ptr<op::Scalar>& op, RegInfo& registers) const = 0;

    virtual void emit(const std::shared_ptr<op::Load>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<op::ScalarLoad>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<op::BroadcastLoad>& op, RegInfo& registers) const = 0;

    virtual void emit(const std::shared_ptr<op::Store>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<op::ScalarStore>& op, RegInfo& registers) const = 0;

    virtual void emit(const std::shared_ptr<opset1::Parameter>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Result>& op, RegInfo& registers) const = 0;

    virtual void emit(const std::shared_ptr<opset1::Add>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Subtract>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Multiply>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Negative>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Erf>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Divide>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Clamp>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Relu>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Sigmoid>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::SquaredDifference>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Power>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::PRelu>& op, RegInfo& registers) const = 0;
    virtual void emit(const std::shared_ptr<opset1::Tanh>& op, RegInfo& registers) const = 0;

    // FIXME: it's something which should be done automatically with constant map, since it's architecture specific
    virtual void emit_table(const std::shared_ptr<op::Scalar>& constant) const = 0;
    virtual void emit_table(const std::shared_ptr<opset1::Erf>& op) const = 0;
    virtual void emit_table(const std::shared_ptr<opset1::Clamp>& op) const = 0;
    virtual void emit_table(const std::shared_ptr<opset1::Sigmoid>& op) const = 0;
};

auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippet::RegInfo;


} // namespace snippet
using snippet::Generator;
using snippet::code;
} // namespace ngraph