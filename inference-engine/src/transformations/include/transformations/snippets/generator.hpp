// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include "snippets_isa.hpp"

namespace ngraph {
namespace snippets {

using code = const uint8_t *;
using RegInfo = std::pair<std::vector<size_t>, std::vector<size_t>>;

TRANSFORMATIONS_API auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo;

class Emitter {
public:
    Emitter(const std::shared_ptr<ngraph::Node>& n) {
    }

    virtual void emit_code(const std::vector<size_t>& in,
                           const std::vector<size_t>& out,
                           const std::vector<size_t>& pool = {},
                           const std::vector<size_t>& gpr  = {}) const = 0;

    virtual void emit_data() const {
    }
};

class TargetMachine {
public:
    virtual auto getJitters() -> std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>>{
        return {};
    }
};

class Schedule {
public:
    Schedule() : work_size({}), is_flat(false), ptr(nullptr) {}
    Schedule(const Shape& ws, bool f, code p) : work_size(ws), is_flat(f), ptr(p) {}

    Shape work_size {};
    bool is_flat {false};
    code ptr {nullptr};
};

class TRANSFORMATIONS_API Generator {
public:
    Generator() = default;
    virtual ~Generator() = default;

    // FIXME: generates code for a specific tile decided by whom?
    virtual code generate(std::shared_ptr<Function>& f) const = 0;

protected:
    // hierarchical IR
    // -snippet - vector of tiles
    //  - parameters gethering
    //  - tile - single body ~ subgraph
    //   - body
    //     - op
    //     - op
    //     - op
    // - tile
    //  - body
    //    - op
    //    - op
    //    - op
    // - data constant table
    mutable std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>> jitters;
};

} // namespace snippets
} // namespace ngraph