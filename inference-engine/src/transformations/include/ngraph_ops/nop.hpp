// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API Nop : public Op {
public:
    static constexpr NodeTypeInfo type_info{"isa-nop", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Nop(const OutputVector& arguments, const OutputVector& results);
    Nop() = default;

private:
};

// or better Tile operation for polyhedral semantics
class TRANSFORMATIONS_API Tile : public Op {
public:
    static constexpr NodeTypeInfo type_info{"isa-tile", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Tile(const OutputVector& arguments);
    Tile() = default;

private:
};

// finally it's
class TRANSFORMATIONS_API Snippet : public Op {
public:
    static constexpr NodeTypeInfo type_info{"isa-snippet", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Snippet(const OutputVector& arguments);
    Snippet() = default;

private:
};

} // namespace op
} // namespace ngraph