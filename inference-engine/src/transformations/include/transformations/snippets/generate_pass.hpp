// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

#include "generator.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GenerateCodePass : public FunctionPass {
public:
    GenerateCodePass(const Generator* generator)
        : FunctionPass()
        , m_generator(generator) {
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    const Generator* m_generator;
};
} // namespace pass
} // namespace ngraph
