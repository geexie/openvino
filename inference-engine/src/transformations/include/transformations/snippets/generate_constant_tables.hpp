// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/pass.hpp"
#include "generator.hpp"

#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GenerateConstntTables : public ngraph::pass::FunctionPass {
public:
    GenerateConstntTables(const Generator* generator)
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