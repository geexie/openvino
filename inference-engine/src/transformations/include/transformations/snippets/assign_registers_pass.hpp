// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API AssignRegistersPass : public FunctionPass {
public:
    AssignRegistersPass() : FunctionPass() {
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

} // namespace pass
} // namespace ngraph
