// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API CanonicalizationPass : public FunctionPass {
public:
    CanonicalizationPass() : FunctionPass() {
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

} // namespace pass
} // namespace ngraph
