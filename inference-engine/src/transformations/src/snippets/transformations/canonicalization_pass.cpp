// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "snippets/transformations/canonicalization_pass.hpp"
#include "ngraph_ops/snippets_isa.hpp"
#include "snippets/remarks.hpp"

// can be refactored with a matcher to broadcast
bool ngraph::pass::CanonicalizationPass::run_on_function(std::shared_ptr<Function> f) {
    remark(11) << "CanonicalizationPass" << std::endl;
    return false;
}