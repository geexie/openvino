// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "transformations/snippets/canonicalize_pass.hpp"
#include "transformations/snippets/remarks.hpp"

// can be refactored with a matcher to broadcast
bool ngraph::pass::CanonicalizationPass::run_on_function(std::shared_ptr<Function> f) {
    remark(11) << "CanonicalizationPass" << std::endl;
    return false;
}