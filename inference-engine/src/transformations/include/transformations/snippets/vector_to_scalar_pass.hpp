// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

// FIXME: this file should be rewritten once codegen is moved to jitters and global string-based constant table

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReplaceLoadsWithScalarLoads: public ngraph::pass::GraphRewrite {
public:
    ReplaceLoadsWithScalarLoads();
};

} // namespace pass
} // namespace ngraph