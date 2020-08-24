// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API InsertExplicitLoadsPass;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::InsertExplicitLoadsPass: public ngraph::pass::GraphRewrite {
public:
    InsertExplicitLoadsPass();
};