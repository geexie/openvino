// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API InsertExplicitLoadsPass: public GraphRewrite {
public:
    InsertExplicitLoadsPass();
};

}  // namespace pass
}  // namespace ngraph

