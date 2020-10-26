// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/nop.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Nop::type_info;

op::Nop::Nop(const OutputVector& arguments, const OutputVector& results) : Op([arguments, results]() -> OutputVector {
    OutputVector x;
    x.insert(x.end(), arguments.begin(), arguments.end());
    x.insert(x.end(), results.begin(), results.end());
    return x;
    }()) {
}
