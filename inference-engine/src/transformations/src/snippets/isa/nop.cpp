// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/isa/nop.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Nop::type_info;

op::Nop::Nop(const Output<Node>& x) : Op({x}) {
}
