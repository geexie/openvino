// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"
#include <ngraph/opsets/opset1.hpp>

#include "load.hpp"
#include "broadcastload.hpp"
#include "fakebroadcast.hpp"
#include "nop.hpp"
#include "scalar.hpp"

namespace ngraph {
namespace snippets {
namespace isa {
#define NGRAPH_OP(a, b) using b::a;
#include "snippets_isa_tbl.hpp"
#undef NGRAPH_OP
} // namespace isa
} // namespace snippets
} // namespace ngraph
