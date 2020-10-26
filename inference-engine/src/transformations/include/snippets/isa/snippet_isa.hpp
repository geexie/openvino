// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

#include "snippets/isa/load.hpp"
#include "snippets/isa/broadcastload.hpp"
#include "snippets/isa/fakebroadcast.hpp"
#include "snippets/isa/nop.hpp"
#include "snippets/isa/scalar.hpp"

namespace ngraph {
namespace snippet {
namespace isa {
#define NGRAPH_OP(a, b) using b::a;
#include "snippet_isa_tbl.hpp"
#undef NGRAPH_OP
} // namespace isa
} // namespace snippet
} // namespace ngraph
