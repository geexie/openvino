// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/insert_explicit_loads_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "ngraph_ops/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <iostream>

/// pass parameter outputs through Load operation
/// FIXME: in would worth to check if Load is already put after `Parameter`, so it would no harm to run this pass multiple times
ngraph::pass::InsertExplicitLoadsPass::InsertExplicitLoadsPass() {
    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Parameter>(), "InsertExplicitLoads"),
            [this](ngraph::pattern::Matcher &m) {
            auto root = m.get_match_root();
            auto load = std::make_shared<ngraph::op::Load> (root);
            ngraph::copy_runtime_info(root, load);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != load) {
                        consumer.replace_source_output(load);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        },
        PassProperty::CHANGE_DYNAMIC_STATE);
}