// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/insert_explicit_loads_pass.hpp"
#include "ngraph_ops/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "transformations/snippets/remarks.hpp"

#include <iostream>

ngraph::pass::InsertExplicitLoadsPass::InsertExplicitLoadsPass() {
     ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto root = m.get_match_root();
        remark(2) << "explicit loads matched" << root->get_friendly_name() << " " << root->get_type_name() << std::endl;

        if (as_type_ptr<ngraph::opset1::Parameter>(root) || as_type_ptr<ngraph::opset1::Constant>(root)) {
            // create load for a parameter
            auto load = std::make_shared<ngraph::op::Load> (root);
            ngraph::copy_runtime_info(root, load);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    remark(2) << "---> " << consumer.get_node()->get_friendly_name() << " " << consumer.get_node()->get_type_name() << std::endl;

                    if (consumer.get_node()->shared_from_this() == load) {
                        continue;
                    }

                    consumer.replace_source_output(load);
                    rewritten = true;
                }
            }

            return rewritten;
        }

        return false;
    };

    auto pn = std::make_shared<ngraph::opset1::Parameter>(element::f32, Shape{});
    auto m = std::make_shared<ngraph::pattern::Matcher>(pn, "InsertExplicitLoads::Parameter");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);

    auto pn1 = std::make_shared<ngraph::pattern::op::Label>(element::f32, Shape{},
        ngraph::pattern::op::NodePredicate([](std::shared_ptr<Node> n) -> bool {
            auto constant = as_type_ptr<ngraph::opset1::Constant>(n);
            return constant && (constant->output(0).get_tensor().get_shape() != Shape()
                && ngraph::shape_size(constant->output(0).get_tensor().get_shape()) > 1);
        }));
    auto m1 = std::make_shared<ngraph::pattern::Matcher>(pn1, "InsertExplicitLoads::Constant");
    this->add_matcher(m1, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}