// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/vector_to_scalar_pass.hpp"
#include "transformations/snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ReplaceLoadsWithScalarLoads::ReplaceLoadsWithScalarLoads() {
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::op::Load>(), "ReplaceLoadsWithScalarLoads"),
            [this](ngraph::pattern::Matcher &m) {
            auto root = m.get_match_root();
            auto load = std::make_shared<ngraph::op::ScalarLoad> (root->input_value(0));
            load->set_friendly_name(root->get_friendly_name());
            ngraph::copy_runtime_info(root, load);
            ngraph::replace_node(root, load);
            return true;
        },
        PassProperty::CHANGE_DYNAMIC_STATE);

    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::op::Store>(), "ReplaceStoresWithScalarStores"),
            [this](ngraph::pattern::Matcher &m) {
            auto root = m.get_match_root();
            auto store = std::make_shared<ngraph::op::ScalarStore> (root->input_value(0));
            store->set_friendly_name(root->get_friendly_name());
            ngraph::copy_runtime_info(root, store);
            ngraph::replace_node(root, store);
            return true;
        },
        PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
