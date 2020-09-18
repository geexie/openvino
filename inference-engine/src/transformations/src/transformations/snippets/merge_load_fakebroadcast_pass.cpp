// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/merge_load_fakebroadcast_pass.hpp"

#include "ngraph_ops/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/snippets/remarks.hpp"

#include <iostream>

/// FIXME: In general it's not optimal to have one parameter comming with and without broadcast in the same subgraph
/// unless we implement tile unrolling by vector width. In this case we would
/// Assumption that every parameter loaded from memory only one should be correct so check if the is only one used of this load unless keep as is
ngraph::pass::MergeLoadFakeBroadcastToBroadcastLoadPass::MergeLoadFakeBroadcastToBroadcastLoadPass() {
    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto load_pattern = std::make_shared<ngraph::op::Load>(param_pattern);
    auto fbn = std::make_shared<ngraph::op::FakeBroadcast>(load_pattern, Shape{1});

    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(fbn, "MergeLoadFakeBroadcastToBroadcastLoadPass"),
        [load_pattern, param_pattern](ngraph::pattern::Matcher &m) {
            auto root = m.get_match_root();
            const auto &pm = m.get_pattern_value_map();
            const auto input = pm.at(load_pattern).get_node_shared_ptr();
            const auto param = pm.at(param_pattern).get_node_shared_ptr();

            if (root->inputs().size() != 1 || input->inputs().size() != 1) {
                throw ngraph_error("cannot rewrite Broadcast load with more than one input");
            }

            auto inshape = root->input(0).get_shape();
            auto outshape = root->output(0).get_shape();
            auto broadcastload = std::make_shared<op::BroadcastLoad>(param, outshape);
            Shape bct(inshape.size(), 0);
            for (int k = 0; k < inshape.size(); k++) {
                if (inshape[k] != outshape[k] && inshape[k] == 1) {
                    bct[k] = 1;
                }
            }
            broadcastload->set_broadcast_info(bct);
            if (broadcastload->is_broadcast(outshape.size()-1)) {
                // broadcastload->set_friendly_name(root->get_friendly_name());
                ngraph::copy_runtime_info(root, broadcastload);
                ngraph::replace_node(root, broadcastload);
                return true;
            } else {
                return false;
            }
        },
        PassProperty::CHANGE_DYNAMIC_STATE);
}