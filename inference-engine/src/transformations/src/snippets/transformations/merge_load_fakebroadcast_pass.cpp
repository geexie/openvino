// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transformations/merge_load_fakebroadcast_pass.hpp"

#include "ngraph_ops/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "snippets/remarks.hpp"

#include <iostream>

ngraph::pass::MergeLoadFakeBroadcastToBroadcastLoadPass::MergeLoadFakeBroadcastToBroadcastLoadPass() {
     ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto root = m.get_match_root();

        auto input = root->input(0).get_source_output().get_node_shared_ptr();
        auto param = input->input(0).get_source_output().get_node_shared_ptr();

        remark(2) << "merge load+fakebroadcast for " << root->get_friendly_name() << " (" << root->get_type_name() << ") + "
            << param->get_friendly_name() << " (" << param->get_type_name() << ")" << std::endl;

        if (root->inputs().size() != 1 || input->inputs().size() != 1) {
            throw ngraph_error("cannot rewrite Broadcast load with more than one input");
        }

        // if (auto fb = as_type_ptr<op::FakeBroadcast>(root))

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
        broadcastload->set_friendly_name(root->get_friendly_name());
        ngraph::copy_runtime_info(root, broadcastload);
        ngraph::replace_node(root, broadcastload);
        return true;
    };

    {
        auto pn = std::make_shared<ngraph::opset1::Parameter>(element::f32, Shape{});
        auto ln = std::make_shared<ngraph::op::Load>(pn);
        auto fbn = std::make_shared<ngraph::op::FakeBroadcast>(ln, Shape{1});
        auto m = std::make_shared<ngraph::pattern::Matcher>(fbn, "MergeLoadFakeBroadcastToBroadcastLoadPass::Param");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }

    {
        auto pn = std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{});
        auto ln = std::make_shared<ngraph::op::Load>(pn);
        auto fbn = std::make_shared<ngraph::op::FakeBroadcast>(ln, Shape{1});
        auto m = std::make_shared<ngraph::pattern::Matcher>(fbn, "MergeLoadFakeBroadcastToBroadcastLoadPass::Constant");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
}