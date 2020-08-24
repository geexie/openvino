// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/pull_up_fakebroadcast_pass.hpp"

#include "ngraph_ops/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

// can be refactored with a matcher to broadcast
bool ngraph::pass::PullUpBroadcastsPass::run_on_function(std::shared_ptr<Function> f) {
    auto ops = f->get_ordered_ops();
    bool rewritten = false;

    for (auto it = ops.rbegin(); it != ops.rend(); it++) {
        if (auto broadcast = as_type_ptr<op::FakeBroadcast>(*it)) {
            auto children = broadcast->output(0).get_target_inputs();
            auto parent = broadcast->input(0).get_source_output();
            auto in = broadcast->get_input_tensor(0).get_shape();
            auto out = broadcast->get_output_tensor(0).get_shape();

            // if (in.size() != out.size()) {
            //     throw ngraph_error("broadcast over different shapes is not supported");
            // }

            Shape bct(in.size(), 0);
            for (int i = 0; i < in.size(); i++) {
                bct[i] = in[i] != out[i];
            }

            std::unordered_set<Node*> nodes;
            for (auto input : broadcast->inputs()) {
                nodes.insert(input.get_source_output().get_node());
            }

            while (!nodes.empty()) {
                auto curr = *nodes.begin();
                nodes.erase(curr);

                if (auto param = dynamic_cast<opset1::Parameter*>(curr)) {
                    // ToDo: actually handle this case by moving broadcast 1 up
                    // auto load = std::make_shared<ngraph::op::snippet::BroadcastLoad>(curr->input(0).get_source_output());
                    // load->set_broadcast_info(bct);
                    // ngraph::copy_runtime_info(curr->shared_from_this(), load);
                    // ngraph::replace_node(curr->shared_from_this(), load);
                }

                if (auto param = dynamic_cast<op::Load*>(curr)) {
                    // FixMe: move not insert one more
                    // auto load = std::make_shared<ngraph::op::snippet::BroadcastLoad>(curr->input(0).get_source_output());
                    // load->set_broadcast_info(bct);
                    // load->set_friendly_name(curr->get_friendly_name());
                    // ngraph::copy_runtime_info(curr->shared_from_this(), load);
                    // ngraph::replace_node(curr->shared_from_this(), load);

                    // // for (auto& child : children) {
                    // //     parent.get_node()->set_argument(child.get_index(), parent);
                    // // }

                    // rewritten = true;
                }

                if (auto param = dynamic_cast<op::FakeBroadcast*>(curr)) {
                    // throw ngraph_error("multiple broadcasts are not supported");
                    // just give up
                    break;
                }

                for (auto input : curr->inputs()) {
                    nodes.insert(input.get_source_output().get_node());
                }
            }
        }
    }

    return rewritten;
}
