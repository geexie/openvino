// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if 0
#include "transformations/snippets/detect_broadcast_pass.hpp"
#include "ngraph_ops/load.hpp"
#include "ngraph_ops/broadcastload.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <iostream>

ngraph::pass::DetectBroadcastPass::DetectBroadcastPass() {
     ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto root = m.get_match_root();
        std::cout << "Ramark: DetectBroadcastPass rewriting for " << root->get_friendly_name() << std::endl;

        bool rewritten = false;
        for (auto output : root->outputs()) {
            std::cout << std::endl << std::endl;
            for (auto consumer : output.get_target_inputs()) {
                std::cout << "---> " << "get_target_inputs " << consumer.get_node()->get_friendly_name() << " index " << output.get_index() << std::endl;

                // is it possible?
                if (auto x = as_type_ptr<ngraph::op::snippet::Load>(consumer.get_node()->shared_from_this())) {
                    ngraph_error("trying to insert load second time for a parameter " + x->get_friendly_name());
                }

                for (auto input : consumer.get_node()->inputs()) {
                    auto producer = input.get_source_output();
                    std::cout << "  --->" << producer.get_node()->get_friendly_name() << std::endl;
                    if (producer.get_node()->shared_from_this() == root) {
                        std::cout << "  ---> we finally found ourself " << producer.get_node()->get_friendly_name() << std::endl;
                        // Detect broadcast here
                        // consumer == Add
                        // producer == Parameter
                        // Task is to decide if we need to insert Load or Broadcast
                        // if broadcast is technically possible
                        if (consumer.get_node()->supports_auto_broadcast()) {
                            // if broadcast is actually take place
                            if (consumer.get_node()->inputs().size() != 2) {
                                // Do we really have such a case (it might be select op)
                                ngraph_error("error auto broadcast config: more than 2 input operation detected");
                            }
                             std::cout << "  ---> supports broadcast" << std::endl;

                            auto this_shape = input.get_shape();
                            std::cout << this_shape << " " << consumer.get_node()->inputs().size() << std::endl;

                            Shape bct(this_shape.size(), 0);


                            // considers only preserved dimensions
                            for (auto other_input : consumer.get_node()->inputs()) {
                                auto other = other_input.get_source_output().get_node()->shared_from_this();
                                std::cout << "  ---> Other: " << other_input.get_shape()
                                    << " " << other->get_friendly_name() << std::endl;

                                if (root != other) {
                                    std::cout << "  ---> considering " << root->get_friendly_name() << " and " <<
                                    other->get_friendly_name() << std::endl;
                                    auto other_shape = other_input.get_shape();
                                    std::cout << "    ----> Other: " << other_shape << std::endl;
                                    if (this_shape.size() == other_shape.size()) {
                                        for (int i = 0; i < this_shape.size(); i++) {
                                            if (this_shape[i] != other_shape[i] && this_shape[i] == 1) {
                                                bct[i] = 1;
                                            }
                                        }
                                    }
                                }
                            }

                            std::cout << "broadcast config" << bct << std::endl;

                            if (std::any_of(bct.begin(), bct.end(), [](decltype(bct[0]) i){return i == 1;})) {
                                std::cout << "insert broadcast" << std::endl;
                                auto load = std::make_shared<ngraph::op::snippet::BroadcastLoad>(root->input(0).get_source_output());
                                load->set_broadcast_info(bct);
                                ngraph::copy_runtime_info(root, load);
                                input.replace_source_output(load);
                                rewritten = true;
                            }
                        }
                    }
                }
            }
        }

        return rewritten;
    };

    auto pn = std::make_shared<ngraph::opset1::Parameter>(element::f32, Shape{});
    auto ld = std::make_shared<op::Load>(pn);
    auto m = std::make_shared<ngraph::pattern::Matcher>(ld, "DetectBroadcast");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);

    auto pn1 = std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{});
    auto ld1 = std::make_shared<op::Load>(pn1);
    auto m1 = std::make_shared<ngraph::pattern::Matcher>(ld1, "DetectBroadcast");
    this->add_matcher(m1, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
#endif
