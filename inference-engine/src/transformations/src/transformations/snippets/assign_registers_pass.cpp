// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/rt_info/register_info.hpp"

#include <ngraph/opsets/opset1.hpp>

#include "ngraph_ops/load.hpp"
#include "ngraph_ops/scalar.hpp"
#include "ngraph_ops/broadcastload.hpp"

// FIXME: use linear time register allocation algorithm
// Assigning internal `vector` register indexes to Ops (virtual registers)
// After this pass changing order of variables or datafrow lead to invalidation of register assignment
bool ngraph::pass::AssignRegistersPass::run_on_function(std::shared_ptr<Function> f) {
    // example from article
    {
        auto p0 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p1 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p2 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p3 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p4 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p5 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p6 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p7 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());

        auto c0 = std::make_shared<op::Scalar>(ngraph::element::f32, Shape(), 3.14f);
        auto c1 = std::make_shared<op::Scalar>(ngraph::element::f32, Shape(), 6.6260701e-34f);

        auto y00 = std::make_shared<op::Load>(p0); y00->set_friendly_name("y00");
        auto y01 = std::make_shared<op::Load>(p1); y01->set_friendly_name("y01");
        auto y02 = std::make_shared<ngraph::opset1::Multiply>(y00, c0); y02->set_friendly_name("y02");
        auto y03 = std::make_shared<ngraph::opset1::Multiply>(y01, c1); y03->set_friendly_name("y03");
        auto y04 = std::make_shared<op::Load>(p2); y04->set_friendly_name("y04");
        auto y05 = std::make_shared<op::Load>(p3); y05->set_friendly_name("y05");
        auto y06 = std::make_shared<ngraph::opset1::Add>(y02, y03); y06->set_friendly_name("y06");
        auto y07 = std::make_shared<ngraph::opset1::Multiply>(y04, c0); y07->set_friendly_name("y07");
        auto y08 = std::make_shared<ngraph::opset1::Multiply>(y05, c1); y08->set_friendly_name("y08");
        auto y09 = std::make_shared<op::Load>(p4); y09->set_friendly_name("y09");
        auto y10 = std::make_shared<op::Load>(p5); y10->set_friendly_name("y10");
        auto y11 = std::make_shared<ngraph::opset1::Add>(y07, y08); y11->set_friendly_name("y11");
        auto y12 = std::make_shared<ngraph::opset1::Multiply>(y09, c0); y12->set_friendly_name("y12");
        auto y13 = std::make_shared<ngraph::opset1::Multiply>(y10, c1); y13->set_friendly_name("y13");
        auto y14 = std::make_shared<op::Load>(p6); y14->set_friendly_name("y14");
        auto y15 = std::make_shared<ngraph::opset1::Add>(y12, y13); y15->set_friendly_name("y15");
        auto y16 = std::make_shared<op::Load>(p7); y16->set_friendly_name("y16");
        auto y17 = std::make_shared<ngraph::opset1::Multiply>(y14, c0); y17->set_friendly_name("y17");
        auto y18 = std::make_shared<ngraph::opset1::Multiply>(y16, c1); y18->set_friendly_name("y18");
        auto y19 = std::make_shared<ngraph::opset1::Add>(y06, y11); y19->set_friendly_name("y19");
        auto y20 = std::make_shared<ngraph::opset1::Add>(y17, y18); y20->set_friendly_name("y20");
        auto y21 = std::make_shared<ngraph::opset1::Add>(y15, y19); y21->set_friendly_name("y21");
        auto y22 = std::make_shared<ngraph::opset1::Add>(y20, y21); y22->set_friendly_name("y22");

        auto f = std::make_shared<ngraph::Function>(ngraph::NodeVector{y22}, ngraph::ParameterVector{p0, p1, p2, p3, p4, p5, p6, p7});

        std::cout << std::endl << std::endl;

        NodeVector stmts;
        stmts.push_back(y00);
        stmts.push_back(y01);
        stmts.push_back(y02);
        stmts.push_back(y03);
        stmts.push_back(y04);
        stmts.push_back(y05);
        stmts.push_back(y06);
        stmts.push_back(y07);
        stmts.push_back(y08);
        stmts.push_back(y09);
        stmts.push_back(y10);
        stmts.push_back(y11);
        stmts.push_back(y12);
        stmts.push_back(y13);
        stmts.push_back(y14);
        stmts.push_back(y15);
        stmts.push_back(y16);
        stmts.push_back(y17);
        stmts.push_back(y18);
        stmts.push_back(y19);
        stmts.push_back(y20);
        stmts.push_back(y21);
        stmts.push_back(y22);

        for (auto n : f->get_ordered_ops()) {
            std::cout << n << std::endl;
        }

        std::cout << std::endl << std::endl;

        for (auto n : stmts) {
            std::cout << n << std::endl;
        }

        using Reg = size_t;

        size_t rdx = 0;
        std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
        for (auto op : stmts) {
            for (auto output : op->outputs()) {
                regs[output.get_tensor_ptr()] = rdx++;
            }
        }

        std::cout << std::endl << std::endl;

        for (auto r : regs) {
            std::cout << r.first << " " << r.first->get_name() << " " << r.first->get_shape() << " " << r.second << std::endl;
        }

        std::cout << std::endl << std::endl;
        std::vector<std::set<Reg>> used;
        std::vector<std::set<Reg>> def;

        for (auto op : stmts) {
            std::set<Reg> u;
            for (auto input : op->inputs()) {
                if (regs.count(input.get_tensor_ptr())) {
                    std::cout << op->get_friendly_name() << " " << input.get_tensor_ptr() << " " << regs[input.get_tensor_ptr()] << std::endl;
                    u.insert(regs[input.get_tensor_ptr()]);
                }
            }
            used.push_back(u);

            std::set<Reg> d;
            if (!std::dynamic_pointer_cast<op::Store>(op)) {
                for (auto output : op->outputs()) {
                    d.insert(regs[output.get_tensor_ptr()]);
                }
            }
            def.push_back(d);
        }

        for (auto n = 0; n < stmts.size(); n++) {
            std::cout << stmts[n]->get_friendly_name() << " " << used[n] << " " << def[n] << std::endl;
        }

        std::cout << std::endl << std::endl;

        // define life intervals
        std::vector<std::set<Reg>> lifeIn(stmts.size(), {});
        std::vector<std::set<Reg>> lifeOut(stmts.size(), {});

        for (auto n = 0; n < stmts.size(); n++) {
            std::cout << lifeIn[n].size() << " " << lifeOut[n].size() << std::endl;
        }

        std::cout << std::endl << std::endl;


        for (int i = 0; i < stmts.size(); i++) {
            for (int n = 0; n < stmts.size(); n++) {
                // std::cout << def[n] << std::endl;
                std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
                lifeIn[n].insert(used[n].begin(), used[n].end());
            }
            for (int n = 0; n < stmts.size(); n++) {
                auto node = stmts[n];
                if (!std::dynamic_pointer_cast<op::Store>(node) /*|| n != stmts.size()-1*/) {
                    for (auto out : node->outputs()) {
                        for (auto port : out.get_target_inputs()) {
                            auto pos = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this());
                            if (pos != stmts.end()) {
                                auto k = pos-stmts.begin();
                                std::cout << n << " " << stmts[n] << " iter = " << k << std::endl;
                                lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                            }
                        }
                    }
                }
            }

            std::cout << std::endl << std::endl;

            for (auto n = 0; n < stmts.size(); n++) {
                std::cout << "# " << i << " " << def[n] << " " << used[n] << " " << lifeIn[n] << " " << lifeOut[n] << std::endl;
            }

            std::cout << std::endl << std::endl;
        }

        struct by_starting {
            auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
                return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
            }
        };

        struct by_ending {
            auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
                return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
            }
        };

        std::set<std::pair<int, int>, by_starting> live_intervals;

        std::reverse(lifeIn.begin(), lifeIn.end());
        auto find_last_use = [lifeIn](int i) -> int {
            int ln = lifeIn.size()-1;
            for (auto& x : lifeIn) {
                if (x.find(i) != x.end()) {
                    return ln;
                }
                ln--;
            }
            return i;
        };

        for (int i = 0; i < stmts.size(); i++) {
            live_intervals.insert(std::make_pair(i, find_last_use(i)));
        }

        for (auto interval : live_intervals) {
            std::cout << interval.first << " ---- " << interval.second << std::endl;
        }

        std::cout << std::endl << std::endl;

        // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
        std::multiset<std::pair<int, int>, by_ending> active;
        std::map<Reg, Reg> register_map;
        std::stack<Reg> bank;
        for (int i = 0; i < 16; i++) bank.push(16-1-i);

        for (auto interval : live_intervals) {
            // check expired
            while (!active.empty()) {
                auto x = *active.begin();
                std::cout << active.size() << std::endl;
                for (auto v : active) {
                    std::cout << v.first << " ---- " << v.second << std::endl;
                }
                std::cout << " [";

                for (std::stack<Reg> dump = bank; !dump.empty(); dump.pop()) {
                    std::cout << dump.top() << ' ';
                }

                std::cout << "]" << std::endl;

                std::cout << "checking " << x.first << " " << x.second << " " << interval.first << " " << interval.second << std::endl;
                if (x.second >= interval.first) {
                    break;
                }
                // FIXME: it would erase both
                active.erase(x);
                bank.push(register_map[x.first]);
                std::cout << x.first << " " << x.second << " was been expired" << std::endl;
                std::cout << std::endl;
                // register_map.erase(x.first);
            }
            // allocate
            if (active.size() == 16) {
                throw ngraph_error("caanot allocate registers for a snippet ");
            } else {
                register_map[interval.first] = bank.top();
                bank.pop();
                active.insert(interval);
            }

            std::cout << interval.first << " " << active.size() << std::endl;
        }

        for (auto v : register_map) {
            std::cout << v.first << " ---- " << v.second + 1 << " [";

            for (std::stack<Reg> dump = bank; !dump.empty(); dump.pop()) {
                std::cout << dump.top() << ' ';
            }

            std::cout << "]" << std::endl;
        }

        std::cout << std::endl << std::endl;

        //
        std::map<std::shared_ptr<descriptor::Tensor>, Reg> physical_regs;

        for (auto reg : regs) {
            physical_regs[reg.first] = register_map[reg.second];
        }

        for (auto r : physical_regs) {
            std::cout << r.first << " " << r.first->get_name() << " " << r.first->get_shape() << " " << r.second << std::endl;
        }
        std::cout << std::endl << std::endl;
    }

    exit(1);

    {
        using Reg = size_t;
        auto ops = f->get_ordered_ops();
        decltype(ops) stmts;
        std::copy_if(ops.begin(), ops.end(), std::back_inserter(stmts), [](decltype(ops[0]) op) {
            return !(std::dynamic_pointer_cast<opset1::Parameter>(op) || std::dynamic_pointer_cast<opset1::Result>(op));
            });

        for (auto op : ops) {
            std::cout << op->get_friendly_name() << std::endl;
        }

        std::cout << std::endl;

        for (auto op : stmts) {
            std::cout << op->get_friendly_name() << std::endl;
        }

        size_t rdx = 0;
        std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
        for (auto op : stmts) {
            for (auto output : op->outputs()) {
                regs[output.get_tensor_ptr()] = rdx++;
            }
        }

        for (auto r : regs) {
            std::cout << r.first << " " << r.first->get_name() << " " << r.first->get_shape() << " " << r.second << std::endl;
        }

        std::vector<std::set<Reg>> used;
        std::vector<std::set<Reg>> def;

        for (auto op : stmts) {
            std::set<Reg> u;
            for (auto input : op->inputs()) {
                if (regs.count(input.get_tensor_ptr())) {
                    std::cout << op->get_friendly_name() << " " << input.get_tensor_ptr() << " " << regs[input.get_tensor_ptr()] << std::endl;
                    u.insert(regs[input.get_tensor_ptr()]);
                }
            }
            used.push_back(u);

            std::set<Reg> d;
            if (!std::dynamic_pointer_cast<op::Store>(op)) {
                for (auto output : op->outputs()) {
                    d.insert(regs[output.get_tensor_ptr()]);
                }
            }
            def.push_back(d);
        }

        for (auto n = 0; n < stmts.size(); n++) {
            std::cout << stmts[n]->get_friendly_name() << used[n] << " " << def[n] << std::endl;
        }

        // define life intervals
        std::vector<std::set<Reg>> lifeIn(stmts.size(), {});
        std::vector<std::set<Reg>> lifeOut(stmts.size(), {});

        for (auto n = 0; n < stmts.size(); n++) {
            std::cout << lifeIn[n].size() << " " << lifeOut[n].size() << std::endl;
        }


        // for (auto n = 0; n < stmts.size(); n++) {
        //     std::cout << n << " " << lifeOut[n] << " " << lifeIn[n] << " " << def[n] << " " << used[n] << " " << stmts[n]->get_friendly_name() << std::endl;

        //     // in[n] = use[n] ∪ (out[n] - def[n])
        //     std::set<Reg> tmp, tmp2;
        //     std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(tmp, tmp.end()));
        //     std::set_union(used[n].begin(), used[n].end(),
        //                tmp.begin(), tmp.end(),
        //                std::inserter(tmp2, tmp.end()));
        //     lifeIn[n] = tmp2;

        //     // out[n] = ∪ in[s]
        //     if (n != 0) {
        //         for (auto k = 0; k < n-1; k++) {
        //             lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
        //         }
        //     }

        //     std::cout << n << " " << lifeOut[n] << " " << lifeIn[n] << std::endl;
        // }

        // for (auto n = 0; n < stmts.size(); n++) {
        //     std::cout << lifeIn[n].size() << " " << lifeOut[n].size() << std::endl;

        //     for (auto x : lifeIn[n]) {
        //         std::cout << x << " ";
        //     }
        //     std::cout << std::endl;

        //     for (auto x : lifeOut[n]) {
        //         std::cout << x << " ";
        //     }
        //     std::cout << std::endl;
        // }

        for (int i = 0; i < stmts.size(); i++) {
            for (int n = 0; n < stmts.size(); n++) {
                // std::cout << def[n] << std::endl;
                std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
                lifeIn[n].insert(used[n].begin(), used[n].end());
            }
            for (int n = 0; n < stmts.size(); n++) {
                auto node = stmts[n];
                if (!std::dynamic_pointer_cast<op::Store>(node) /*|| n != stmts.size()-1*/) {
                    for (auto out : node->outputs()) {
                        for (auto port : out.get_target_inputs()) {
                            auto pos = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this());
                            if (pos != stmts.end()) {
                                auto k = pos-stmts.begin();
                                std::cout << n << " " << stmts[n] << " iter = " << k << std::endl;
                                lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                            }
                        }
                    }
                }
            }

            std::cout << std::endl << std::endl;

            for (auto n = 0; n < stmts.size(); n++) {
                std::cout << "# " << i << " " << def[n] << " " << used[n] << " " << lifeIn[n] << " " << lifeOut[n] << std::endl;
            }

            std::cout << std::endl << std::endl;
        }
    }

    size_t idx_start = 0;
    size_t idx_max = 15; // Get this from targeet machine
    size_t constantID = 0;

    for (auto n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        // nothing to do for function signature
        if (std::dynamic_pointer_cast<opset1::Parameter>(n) || std::dynamic_pointer_cast<opset1::Result>(n)) {
            continue;
        }

        // store only effective address
        if (auto result = std::dynamic_pointer_cast<op::Store>(n)) {
            auto ea = static_cast<int64_t>(f->get_result_index(result) + f->get_parameters().size());
            rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
            continue;
        }
        // std::cout << n->get_type_name() << std::endl;
        // store effective address and procced with vector registers
        if (as_type_ptr<ngraph::op::Load>(n) || as_type_ptr<ngraph::op::BroadcastLoad>(n)) {
            auto source = n->get_input_source_output(0).get_node_shared_ptr();

            if (auto param = as_type_ptr<opset1::Parameter>(source)) {
                auto ea = static_cast<int64_t>(f->get_parameter_index(param));
                rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
            } else if (auto constant = as_type_ptr<opset1::Constant>(source)) {
                auto ea = static_cast<int64_t>(f->get_parameters().size() + f->get_results().size() + 1 + constantID);
                rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
                constantID++;
            } else {
                throw ngraph_error("load/broadcast should follow only Parameter or non-Scalar constant");
            }
        }

        std::vector<size_t> regs; regs.reserve(n->outputs().size());
        for (auto output : n->outputs()) {
            if (idx_start > idx_max) {
                idx_start = 0;
                // FIXME: implement somewhat notmal register allocation logic
                throw ngraph::ngraph_error(std::string("cannot allocate register for ") + n->get_friendly_name());
            }
            regs.push_back(idx_start);
            idx_start++;
        }

        remark(12) << "allocating registers to " << n->get_friendly_name() << " (" << n->get_type_info().name << ") ";
        for (auto reg : regs) {
            std::cout << reg << " ";
        }
        std::cout << std::endl;
        rt["reginfo"] = std::make_shared<VariantWrapper<std::vector<size_t>>>(VariantWrapper<std::vector<size_t>>(regs));
    }

    return false;
}
