// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/rt_info/register_info.hpp"

#include <ngraph/opsets/opset1.hpp>

#include "ngraph_ops/load.hpp"
#include "ngraph_ops/broadcastload.hpp"

// FIXME: use linear time register allocation algorithm
// Assigning internal `vector` register indexes to Ops (virtual registers)
// After this pass changing order of variables or datafrow lead to invalidation of register assignment
bool ngraph::pass::AssignRegistersPass::run_on_function(std::shared_ptr<Function> f) {
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
                std::cout << def[n] << std::endl;
                std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
                lifeIn[n].insert(used[n].begin(), used[n].end());

                auto node = stmts[n];
                if (!std::dynamic_pointer_cast<op::Store>(node)) {
                    for (auto out : node->outputs()) {
                        for (auto port : out.get_target_inputs()) {
                            auto k = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this())-stmts.begin();
                            std::cout << n << " " << stmts[n] << "iter = " << k << std::endl;
                            lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                        }
                    }
                }
            }

            for (auto n = 0; n < stmts.size(); n++) {
                std::cout << "#1 " << def[n] << " " << used[n] << " " << lifeIn[n] << " " << lifeOut[n] << std::endl;
            }
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
