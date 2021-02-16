// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/register_info.hpp"

#include <ngraph/opsets/opset1.hpp>

#include "transformations/snippets/load.hpp"
#include "transformations/snippets/scalar.hpp"
#include "transformations/snippets/broadcastload.hpp"

#include <iterator>

// Assigning internal `vector` register indexes to Ops (virtual registers)
// After this pass changing order of variables or datafrow lead to invalidation of register assignment
bool ngraph::pass::AssignRegistersPass::run_on_function(std::shared_ptr<Function> f) {
    int reg64_tmp_start { 8 }; // R8, R9, R10, R11, R12, R13, R14, R15 inputs+outputs+1
    using Reg = size_t;
    auto ops = f->get_ordered_ops();
    decltype(ops) stmts;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(stmts), [](decltype(ops[0]) op) {
        return !(std::dynamic_pointer_cast<opset1::Parameter>(op) || std::dynamic_pointer_cast<opset1::Result>(op));
        });

    size_t rdx = 0;
    std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
    for (auto op : stmts) {
        for (auto output : op->outputs()) {
            regs[output.get_tensor_ptr()] = rdx++;
            // std::cout << output.get_node()->get_friendly_name() << " " << output.get_node() << " " << op << " " << op->outputs().size() <<
            // output.get_tensor_ptr() << std::endl;
        }
    }

    std::vector<std::set<Reg>> used;
    std::vector<std::set<Reg>> def;

    for (auto op : stmts) {
        std::set<Reg> u;
        for (auto input : op->inputs()) {
            if (regs.count(input.get_tensor_ptr())) {
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

    // for (auto n = 0; n < stmts.size(); n++) {
        // std::cout << stmts[n]->get_friendly_name() << used[n] << " " << def[n] << std::endl;
    // }

    // define life intervals
    std::vector<std::set<Reg>> lifeIn(stmts.size(), std::set<Reg>());
    std::vector<std::set<Reg>> lifeOut(stmts.size(), std::set<Reg>());

    for (size_t i = 0; i < stmts.size(); i++) {
        for (size_t n = 0; n < stmts.size(); n++) {
            std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
            lifeIn[n].insert(used[n].begin(), used[n].end());
        }
        for (size_t n = 0; n < stmts.size(); n++) {
            auto node = stmts[n];
            if (!std::dynamic_pointer_cast<op::Store>(node) /*|| n != stmts.size()-1*/) {
                for (auto out : node->outputs()) {
                    for (auto port : out.get_target_inputs()) {
                        auto pos = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this());
                        if (pos != stmts.end()) {
                            auto k = pos-stmts.begin();
                            lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                        }
                    }
                }
            }
        }

        // std::cout << std::endl << std::endl;

        // for (auto n = 0; n < stmts.size(); n++) {
        //     std::cout << "# " << i << " " << def[n] << " " << used[n] << " " << lifeIn[n] << " " << lifeOut[n] << std::endl;
        // }

        // std::cout << std::endl << std::endl;
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

    for (size_t i = 0; i < stmts.size(); i++) {
        live_intervals.insert(std::make_pair(i, find_last_use(i)));
    }

    // for (auto interval : live_intervals) {
    //     std::cout << interval.first << " ---- " << interval.second << std::endl;
    // }

    // std::cout << std::endl << std::endl;

    // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
    std::multiset<std::pair<int, int>, by_ending> active;
    std::map<Reg, Reg> register_map;
    std::stack<Reg> bank;
    for (int i = 0; i < 16; i++) bank.push(16-1-i);

    for (auto interval : live_intervals) {
        // check expired
        while (!active.empty()) {
            auto x = *active.begin();
            // std::cout << "start " << interval.first << "active.size() = " << active.size() << std::endl;
            // for (auto v : active) {
            //     std::cout << v.first << " ---- " << v.second << std::endl;
            // }
            // std::cout << "bank = [";

            // for (std::stack<Reg> dump = bank; !dump.empty(); dump.pop()) {
            //     std::cout << dump.top() << ' ';
            // }
            // std::cout << "]" << std::endl;

            // std::cout << "checking " << x.first << " ---- " << x.second << " " << interval.first << " ---- " << interval.second << std::endl;
            if (x.second >= interval.first) {
                break;
            }
            // FIXME: it would erase both
            active.erase(x);
            bank.push(register_map[x.first]);
            // std::cout << x.first << " " << x.second << " was been expired" << std::endl;
            // std::cout << std::endl;
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

        // std::cout << "end " << interval.first << " active.size() = " << active.size() << std::endl;
    }

    // for (auto v : register_map) {
    //     std::cout << v.first << " ---- " << v.second + 1 << std::endl;
    // }

    // std::cout << "bank = " << " [";

    // for (std::stack<Reg> dump = bank; !dump.empty(); dump.pop()) {
    //     std::cout << dump.top() << ' ';
    // }

    // std::cout << "]" << std::endl;
    // std::cout << std::endl << std::endl;

    //
    std::map<std::shared_ptr<descriptor::Tensor>, Reg> physical_regs;

    for (auto reg : regs) {
        physical_regs[reg.first] = register_map[reg.second];
    }

    // for (auto r : physical_regs) {
    //     std::cout << r.first << " " << r.first->get_name() << " " << r.first->get_shape() << " " << r.second << std::endl;
    // }
    // std::cout << std::endl << std::endl;

    size_t constantID = 0;

    for (auto n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        // nothing to do for function signature
        if (std::dynamic_pointer_cast<opset1::Parameter>(n) || std::dynamic_pointer_cast<opset1::Result>(n)) {
            continue;
        }

        // store only effective address
        if (auto result = std::dynamic_pointer_cast<op::Store>(n)) {
            auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_result_index(result) + f->get_parameters().size());
            rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
            continue;
        }
        // std::cout << n->get_type_name() << std::endl;
        // store effective address and procced with vector registers
        if (as_type_ptr<ngraph::op::Load>(n) || as_type_ptr<ngraph::op::BroadcastLoad>(n)) {
            auto source = n->get_input_source_output(0).get_node_shared_ptr();

            if (auto param = as_type_ptr<opset1::Parameter>(source)) {
                auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_parameter_index(param));
                rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
            } else if (auto constant = as_type_ptr<opset1::Constant>(source)) {
                auto ea = reg64_tmp_start+static_cast<int64_t>(f->get_parameters().size() + f->get_results().size() + 1 + constantID);
                rt["effectiveAddress"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(ea));
                constantID++;
            } else {
                throw ngraph_error("load/broadcast should follow only Parameter or non-Scalar constant");
            }
        }

        std::vector<size_t> regs; regs.reserve(n->outputs().size());
        for (auto output : n->outputs()) {
            auto allocated = physical_regs[output.get_tensor_ptr()];
            regs.push_back(allocated);
        }

        // remark(12) << "allocating registers to " << n->get_friendly_name() << " (" << n->get_type_info().name << ") ";
        // for (auto reg : regs) {
        //     std::cout << reg << " ";
        // }
        // std::cout << std::endl;
        rt["reginfo"] = std::make_shared<VariantWrapper<std::vector<size_t>>>(VariantWrapper<std::vector<size_t>>(regs));
    }

    return false;
}
