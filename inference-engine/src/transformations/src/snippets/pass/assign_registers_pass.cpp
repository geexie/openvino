// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/assign_registers_pass.hpp"
#include "snippets/generator.hpp"

#include "ngraph_ops/load.hpp"
#include "ngraph_ops/broadcastload.hpp"

#include <ngraph/pass/visualize_tree.hpp>

#include <ngraph/variant.hpp>

#include "snippets/remarks.hpp"

// Assigning internal `vector` register indexes to Ops (virtual registers)
bool ngraph::pass::AssignRegistersPass::run_on_function(std::shared_ptr<Function> f) {
    size_t idx_start = 0;
    size_t idx_max = 15;
    size_t constantID = 0;

    for (auto n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        // nothing to do for function signature
        if (std::dynamic_pointer_cast<opset1::Parameter>(n)) {
            continue;
        }

        // store only effective address
        if (auto result = std::dynamic_pointer_cast<opset1::Result>(n)) {
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
                throw ngraph::ngraph_error(std::string("cannot allocate register for ") + n->get_friendly_name());
            }
            regs.push_back(idx_start);
            idx_start++;
        }

        remark(2) << "allocating registers to " << n->get_friendly_name() << " ";
        for (auto reg : regs) {
            remark(2) << reg << " ";
        }
        remark(2) << std::endl;
        rt["reginfo"] = std::make_shared<VariantWrapper<std::vector<size_t>>>(VariantWrapper<std::vector<size_t>>(regs));
    }

    return false;
}

template <typename T>
ngraph::VariantImpl<T>::~VariantImpl() { }

template class ngraph::VariantImpl<std::vector<size_t>>;

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<std::vector<size_t>>::type_info;
