// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generator.hpp"
#include "transformations/snippets/generate_pass.hpp"
#include "transformations/snippets/assign_registers_pass.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/generator.hpp"
#include "transformations/rt_info/register_info.hpp"

#include "ngraph_ops/scalar.hpp"
#include "ngraph_ops/nop.hpp"

#include <ngraph/pass/visualize_tree.hpp>

bool ngraph::pass::GenerateCodePass::run_on_function(std::shared_ptr<Function> func) {
    for (auto n : func->get_ordered_ops()) {
        auto regs = ngraph::snippet::getRegisters(n);
        if (m_generator->jitters.find(n->get_type_info()) != m_generator->jitters.end()) {
            m_generator->jitters[n->get_type_info()](n)->emit(regs.first, regs.second);
        } else {
            throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
        }
    }
    return false;
}
