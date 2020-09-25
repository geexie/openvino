// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generate_pass.hpp"

bool ngraph::pass::GenerateCodePass::run_on_function(std::shared_ptr<Function> func) {
    // for (auto n : func->get_ordered_ops()) {
        // auto regs = ngraph::snippet::getRegisters(n);
    //     if (m_generator->jitters.find(n->get_type_info()) != m_generator->jitters.end()) {
    //         m_generator->jitters[n->get_type_info()](n)->emit(regs.first, regs.second);
    //     } else {
    //         throw ngraph::ngraph_error(std::string("unknown operation ") + n->get_type_info().name);
    //     }
    // }

    return false;
}
