// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generate_constant_tables.hpp"

bool ngraph::pass::GenerateConstntTables::run_on_function(std::shared_ptr<Function> func) {
    for (auto n : func->get_ordered_ops()) {
        // std::cout << "Remark: generate contant tables for " << n->get_friendly_name() << std::endl;

        if (auto op = std::dynamic_pointer_cast<opset1::Erf>(n)) {
            m_generator->emit_table(op);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Constant>(n)) {
            m_generator->emit_table(op);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Clamp>(n)) {
            m_generator->emit_table(op);
        } else if (auto op = std::dynamic_pointer_cast<opset1::Sigmoid>(n)) {
            m_generator->emit_table(op);
        }
    }

    return false;
}
