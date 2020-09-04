// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/setup_stack_pass.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>

// Perhaps, it's better to call this pass setup constants
// can this pass to be merged with actual constant generation somehow?
bool ngraph::pass::SetupStackTemporalsOffsetPass::run_on_function(std::shared_ptr<Function> f) {
    size_t table_offset = 0;

    for (auto n : f->get_ordered_ops()) {
        // conseder only operations which require stack to setup constants
        if (auto op = std::dynamic_pointer_cast<opset1::Erf>(n)) {
            auto& rt = n->get_rt_info();
            auto x = static_cast<int64_t>(table_offset);
            rt["stackinfo"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(x));
            table_offset += 14;
        } else if (auto op = std::dynamic_pointer_cast<opset1::Constant>(n)) {
            // FixMe: why not?
            if (op->outputs().size() != 1) {
                throw ngraph_error("constant with more than 1 output is not supported");
            }

            auto out_shape = op->output(0).get_tensor().get_shape();
            if (out_shape == Shape() || ngraph::shape_size(out_shape) == 1) {
                auto& rt = n->get_rt_info();
                auto x = static_cast<int64_t>(table_offset);
                rt["stackinfo"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(x));
                table_offset++;
            } else {
                // refactor make is scalar operation
                // throw ngraph_error("non scalar constant support is not implemented");
            }
        } else if (auto op = std::dynamic_pointer_cast<opset1::Clamp>(n)) {
            auto& rt = n->get_rt_info();
            auto x = static_cast<int64_t>(table_offset);
            rt["stackinfo"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(x));
            table_offset += 2;
        } else if (auto op = std::dynamic_pointer_cast<opset1::Sigmoid>(n)) {
            auto& rt = n->get_rt_info();
            auto x = static_cast<int64_t>(table_offset);
            rt["stackinfo"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(x));
            table_offset += 25*8;
        }
    }

    return false;
}