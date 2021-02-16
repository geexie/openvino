// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/generator.hpp"
#include "transformations/snippets/remarks.hpp"
#include "transformations/snippets/register_info.hpp"

auto ngraph::snippets::getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo {
    auto rt = n->get_rt_info();

    std::vector<size_t> rout;
    if (auto rinfo = rt["reginfo"]) {
        auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
        for (auto reg : reginfo) {
            rout.push_back(reg);
        }
    }

    std::vector<size_t> rin;
    for (auto input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        if (auto rinfo = rt["reginfo"]) {
            auto reginfo = ngraph::as_type_ptr<ngraph::VariantWrapper<std::vector<size_t>>>(rinfo)->get();
            for (auto reg : reginfo) {
                rin.push_back(reg);
            }
        }
    }
    // for (auto r : rin) std::cout << r << " " ;
    // std::cout << std::endl;

    // for (auto r : rout) std::cout << r << " " ;
    // std::cout << std::endl;
    return std::make_pair(rin, rout);
}