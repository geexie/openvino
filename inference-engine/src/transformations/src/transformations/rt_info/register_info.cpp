// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/register_info.hpp"

template class ngraph::VariantImpl<std::vector<size_t>>;

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<std::vector<size_t>>::type_info;
