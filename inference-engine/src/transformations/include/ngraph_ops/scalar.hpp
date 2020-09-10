// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace op {

/// \brief Class for constants.
class TRANSFORMATIONS_API Scalar  : public Constant {
public:
    static constexpr NodeTypeInfo type_info{"Scalar", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Scalar() = default;
    Scalar(const std::shared_ptr<runtime::Tensor>& tensor) : Constant(tensor) {}
    template <typename T>
    Scalar(const element::Type& type, Shape shape, const std::vector<T>& values) : Constant(type, shape, values) {}
    Scalar(const element::Type& type, const Shape& shape) : Constant(type, shape) {}
    template <class T, class = typename std::enable_if<std::is_fundamental<T>::value>::type>
    Scalar(const element::Type& type, Shape shape, T value) : Constant(type, shape, value) {}
    Scalar(const element::Type& type, Shape shape, const std::vector<std::string>& values) : Constant(type, shape, values) {}
    Scalar(const element::Type& type, const Shape& shape, const void* data) : Constant(type, shape, data) {}

    Scalar(const Constant& other) : Constant(other) {}
    Scalar(const Scalar& other) : Constant(other) {}
    Scalar& operator=(const Scalar&) = delete;
    ~Scalar() override {}
};

} // namespace op
} // namespace ngraph