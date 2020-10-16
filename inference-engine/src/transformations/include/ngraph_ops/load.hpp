// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/op/op.hpp>
#include <ngraph/op/parameter.hpp>

namespace ngraph {
namespace op {

/// ====================================================================================================================================================
/// `BlockedParameter` is used to represent blocked input for a subgraph.
/// ====================================================================================================================================================
class TRANSFORMATIONS_API BlockedParameter : public Parameter {
public:
    static constexpr NodeTypeInfo type_info{"BlockedParameter", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    BlockedParameter() = default;
    BlockedParameter(const ngraph::element::Type& element_type, const PartialShape& pshape, const bool cacheable = false)
        : Parameter(element_type, pshape, cacheable) {
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<BlockedParameter>(m_element_type, m_partial_shape);
    }
private:
};

/// ====================================================================================================================================================
/// ScalarLoad == scalar instruction + post increment
/// Load (VectorLoad) == vector instruction + post increment
/// BroadcastLoad == scalar instruction - post increment
/// BlockedLoad == vector instruction - post increment
/// ====================================================================================================================================================
class TRANSFORMATIONS_API Load : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Load", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Load(const Output<Node>& x);
    Load() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
};

class TRANSFORMATIONS_API ScalarLoad : public Load {
public:
    static constexpr NodeTypeInfo type_info{"ScalarLoad", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ScalarLoad(const Output<Node>& x);
    ScalarLoad() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarLoad>(new_args.at(0));
    }
private:
};

class TRANSFORMATIONS_API BlockedLoad : public Load {
public:
    static constexpr NodeTypeInfo type_info{"BlockedLoad", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    BlockedLoad(const Output<Node>& x);
    BlockedLoad() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<BlockedLoad>(new_args.at(0));
    }
private:
};

class TRANSFORMATIONS_API VectorLoad : public Load {
public:
    static constexpr NodeTypeInfo type_info{"VectorLoad", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    VectorLoad(const Output<Node>& x);
    VectorLoad() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<VectorLoad>(new_args.at(0));
    }
private:
};

class TRANSFORMATIONS_API Store : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Store", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Store(const Output<Node>& x);
    Store() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;

private:
};

class TRANSFORMATIONS_API VectorStore : public Store {
public:
    static constexpr NodeTypeInfo type_info{"VectorStore", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    VectorStore(const Output<Node>& x);
    VectorStore() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<VectorStore>(new_args.at(0));
    }
private:
};

class TRANSFORMATIONS_API ScalarStore : public Store {
public:
    static constexpr NodeTypeInfo type_info{"ScalarStore", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ScalarStore(const Output<Node>& x);
    ScalarStore() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarStore>(new_args.at(0));
    }
private:
};

class TRANSFORMATIONS_API LEA : public Op {
public:
    static constexpr NodeTypeInfo type_info{"LEA", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    LEA(const Output<Node>& x);
    LEA() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<LEA>(new_args.at(0));
    }

    void validate_and_infer_types() override;
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
private:
};

} // namespace op
} // namespace ngraph