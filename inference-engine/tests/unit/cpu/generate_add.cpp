// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>
#include <array>

#include <transformations/snippets/generator.hpp>

#include <ngraph/function.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include <ngraph_ops/subgraph.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_gelu.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

using namespace testing;

auto gen_inputs(const ngraph::Shape& shape, size_t n = 2) -> std::vector<std::vector<std::uint8_t>>{
    std::vector<std::vector<std::uint8_t>> referenceInputs(n);
    for (int k = 0; k < n; k++) {
        referenceInputs[k].resize(ngraph::shape_size(shape)*sizeof(float));
        float* in0 = reinterpret_cast<float*>(&referenceInputs[k][0]);

        for (int i = 0; i < ngraph::shape_size(shape); i++) {
            if (k % 3 == 0) {
                in0[i] = i/2048.f;
            } else if (k % 3 == 1) {
                in0[i] = 1-i/2048.f;
            } else {
                in0[i] = i/1024.f;
            }
        }
    }
    return referenceInputs;
}

auto compare(std::shared_ptr<ngraph::Function>& s, std::shared_ptr<ngraph::Function>& f, std::vector<std::vector<std::uint8_t>>& in) -> bool{
    std::vector<std::vector<std::uint8_t>> act = ngraph::helpers::interpreterFunction(s, in);
    std::vector<std::vector<std::uint8_t>> exp = ngraph::helpers::interpreterFunction(f, in);

    const float* pexp = reinterpret_cast<float*>(&exp[0][0]);
    const float* pact = reinterpret_cast<float*>(&act[0][0]);

    bool isCorrect = true;
    for (int i = 0; i < ngraph::shape_size(f->get_result()->get_shape()); i++) {
        if (std::abs(pexp[i]-pact[i]) > std::numeric_limits<float>::epsilon()
            || std::isnan(pexp[i]) != std::isnan(pact[i])) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << " diff " << std::abs(pexp[i]-pact[i]) << std::endl;
        }
    }
    return isCorrect;
}

auto wrapAsSnippet(std::shared_ptr<ngraph::Function>& f, const ngraph::Shape& shape0, const ngraph::Shape& shape1)
-> std::shared_ptr<ngraph::Function>{
    auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input0, input1}, ngraph::clone_function(*f.get()));
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input0, input1});
}

auto wrapAsSnippet(std::shared_ptr<ngraph::Function>& f, const ngraph::Shape& shape0)
-> std::shared_ptr<ngraph::Function>{
    auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input0}, ngraph::clone_function(*f.get()));
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input0});
}

auto saveToDot(std:: string name, std::shared_ptr<ngraph::Function>& f) -> void {
// #define ENABLE_SAVE_TO_DOT
#if defined (ENABLE_SAVE_TO_DOT)
    std::vector<std::shared_ptr<ngraph::Function>> module{f};
    ngraph::pass::VisualizeTree(name).run_on_module(module);
#else
    (void) name;
    (void) f;
#endif
}

const auto defaultShape = ngraph::Shape{1, 1, 1, 8*2+7};
const auto vecShape4d = ngraph::Shape{1, 4, 16, 32};

TEST(SnippetsTests, GenerateAddParams) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto f = ([] (const ngraph::Shape& shape) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input0, input1});
    })(shape);

    auto s = wrapAsSnippet(f, shape, shape);
    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);
    saveToDot("GenerateAddParams.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddConstant) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto f = ([] (const ngraph::Shape& shape) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);

        std::vector<float> vals(ngraph::shape_size(shape));
        for (int i = 0; i < ngraph::shape_size(shape); i++) {
            vals[i] = 1-i/2048.f;
        }
        auto input1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, shape, vals);
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input0});
    })(shape);

    auto s = wrapAsSnippet(f, shape);
    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);
    saveToDot("GenerateAddConstant.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddConstantScalar) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto f = ([] (const ngraph::Shape& shape) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        // fix for non empty shape
        auto input1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, /*ngraph::Shape{1}*/ngraph::Shape(), std::vector<float>({42.f}));
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input0});
    })(shape);

    auto s = wrapAsSnippet(f, shape);
    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);
    saveToDot("GenerateAddConstantScalar.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

#if 0
TEST(SnippetsTests, GenerateAddBroadcastX) {
    auto shape0 = ngraph::Shape{1, 42, 16, 64};
    auto shape1 = ngraph::Shape{1, 42, 16,  1};

    auto f = ([] (const ngraph::Shape& shape0, const ngraph::Shape& shape1) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);
        auto neg    = std::make_shared<ngraph::opset1::Negative>(add);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{neg}, ngraph::ParameterVector{input0, input1});
    })(shape0, shape1);

    auto s = wrapAsSnippet(f, shape0, shape1);

    std::vector<std::vector<std::uint8_t>> referenceInputs(2);
    referenceInputs[0].resize(ngraph::shape_size(shape0)*sizeof(float));
    referenceInputs[1].resize(ngraph::shape_size(shape1)*sizeof(float));

    float* in0 = reinterpret_cast<float*>(&referenceInputs[0][0]);
    float* in1 = reinterpret_cast<float*>(&referenceInputs[1][0]);
    for (int i = 0; i < ngraph::shape_size(shape0); i++) {
        in0[i] = i/2048.f;
    }
    in1[0] = 1.f;

    auto isCorrect = compare(s, f, referenceInputs);

    // std::vector<std::vector<std::uint8_t>> act = ngraph::helpers::interpreterFunction(s, referenceInputs);
    // std::vector<std::vector<std::uint8_t>> exp = ngraph::helpers::interpreterFunction(f, referenceInputs);

    // std::cout << ngraph::shape_size(shape0) << " tensor size" << std::endl;

    // const float* pexp = reinterpret_cast<float*>(&exp[0][0]);
    // const float* pact = reinterpret_cast<float*>(&act[0][0]);

    // bool isCorrect = true;
    // for (int i = 0; i < ngraph::shape_size(shape0); i++) {
    //     if (pexp[i] != pact[i]) {
    //         isCorrect = false;
    //         std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
    //     }
    // }

    saveToDot("GenerateAddBroadcastX.dot", s);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}
#endif

TEST(SnippetsTests, GenerateAddBroadcastX2Edges) {
    auto shape0 = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};
    auto shape1 = ngraph::Shape{1, /*4*/1, /*16*/1, 1};

    auto f = ([] (const ngraph::Shape& shape0, const ngraph::Shape& shape1) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);

        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);
        auto mul    = std::make_shared<ngraph::opset1::Add>(input1, input2);

        // auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4},
            // std::vector<int64_t>{ (int64_t)shape0[0], (int64_t)shape0[1], (int64_t)shape0[2], (int64_t)shape0[3]});

        // auto bct1 = std::make_shared<ngraph::opset1::Broadcast>(mul, target_shape);
        auto sub = std::make_shared<ngraph::opset1::Add>(add, mul);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{sub}, ngraph::ParameterVector{input0, input1, input2});
    })(shape0, shape1);

    auto s = ([f] (const ngraph::Shape& shape0, const ngraph::Shape& shape1) -> std::shared_ptr<ngraph::Function>{
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input3 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto input4 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input2, input3, input4}, ngraph::clone_function(*f.get()));
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input2, input3, input4});
    })(shape0, shape1);

    std::vector<std::vector<std::uint8_t>> referenceInputs(3);
    referenceInputs[0].resize(ngraph::shape_size(shape0)*sizeof(float));
    referenceInputs[1].resize(ngraph::shape_size(shape1)*sizeof(float));
    referenceInputs[2].resize(ngraph::shape_size(shape1)*sizeof(float));

    float* in0 = reinterpret_cast<float*>(&referenceInputs[0][0]);
    float* in1 = reinterpret_cast<float*>(&referenceInputs[1][0]);
    float* in2 = reinterpret_cast<float*>(&referenceInputs[2][0]);
    for (int i = 0; i < ngraph::shape_size(shape0); i++) {
        in0[i] = i/2048.f;
    }

    in1[0] = 1.f;
    in2[0] = 0.42f;

    // ToDO: implement tile selection logic & broadcast emission to make it working. Broadcast substitution works

    std::vector<std::vector<std::uint8_t>> act = ngraph::helpers::interpreterFunction(s, referenceInputs);
    std::vector<std::vector<std::uint8_t>> exp = ngraph::helpers::interpreterFunction(f, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0][0]);
    const float* pact = reinterpret_cast<float*>(&act[0][0]);

    bool isCorrect = true;
    for (int i = 0; i < ngraph::shape_size(shape0); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    saveToDot("GenerateAddBroadcastX2Edges.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}
#if 1
TEST(SnippetsTests, GenerateAddBroadcastY) {
    auto shape0 = ngraph::Shape{1, /*4*/1, /*16*/4, /*32*/8*2+7};
    auto shape1 = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto f = ([] (const ngraph::Shape& shape0, const ngraph::Shape& shape1) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input0, input1});
    })(shape0, shape1);

    auto s = wrapAsSnippet(f, shape0, shape1);

    std::vector<std::vector<std::uint8_t>> referenceInputs(2);
    referenceInputs[0].resize(ngraph::shape_size(shape0)*sizeof(float));
    referenceInputs[1].resize(ngraph::shape_size(shape1)*sizeof(float));

    float* in0 = reinterpret_cast<float*>(&referenceInputs[0][0]);
    float* in1 = reinterpret_cast<float*>(&referenceInputs[1][0]);
    for (int i = 0; i < ngraph::shape_size(shape0); i++) {
        in0[i] = i/2048.f;
    }
    for (int i = 0; i < ngraph::shape_size(shape1); i++) {
        in1[i] = 1-i/2048.f;
    }

    std::vector<std::vector<std::uint8_t>> act = ngraph::helpers::interpreterFunction(s, referenceInputs);
    std::vector<std::vector<std::uint8_t>> exp = ngraph::helpers::interpreterFunction(f, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0][0]);
    const float* pact = reinterpret_cast<float*>(&act[0][0]);

    bool isCorrect = true;
    for (int i = 0; i < ngraph::shape_size(shape0); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    saveToDot("GenerateAddBroadcastY.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegate) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto f = ([] (const ngraph::Shape& shape) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto add    = std::make_shared<ngraph::opset1::Add>(input0, input1);
        auto nagate = std::make_shared<ngraph::opset1::Negative>(add);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{nagate}, ngraph::ParameterVector{input0, input1});
    })(shape);

    auto s = ([f] (const ngraph::Shape& shape) -> std::shared_ptr<ngraph::Function>{
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto input3 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
        auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input2, input3}, ngraph::clone_function(*f.get()));
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input2, input3});
    })(shape);

    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);

    saveToDot("GenerateAddNegate.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAdd) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto add = std::make_shared<ngraph::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ngraph::opset1::Negative>(add);
    auto input3 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto add2 = std::make_shared<ngraph::opset1::Add>(nagate, input3);
    std::shared_ptr<ngraph::Function> f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add2}, ngraph::ParameterVector{input1, input2, input3});

    auto input11 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto input21 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto input31 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input11, input21, input31}, ngraph::clone_function(*f.get()));
    std::shared_ptr<ngraph::Function>  s = std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input11, input21, input31});

    auto referenceInputs = gen_inputs(shape, 3);
    bool isCorrect = compare(s, f, referenceInputs);

    saveToDot("GenerateAddNegateAdd.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAddMultiEdge) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto add    = std::make_shared<ngraph::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ngraph::opset1::Negative>(add);
    auto add2 = std::make_shared<ngraph::opset1::Add>(nagate, input1);
    std::shared_ptr<ngraph::Function> f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add2}, ngraph::ParameterVector{input1, input2});

    auto s = wrapAsSnippet(f, shape, shape);
    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAddMultiEdgeConst) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto input2 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.42});
    auto add = std::make_shared<ngraph::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ngraph::opset1::Negative>(add);
    auto add2 = std::make_shared<ngraph::opset1::Add>(nagate, input1);
    std::shared_ptr<ngraph::Function> f = std::make_shared<ngraph::Function>(ngraph::NodeVector{add2}, ngraph::ParameterVector{input1});

    auto input11 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input11}, ngraph::clone_function(*f.get()));
    std::shared_ptr<ngraph::Function>  s = std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input11});

    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateErf) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto gelu   = std::make_shared<ngraph::opset1::Erf>(input1);
    std::shared_ptr<ngraph::Function> f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input1});

    auto input11 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input11}, ngraph::clone_function(*f.get()));
    std::shared_ptr<ngraph::Function> s = std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input11});

    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);

    saveToDot("GenerateErf.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateGelu) {
    auto shape = ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7};

    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto gelu   = std::make_shared<ngraph::opset2::Gelu>(input1);
    // auto gelu   = std::make_shared<ngraph::opset1::Erf>(input1);
    std::shared_ptr<ngraph::Function> f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input1});

    // saveToDot("snippet_gelu0.dot", f);
    ngraph::pass::InitNodeInfo().run_on_function(f);
    ngraph::pass::ConvertGELU().run_on_function(f);
    ngraph::pass::ConstantFolding().run_on_function(f);
    // saveToDot("snippet_gelu1.dot", f);

    auto input11 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape);
    auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input11}, ngraph::clone_function(*f.get()));
    std::shared_ptr<ngraph::Function> s = std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input11});

    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);

    saveToDot("GenerateGelu.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

// ToDO: implement tile selection logic & broadcast emission to make it working. Broadcast substitution works
TEST(SnippetsTests, GenerateAddBroadcastAutomatic) {
    std::array<ngraph::Shape, 3> shapes {
        ngraph::Shape{1, /*4*/1, /*16*/1, /*32*/8*2+7},
        ngraph::Shape{1, /*4*/1, /*16*/1, 1},
        ngraph::Shape{1, /*4*/1, /*16*/1, 1}
    };

    auto f = ([] (const ngraph::Shape& shape0, const ngraph::Shape& shape1, const ngraph::Shape& shape2) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape2);

        auto add = std::make_shared<ngraph::opset1::Add>(input0, input1);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(input1, input2);
        auto sub = std::make_shared<ngraph::opset1::Subtract>(add, mul);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{sub}, ngraph::ParameterVector{input0, input1, input2});
    })(shapes[0], shapes[1], shapes[2]);

    auto s = ([f] (const ngraph::Shape& shape0, const ngraph::Shape& shape1, const ngraph::Shape& shape2) -> std::shared_ptr<ngraph::Function>{
        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape0);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape1);
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, shape2);
        auto snippet = std::make_shared<ngraph::op::Subgraph>(ngraph::OutputVector{input0, input1, input2}, ngraph::clone_function(*f.get()));
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{snippet}, ngraph::ParameterVector{input0, input1, input2});
    })(shapes[0], shapes[1], shapes[2]);

    std::vector<std::vector<std::uint8_t>> referenceInputs(3);
    for (int k = 0; k < referenceInputs.size(); k++) {
        referenceInputs[k].resize(ngraph::shape_size(shapes[k])*sizeof(float));

        auto in0 = reinterpret_cast<float*>(&referenceInputs[k][0]);
        for (int i = 0; i < ngraph::shape_size(shapes[k]); i++) {
            in0[i] = k == 0 ? /*i/2048.f*/0.f : (k == 1 ? 1.f : 0.42f);
        }
    }

    auto exp = ngraph::helpers::interpreterFunction(f, referenceInputs);
    auto act = ngraph::helpers::interpreterFunction(s, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0][0]);
    const float* pact = reinterpret_cast<float*>(&act[0][0]);

    bool isCorrect = true;
    for (int i = 0; i < ngraph::shape_size(shapes[0]); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    // saveToDot("GenerateAddBroadcastAutomatic.dot", s);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}
#endif