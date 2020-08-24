// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

#include "jit_generator.hpp"

namespace ngraph {
namespace snippet {

class NGRAPH_API Register {
public:
    enum class Type {
        vector,
        scalar
    };

    Register() = default;

private:
    size_t id;
    Type type;
};

class NGRAPH_API RegisterFile {
public:
    RegisterFile() = default;

    virtual Register& getABIParameters() = 0;

    virtual Register& getFreeRegister() = 0;
};

// Make it so once register is distroed it gets back to the regsister file
class NGRAPH_API CpuRegisterFile : RegisterFile {
public:
    CpuRegisterFile() : RegisterFile() {}

    std::vector<Register> acquire(Register::Type, size_t N) {
        return std::vector<Register>({});
    }
private:
    Register parametersPtr {};
    Register stackPtr;

    std::vector<Register> reserved;
    std::vector<Register> free;
    std::vector<Register> allocated;
    std::vector<Register> scalars;
};

} // namespace snippet
using snippet::Register;
using snippet::RegisterFile;
using snippet::CpuRegisterFile;
} // namespace ngraph