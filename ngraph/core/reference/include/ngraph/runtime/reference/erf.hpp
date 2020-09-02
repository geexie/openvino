//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <cstddef>

static float error_function(float x) {
    const float clip_bound = 2.86f;
    //  Points clip_bound and -clip_bound are extremums for this polynom
    //  So in order to provide better accuracy comparing to std::erf we have to clip input range
    if (x > clip_bound)
        return 1;
    if (x < -clip_bound)
        return -1;

    //  A polynomial approximation of the error function
    const float erfNumerator[4] = { 90.0260162353515625f, 2232.00537109375f,
        7003.3251953125f, 55592.30078125f };
    const float erfDenominator[5] = { 33.56171417236328125f, 521.35797119140625f,
        4594.32373046875f, 22629.0f, 49267.39453125f };
    float polynom = 9.60497379302978515625f;
    float x2 = x * x;
    for (float c : erfNumerator) {
        polynom = polynom * x2 + c;
    }
    x *= polynom;
    polynom = 1.0f;
    for (float c : erfDenominator) {
        polynom = polynom * x2 + c;
    }
    return x / polynom;
}

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void erf(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = (T)error_function((float)arg[i]);
                }
            }
        }
    }
}
