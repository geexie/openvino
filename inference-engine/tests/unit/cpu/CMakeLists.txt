# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
            ${IE_MAIN_SOURCE_DIR}/src/transformations/include
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            unitTestUtils
            mkldnn
            inference_engine_transformations
            inference_engine_lp_transformations
            ngraphFunctions
        ADD_CPPLINT
        LABELS
            CPU
)

ie_faster_build(${TARGET_NAME}
    UNITY
)
include_directories(
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
    ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include)

include_directories(SYSTEM
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include)
