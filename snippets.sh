#!/bin/sh

set -e

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./attrs.xml \
    -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/person-attributes-recognition-crossroad-0230/caffe2/onnx/FP32/1/dldt/person-attributes-recognition-crossroad-0230.xml \
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./reid.xml \
    -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/person-reidentification-retail-0270/caffe2/onnx/FP32/1/dldt/person-reidentification-retail-0270.xml \
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./mobilenet.xml \
    -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/mobilenet-v3-large-1.0-224/caffe2/onnx/FP32/1/dldt/mobilenet-v3-large-1.0-224.xml \
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./asl.xml -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/asl-recognition-0004/caffe2/onnx/FP32/1/dldt/asl-recognition-0004.xml \
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./uncased.xml -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/bert-base-uncased/tf/tf_frozen/FP32/1/dldt/bert-base-uncased.xml\
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./small.xml -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/bert-small-uncased-whole-word-masking-squad-0001/caffe2/onnx/FP32/1/dldt/bert-small-uncased-whole-word-masking-squad-0001.xml \
    -niter 1

../bin/intel64/Release/benchmark_app \
    -exec_graph_path ./large.xml -nthreads 1 -nstreams 1 \
    -m /Users/mkolpako/snippets/1/bert-large-uncased-whole-word-masking-squad-fp32-0001/caffe2/onnx/FP32/1/dldt/bert-large-uncased-whole-word-masking-squad-fp32-0001.xml \
    -niter 1
