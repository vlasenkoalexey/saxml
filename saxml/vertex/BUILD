load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_binary", "pytype_strict_library")

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

load(
    "//saxml:saxml.bzl",
    "proto_library",
    "py_grpc_library",
    "py_proto_library",
)

pytype_strict_binary(
    name = "launcher",
    srcs = ["launcher.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":grpc_prediction_server",
        ":sax_model_server",
        ":http_prediction_server",
        ":translate",
        "//third_party/py/grpcio",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//saxml/client/python:sax",
        "//third_party/py/tornado",
    ],
)

pytype_strict_library(
    name = "http_prediction_server",
    srcs = ["http_prediction_server.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":sax_model_server",
        ":translate",
        "//third_party/py/grpcio",
        "//third_party/py/absl-py/logging",
        "//saxml/client/python:sax",
        "//saxml/protobuf:admin_py_pb2",
        "//saxml/protobuf:admin_py_pb2_grpc",
        "//third_party/py/tornado",
    ],
)

proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    has_services = True,
    deps = [
        "//saxml/protobuf:common_proto",
        "//saxml/protobuf:lm_proto",
    ],
)

py_proto_library(
    name = "prediction_service_py_pb2",
    api_version = 2,
    deps = [":prediction_service_proto"],
    extra_deps = [
      "//saxml/protobuf:common_py_pb2", 
      "//saxml/protobuf:lm_py_pb2"],
)

py_grpc_library(
    name = "prediction_service_py_pb2_grpc",
    srcs = [":prediction_service_proto"],
    deps = [":prediction_service_py_pb2"],
)

pytype_strict_library(
    name = "grpc_prediction_server",
    srcs = ["grpc_prediction_server.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":prediction_service_py_pb2_grpc",
        ":sax_model_server",
        ":translate",
        "//third_party/py/grpcio",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//saxml/client/python:sax",
        "//saxml/protobuf:admin_py_pb2",
        "//saxml/protobuf:admin_py_pb2_grpc",
    ],
)

pytype_strict_library(
    name = "sax_model_server",
    srcs = ["sax_model_server.py"],
    deps = [
        ":constants",
        "//third_party/py/absl-py/logging",
        "//third_party/py/tornado",
    ],
)

pytype_strict_library(
    name = "translate",
    srcs = ["translate.py"],
    deps = [
        ":constants",
        "//saxml/client/python:sax",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pybind_extension(
    name = "my_pb_mod",  # This name is not actually created!
    srcs = ["my_pb_mod.cc"],
)

py_library(
    name = "my_pb_mod",
    data = [":my_pb_mod.so"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    deps = [
        ":my_pb_mod"
    ],
)