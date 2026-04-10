cmake_minimum_required(VERSION 3.20)
project(arena_native LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(arena_native src/bindings.cpp)
target_compile_features(arena_native PRIVATE cxx_std_20)

