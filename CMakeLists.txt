cmake_minimum_required(VERSION 3.16)

# Optional standalone AAS preparation utility. The UDP bot itself is a Go module.
project(Q2AASPreparation LANGUAGES C CXX)

add_subdirectory(workspace/tools/bspc)
