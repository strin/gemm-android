cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(TestGEMM C CXX)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11 -DCMAKE_BUILD")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -std=c++11")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

message("opencl includes" ${OPENCL_INCLUDES})
include_directories(${OPENCL_INCLUDES} "common")
include_directories(${OPENCL_INCLUDES} "GEMM")

add_executable(intelgemm ${PROJECT_SOURCE_DIR}/GEMM/gemm.cpp
                    ${PROJECT_SOURCE_DIR}/common/basic.cpp
                    ${PROJECT_SOURCE_DIR}/common/cmdparser.cpp
                    ${PROJECT_SOURCE_DIR}/common/oclobject.cpp
                    ${PROJECT_SOURCE_DIR}/common/utils.cpp
                    ${PROJECT_SOURCE_DIR}/common/yuv_utils.cpp
                    ${PROJECT_SOURCE_DIR}/GEMM/cmdoptions.cpp)

target_link_libraries(intelgemm ${OPENCL_LIBS})

