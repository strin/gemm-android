// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly


#include <limits>
#include <cmath>

#include "cmdoptions.hpp"

using namespace std;


#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif

CmdParserGEMM::CmdParserGEMM (int argc, const char** argv) :
    CmdParserCommon(argc, argv),
    size(
        *this,
        's',
        "size",
        "<integer>",
        "Size of matrix in elements.",
        3968
    ),
    iterations(
        *this,
        'i',
        "iterations",
        "<integer>",
        "Number of kernel invocations. For each invoction, "
            "performance information will be printed. "
            "Zero is allowed: in this case no kernel invocation "
            " is performed but all other host stuff is created.",
        10
    ),
    arithmetic(
        *this,
        'a',
        "arithmetic",
        "",
        "Type of elements and all calculations.",
        "float"
    ),
    arithmetic_float(arithmetic, "float"),
    arithmetic_double(arithmetic, "double"),
    kernel(
        *this,
        0,
        "kernel",
        "",
        "Determines format of matrices involved in multiplication. "
            "There are two supported form: nn and nt; nn is for case when "
            "both matrices A and B are in column-major form; nt is for case "
            "when A is in column-major form, but B is in row major format "
            "(i.e. transposed). Matrices A and C are always in column major "
            "format.",
        "nn"
    ),
    kernel_nn(kernel, "nn"),
    kernel_nt(kernel, "nt"),
    validation(
        *this,
        0,
        "validation",
        "",
        "Enables validation procedure on host (slow for big matrices).",
        false
    ),
    tile_size_M(
        *this,
        0,
        "tile-size-M",
        "<integer>",
        "Size of tile for matrix A.",
        1
    ),
    tile_group_M(
        *this,
        0,
        "tile-group-M",
        "<integer>",
        "Grouping parameter for matrix A. "
            "Also defines work group size in 0-dimension.",
        16
    ),
    tile_size_N(
        *this,
        0,
        "tile-size-N",
        "<integer>",
        "Size of tile for matrix B.",
        128
    ),
    tile_group_N(
        *this,
        0,
        "tile-group-N",
        "<integer>",
        "Grouping parameter for matrix B. "
            "Also defines work group size in 1-dimension.",
        1
    ),
    tile_size_K(
        *this,
        0,
        "tile-size-K",
        "<integer>",
        "Size of block in dot-product direction (applicable for "
            "nn kernel only).",
        8
    )
{
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif


void CmdParserGEMM::parse ()
{
    CmdParserCommon::parse();

    // Test a small part of parameters for consistency
    // in this function. The major part of checks is placed in
    // validateParameters function. But to call it you need
    // further specialization on what OpenCL objects and their
    // capabilities are.

    if(arithmetic_float.isSet() && arithmetic_double.isSet())
    {
        throw CmdParser::Error(
            "Both float and double are chosen. "
            "Should be only one of them."
        );
    }

    if(!arithmetic_float.isSet() && !arithmetic_double.isSet())
    {
        throw CmdParser::Error(
            "Neither float nor double are chosen. "
            "One of them should be chosen."
        );
    }
}


size_t CmdParserGEMM::estimateMaxMatrixSize (
    OpenCLBasic& oclobjects,
    size_t size_of_element,
    size_t alignment
)
{
    cl_ulong max_alloc_size = 0;
    cl_int err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(max_alloc_size),
        &max_alloc_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    cl_ulong max_global_mem_size = 0;
    err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(max_global_mem_size),
        &max_global_mem_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    double max_matrix_size = sqrt(
        min(
            double(numeric_limits<size_t>::max()),
            min(double(max_alloc_size), double(max_global_mem_size)/3)
        ) / size_of_element
    );

    assert(alignment%size_of_element == 0);

    // the following is effect of a bit conservative
    // estimation of the overhead on a row alignment
    max_matrix_size -= alignment/size_of_element;

    assert(max_matrix_size < double(numeric_limits<size_t>::max()));

    return static_cast<size_t>(max_matrix_size);
}


void CmdParserGEMM::validateTile (
    const CmdOption<size_t>& tile_group,
    const CmdOption<size_t>& tile_size,
    size_t max_group_value
)
{
    validatePositiveness(tile_group);
    validatePositiveness(tile_size);

    tile_group.validate(
        size.getValue() % tile_group.getValue() == 0,
        "should divide matrix size without a remainder"
    );

    tile_size.validate(
        size.getValue() % tile_size.getValue() == 0,
        "should divide matrix size without a remainder"
    );

    tile_group.validate(
        tile_group.getValue() <= max_group_value,
        "too big value; should be <= " + to_str(max_group_value)
    );

    if(
        size.getValue() %
        (tile_group.getValue() * tile_size.getValue()) != 0
    )
    {
        throw CmdParser::Error(
            "Multiplication of " + tile_group.name() + " and " + tile_size.name() +
            " parameters should divide matrix size without a remainder."
        );
    }
}


void CmdParserGEMM::validateParameters (
    OpenCLBasic& oclobjects,
    OpenCLProgramOneKernel& executable,
    size_t size_of_element,
    size_t alignment
)
{
    validatePositiveness(size);

    size_t max_matrix_size =
        estimateMaxMatrixSize(oclobjects, size_of_element, alignment);

    size.validate(
        size.getValue() <= max_matrix_size,
        "requested value is too big; should be <= " + to_str(max_matrix_size)
    );

    iterations.validate(
        iterations.getValue() >= 0,
        "negative value is provided; should be positive or zero"
    );

    size_t max_work_item_sizes[3] = {0};
    deviceMaxWorkItemSizes(oclobjects.device, max_work_item_sizes);

    validateTile(tile_group_M, tile_size_M, max_work_item_sizes[0]);
    validateTile(tile_group_N, tile_size_N, max_work_item_sizes[1]);

    size_t work_group_size =
        tile_group_M.getValue() * tile_group_N.getValue();

    size_t max_device_work_group_size =
        deviceMaxWorkGroupSize(oclobjects.device);

    size_t max_kernel_work_group_size =
        kernelMaxWorkGroupSize(executable.kernel, oclobjects.device);

    size_t max_work_group_size =
        min(max_device_work_group_size, max_kernel_work_group_size);

    if(work_group_size > max_kernel_work_group_size)
    {
        throw CmdParser::Error(
            "Work group size required based on " +
            tile_group_M.name() + " and " + tile_group_N.name() +
            " is greater than allowed for this kernel and/or device. " +
            "Maximum possible value is " +
            to_str(max_kernel_work_group_size) + "."
        );
    }

    validatePositiveness(tile_size_K);

    tile_size_K.validate(
        size.getValue() % tile_size_K.getValue() == 0,
        "should divide matrix size without a remainder"
    );
}
