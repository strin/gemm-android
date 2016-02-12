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

#ifndef _INTEL_OPENCL_SAMPLE_GEMM_CMDOPTIONS_HPP_
#define _INTEL_OPENCL_SAMPLE_GEMM_CMDOPTIONS_HPP_

#include "oclobject.hpp"
#include "cmdparser.hpp"


// All command-line options for GEMM sample
class CmdParserGEMM : public CmdParserCommon
{
public:
    // For these options description, please refer to the constructor definition.

    CmdOption<size_t> size;
    CmdOption<int> iterations;

    CmdOption<string> arithmetic;
        CmdEnum<string> arithmetic_float;
        CmdEnum<string> arithmetic_double;

    CmdOption<string> kernel;
        CmdEnum<string> kernel_nt;
        CmdEnum<string> kernel_nn;

    CmdOption<bool> validation;

    CmdOption<size_t> tile_size_M;
    CmdOption<size_t> tile_group_M;

    CmdOption<size_t> tile_size_N;
    CmdOption<size_t> tile_group_N;

    CmdOption<size_t> tile_size_K;

    CmdParserGEMM (int argc, const char** argv);
    virtual void parse ();

    // Check if all parameters have correct and consistent
    // values based on device capabilities.
    void validateParameters (
        OpenCLBasic& oclobjects,
        OpenCLProgramOneKernel& executable,
        size_t size_of_element, // size of one element of matrix in bytes
        size_t alignment    // alignment requirements in bytes
    );

private:

    template <typename T>
    void validatePositiveness (const CmdOption<T>& parameter)
    {
        parameter.validate(
            parameter.getValue() > 0,
            "negative or zero value is provided; should be positive"
        );
    }

    size_t estimateMaxMatrixSize (
        OpenCLBasic& oclobjects,
        size_t size_of_element,
        size_t alignment
    );

    void validateTile (
        const CmdOption<size_t>& tile_group,
        const CmdOption<size_t>& tile_size,
        size_t max_group_value
    );
};


#endif  // end of the include guard
