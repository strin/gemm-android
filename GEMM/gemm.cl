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


/*  The following macros should be defined in the build options:

    - T -- type of matrix elements and all calculations

    Tiles sizes and group sizes:

    - TILE_SIZE_M -- the number of elements of A processed in one WI
    - TILE_GROUP_M -- group/tile size along matrix A (NDRange dimension 0)
    - TILE_SIZE_N -- the number of elements of B processed in one WI
    - TILE_GROUP_N -- group/tile size along matrix B (NDRange dimension 1)
    - TILE_SIZE_K -- size of a tile along dot-product dimension

    There are two kernels: gemm_nt and gemm_nn; the difference is in B matrix format.
    Letters n and t are for column-major (non-transposed) and row-major matrix format
    (transposed) correspondingly.
*/


#ifdef SAMPLE_NEEDS_DOUBLE
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif


// C := alpha*A*B + beta*C
// A is in column-major form
// B is in row-major form (transposed; this is different from gemm_nn)
// C is in column-major form
__attribute__((reqd_work_group_size(TILE_GROUP_M, TILE_GROUP_N, 1)))
kernel void gemm_nt (
    global const T * restrict A,
    int lda,    // column stride in elements for matrix A
    global const T * restrict B,
    int ldb,    // row stride in elements for matrix B
    global T * restrict C,
    int ldc,    // column stride in elements for matrix C
    int k,        // number of columns/rows in a matrix
    T alpha,
    T beta
)
{
    // Indices for matrices A and B are calculated similarly
    // as they are in different formats (the first one is in
    // column-major form and the second one is in row-major) and
    // matrix multiplication involves "natural transpose" for
    // one of the matrix.

    int Aind = get_group_id(0)*TILE_GROUP_M*TILE_SIZE_M + get_local_id(0);
    int Bind = get_group_id(1)*TILE_GROUP_N*TILE_SIZE_N + get_local_id(1);
    int Cind = Aind + Bind*ldc;

    T c[TILE_SIZE_M*TILE_SIZE_N] = {(T)0};

    // main accumulation loop
    for(int l = 0; l < k; ++l)
    {
        for(int i = 0; i < TILE_SIZE_M; ++i)
            for(int j = 0; j < TILE_SIZE_N; ++j)
                c[i*TILE_SIZE_N + j] +=
                    A[Aind + i*TILE_GROUP_M] *
                    B[Bind + j*TILE_GROUP_N];
        Aind += lda;
        Bind += ldb;
    }

    // Store accumulated results from c to C with alpha and beta multiplication
    for(int i = 0; i < TILE_SIZE_M; ++i)
        for(int j = 0; j < TILE_SIZE_N; ++j)
        {
            int Ccur = Cind + i*TILE_GROUP_M + j*TILE_GROUP_N*ldc;
            C[Ccur] = alpha*c[i*TILE_SIZE_N + j] + beta*C[Ccur];
        }
}


// C := alpha*A*B + beta*C
// A is in column-major form
// B is in column-major form (this is different from gemm_nt)
// C is in column-major form
__attribute__((reqd_work_group_size(TILE_GROUP_M, TILE_GROUP_N, 1)))
kernel void gemm_nn (
    global const T * restrict A,
    int lda,    // column stride in elements for matrix A
    global const T * restrict B,
    int ldb,    // column stride in elements for matrix B
    global T * restrict C,
    int ldc,    // column stride in elements for matrix C
    int k,
    T alpha,
    T beta
)
{
    // Indices for matrices A and B are calculated differently
    // because they have the same format (both column-major) and
    // matrix multiplication involves "natural transpose" for
    // one of the matrix.

    int Aind = get_group_id(0)*TILE_GROUP_M*TILE_SIZE_M + get_local_id(0);
    int Bind = get_group_id(1)*TILE_GROUP_N*TILE_SIZE_N + get_local_id(1);
    int Cind = Aind + Bind*ldc;

    Bind *= ldb;    // matrix B is in column-major form

    T c[TILE_SIZE_M*TILE_SIZE_N] = {(T)0};

    // Main accumulation loop
    for(int l_block = 0; l_block < k; l_block += TILE_SIZE_K)
    {
        for(int i = 0; i < TILE_SIZE_M; ++i)
            for(int j = 0; j < TILE_SIZE_N; ++j)
                for(int l = 0; l < TILE_SIZE_K; ++l)
                    c[i*TILE_SIZE_N + j] +=
                        A[Aind + l*lda + i*TILE_GROUP_M] *
                        B[Bind + l + j*ldb*TILE_GROUP_N];
        Aind += lda*TILE_SIZE_K;
        Bind += TILE_SIZE_K;
    }

    // Store accumulated results from c to C with alpha and beta multiplication
    for(int i = 0; i < TILE_SIZE_M; ++i)
        for(int j = 0; j < TILE_SIZE_N; ++j)
        {
            int Ccur = Cind + i*TILE_GROUP_M + j*TILE_GROUP_N*ldc;
            C[Ccur] = alpha*c[i*TILE_SIZE_N + j] + beta*C[Ccur];
        }
}
