#define dot16(a,b) \
    (dot(a.lo.lo, b.lo.lo) + dot(a.lo.hi, b.lo.hi)  \
    + dot(a.hi.lo, b.hi.lo) + dot(a.hi.hi, b.hi.hi))


__kernel void gemm_tn (
    __global const T * restrict A,
    int lda,    // column stride in elements for matrix A
    __global const T * restrict B,
    int ldb,    // row stride in elements for matrix B
    __global T * restrict C,
    int ldc,    // column stride in elements for matrix C
    int k        // number of columns/rows in a matrix
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    float sum = 0.0f;
  
    A += i * k;
    B += j * k;

    for (int l = 0; l < k; l += 16)
    {
        sum += dot16(vload16(0, A), vload16(0, B));

        A += 16; // this is faster than A[i* k + l]. 11 GFlops vs. 10 GFlops.
        B += 16;
    }

    C[i * k + j] = sum;
}

