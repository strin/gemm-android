#define dot8(a,b) \
    (dot(a.hi, b.hi) + dot(a.lo, b.lo))


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

    for (int l = 0; l < k; l += 8)
    {
        float8 x = vload8(0, A);
        float8 y = vload8(0, B);

        sum += dot8(x, y);

        A += 8; // this is faster than A[i* k + l]. 11 GFlops vs. 10 GFlops.
        B += 8;
    }

    C[i * k + j] = sum;
}

