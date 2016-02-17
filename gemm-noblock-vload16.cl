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

    float16 sum = (float16)0.0f;
  
    A += i * k;
    B += j * k;

    for (int l = 0; l < k; l += 16)
    {
        float16 x = vload16(0, A);
        float16 y = vload16(0, B);

        sum += x * y;

        A += 16; // this is faster than A[i* k + l]. 11 GFlops vs. 10 GFlops.
        B += 16;
    }

    C[i * k + j] = sum.s0 + sum.s1 + sum.s2 + sum.s3
                                  + sum.s4 + sum.s5 + sum.s6 + sum.s7
                                  + sum.s8 + sum.s9 + sum.sa + sum.sb
                                      + sum.sc + sum.sd + sum.se + sum.sf;
                    ;
}

