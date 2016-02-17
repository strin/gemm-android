__kernel void gemm_tn (
    __global const T * restrict A,
    int lda,    // column stride in elements for matrix A
    __global const T * restrict B,
    int ldb,    // row stride in elements for matrix B
    __global T * restrict C,
    int ldc,    // column stride in elements for matrix C
    int k        // number of columns/rows in a matrix
) {
    const int i = get_global_id(0) * 4;
    const int j = get_global_id(1) * 4;
    
    float16 sum = (float16)0.0f;

    for (int l = 0; l < k; l += 4)
    {
        float8 a01 = (float8) (vload4(0, &A[i * k]), vload4(0, &A[(i+1) * k]));
        float8 a23 = (float8) (vload4(0, &A[(i+2) * k]), vload4(0, &A[(i+3) * k]));
        float8 b01 = (float8) (vload4(0, &B[j * k]), vload4(0, &B[(j+1) * k]));
        float8 b23 = (float8) (vload4(0, &B[(j+2) * k]), vload4(0, &B[(j+3) * k]));

        sum += (float16) (dot(a01.lo, b01.lo), dot(a01.lo, b01.hi), dot(a01.lo, b23.lo), dot(a01.lo, b23.hi),
                          dot(a01.hi, b01.lo), dot(a01.hi, b01.hi), dot(a01.hi, b23.lo), dot(a01.hi, b23.hi),
                          dot(a23.lo, b01.lo), dot(a23.lo, b01.hi), dot(a23.lo, b23.lo), dot(a23.lo, b23.hi),
                          dot(a23.hi, b01.lo), dot(a23.hi, b01.hi), dot(a23.hi, b23.lo), dot(a23.hi, b23.hi));
        
        A += 4; 
        B += 4;
    }

    vstore4(sum.lo.lo, 0, &C[i * k + j]);
    vstore4(sum.lo.hi, 0, &C[(i + 1) * k + j]);
    vstore4(sum.hi.lo, 0, &C[(i + 2) * k + j]);
    vstore4(sum.hi.hi, 0, &C[(i + 3) * k + j]);
}