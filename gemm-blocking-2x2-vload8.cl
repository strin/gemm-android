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
    const int i = get_global_id(0) * 2;
    const int j = get_global_id(1) * 2;
    
    float4 ab = (float4)0.0f;

    for (int l = 0; l < k; l += 8)
    {
        float8 a0 = vload8(0, &A[i * k]);
        float8 a1 = vload8(0, &A[(i+1) * k]);
        float8 b0 = vload8(0, &B[j * k]);
        float8 b1 = vload8(0, &B[(j+1) * k]);

        ab += ( float4 ) ( dot8 (a0 , b0 ), dot8 (a0 , b1 ), dot8 (a1 , b0 ), dot8 (a1 , b1 ));
        
        A += 8; 
        B += 8;
    }

    vstore2(ab.s01, 0, &C[i * k + j]);
    vstore2(ab.s23, 0, &C[(i+1) * k + j]);
}