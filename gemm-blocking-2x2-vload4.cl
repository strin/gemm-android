#define DOT(a,b) \
    (a.S0 * b.S0 + a.S1 * b.S1 + a.S2 * b.S2 + a.S3 * b.S3 \
    +a.S4 * b.S4 + a.S5 * b.S5 + a.S6 * b.S6 + a.S7 * b.S7) 

#define SUM(a) \
    (a.S0 + a.S1 + a.S2 + a.S3 + a.S4 + a.S5 + a.S6 + a.S7)

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

    for (int l = 0; l < k; l += 4)
    {
        float4 a0 = vload4(0, &A[i * k]);
        float4 a1 = vload4(0, &A[(i+1) * k]);
        float4 b0 = vload4(0, &B[j * k]);
        float4 b1 = vload4(0, &B[(j+1) * k]);

        ab += ( float4 ) ( dot (a0 , b0 ), dot (a0 , b1 ), dot (a1 , b0 ), dot (a1 , b1 ));
        
        A += 4; 
        B += 4;
    }

    /*for(int ib = 0; ib < 2; ib++) {
        for(int jb = 0; jb < 2; jb++) {
            C[(i+ib) * k + (j+jb)] = SUM(sum[ib][jb]);
        }
    }*/
    vstore2(ab.s01, 0, &C[i * k + j]);
    vstore2(ab.s23, 0, &C[(i+1) * k + j]);
}