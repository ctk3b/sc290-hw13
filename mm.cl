__kernel void mm(const unsigned int N,
                const unsigned int M,
                const unsigned int K,
                __global float* A,
                __global float* B,
                __global float* C)
{
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    //printf("%d %d\n", row, col);
    unsigned int k;
    
    float sum = 0;
    for(k = 0; k < K; k++){
        sum += A[K*row + k] * B[M*k + col]; 
    }
    C[M*row + col] = sum;
}
