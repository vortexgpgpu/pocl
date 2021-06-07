__global__ void vecadd(float *a, float *b, float* c) 
{
    // Get our global thread ID
    int id = blockIdx.x;
 
    // Make sure we do not go out of bounds
    c[id] = a[id] + b[id];
}
