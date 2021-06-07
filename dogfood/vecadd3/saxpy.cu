__global__ void saxpy(float *src, float *dst, float factor)
{
  int i = blockIdx.x;
  dst[i] += src[i] * factor;
}

