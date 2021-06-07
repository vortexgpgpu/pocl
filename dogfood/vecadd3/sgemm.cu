__global__ void sgemm (float *A, float *B, float *C, int N)
{
  // Thread identifiers
  const int r = blockIdx.x; // Row ID
  const int c = blockIdx.y; // Col ID

  // Compute a single element (loop a K)
  float acc = 0.0f;
  for (int k = 0; k < N; k++) {
    acc += A[k * N + r] * B[c * N + k];
  }

  // Store the result
  C[c * N + r] = acc;
}
