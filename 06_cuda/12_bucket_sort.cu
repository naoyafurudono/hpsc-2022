#include <cstdio>
#include <cstdlib>

__global__ void backet_sort(int *key, int *bucket, int *offset, int n, int range, int*b)
{
  int i = threadIdx.x;
  bucket[i] = 0;
  offset[i] = 0;
  __syncthreads();

  if(i==0)
  {
    for (int j=0; j<n; j++)
      bucket[key[j]]++;
  }
  __syncthreads();

  // scan
  if (i!=0) offset[i] = bucket[i-1];
  for (int j=1; j<range; j<<=1)
  {
    b[i] = offset[i];
    __syncthreads();
    if(i>=j) offset[i] += b[i-j];
    __syncthreads();
  }

  for (int j=0; bucket[i]>0; bucket[i]--, ++j) {
    key[offset[i]+j] = i;
  }
}


int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset, *b ;
  cudaMallocManaged( &key, n*sizeof(int) );
  cudaMallocManaged( &bucket, range*sizeof(int) );
  cudaMallocManaged( &offset, range*sizeof(int) );
  cudaMallocManaged( &b, range*sizeof(int) );

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  backet_sort<<<1, range>>>(key, bucket, offset, n, range, b);
  cudaDeviceSynchronize();
  
  cudaFree(b);
  cudaFree(offset);
  cudaFree(bucket);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
}
