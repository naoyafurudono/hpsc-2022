#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

  void print(__m256 vec) {
    float b[8];
    _mm256_store_ps(b, vec);
    for(int i=0;i<8;i++)
	printf("%f ", b[i]);
    printf("\n");
  }
// 小数第二位にforを使った場合と比べて誤差が生じる。
// TODO 原因を見つける。
int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 zero = _mm256_setzero_ps();
  float buffer[N];
  for(int i=0; i<N; i++) {
    // rx  = x[i] - x[j];
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 xvec = _mm256_load_ps(x);
    __m256 rxvec = _mm256_sub_ps(xivec, xvec);

    // ry  = y[i] - y[j];
    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 yvec = _mm256_load_ps(y);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);

    // compute r*r and mask
    __m256 rtmp = _mm256_mul_ps(rxvec,rxvec);
    rtmp = _mm256_fmadd_ps(ryvec,ryvec,rtmp);
    __m256 mask = _mm256_cmp_ps(zero, rtmp, _CMP_NEQ_UQ);
    //print(rtmp);

    // compute 1/(r*r*r)
    __m256 rrcpvec = _mm256_rsqrt_ps(rtmp);
    __m256 rrcp3vec =_mm256_mul_ps(rrcpvec, rrcpvec);
    rrcp3vec =_mm256_mul_ps(rrcpvec, rrcp3vec);

    __m256 mvec = _mm256_load_ps(m);
    __m256 minus1 = _mm256_set1_ps(-1);

    __m256 xsubvec = _mm256_mul_ps(rxvec,mvec);
    xsubvec = _mm256_mul_ps(xsubvec, rrcp3vec);
    xsubvec = _mm256_mul_ps(xsubvec, minus1);
    xsubvec = _mm256_blendv_ps(zero, xsubvec, mask);
    // printf("xsubvec:\n");
    // print(xsubvec);
    // print(zero);

    __m256 bvec = _mm256_permute2f128_ps(xsubvec, xsubvec,1);
    bvec = _mm256_add_ps(bvec, xsubvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    _mm256_store_ps(buffer, bvec);
    fx[i] += buffer[0];

    __m256 ysubvec = _mm256_mul_ps(ryvec,mvec);
    ysubvec = _mm256_mul_ps(ysubvec, rrcp3vec);
    ysubvec = _mm256_mul_ps(ysubvec, minus1);
    ysubvec = _mm256_blendv_ps(zero, ysubvec, mask);

    bvec = _mm256_permute2f128_ps(ysubvec, ysubvec,1);
    bvec = _mm256_add_ps(bvec, ysubvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    bvec = _mm256_hadd_ps(bvec,bvec);
    _mm256_store_ps(buffer, bvec);
    fy[i] += buffer[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
 /* 
  printf("ans: \n");
  for(int i=0; i<N; i++) {
    fx[i] = fy[i] = 0;
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
*/
}
