/** based on SimdAvx2Integral.cpp from the Simd library http://ermig1979.github.io/Simd
input: a 3d uint8 array of size [X,Y,Z]
output: a 3d uint32 array of size [X+1,Y+1,Z+1]
doesn't modify the first index of each axis, the rest are set to the cumulative sum of the input array
that is, output[x+1,y+1,z+1] = sum_0^x (sum_0^y (sum_0^z input[x,y,z]) )
This is called an integral image by some, a summed-area table by others.
This version uses Intel's AVX2 commands, requires a fairly recent (2013ish) Intel processor

compile with
gcc -O3 -mavx -mavx2 -Wall -Wfatal-errors -shared -fPIC -o cumsum3d.so cumsum3d.c
*/
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#define SIMD_CHAR_AS_LONGLONG(a) (((long long)a) & 0xFF)
#define SIMD_INT_AS_LONGLONG(a) (((long long)a) & 0xFFFFFFFF)
#define SIMD_LL_SET1_EPI8(a) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(a) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 16) | (SIMD_CHAR_AS_LONGLONG(a) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 32) | (SIMD_CHAR_AS_LONGLONG(a) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(a) << 48) | (SIMD_CHAR_AS_LONGLONG(a) << 56)
#define SIMD_LL_SETR_EPI8(a, b, c, d, e, f, g, h) \
    SIMD_CHAR_AS_LONGLONG(a) | (SIMD_CHAR_AS_LONGLONG(b) << 8) | \
    (SIMD_CHAR_AS_LONGLONG(c) << 16) | (SIMD_CHAR_AS_LONGLONG(d) << 24) | \
    (SIMD_CHAR_AS_LONGLONG(e) << 32) | (SIMD_CHAR_AS_LONGLONG(f) << 40) | \
    (SIMD_CHAR_AS_LONGLONG(g) << 48) | (SIMD_CHAR_AS_LONGLONG(h) << 56)
#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)
#define SIMD_MM256_SET1_EPI8(a) \
    {SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a), \
    SIMD_LL_SET1_EPI8(a), SIMD_LL_SET1_EPI8(a)}
#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_LL_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7), SIMD_LL_SETR_EPI8(a8, a9, aa, ab, ac, ad, ae, af), \
    SIMD_LL_SETR_EPI8(b0, b1, b2, b3, b4, b5, b6, b7), SIMD_LL_SETR_EPI8(b8, b9, ba, bb, bc, bd, be, bf)}
#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3), \
    SIMD_LL_SET2_EPI32(a4, a5), SIMD_LL_SET2_EPI32(a6, a7)}

const __m256i MASK = SIMD_MM256_SETR_EPI8(
            0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00);
const __m256i PACK = SIMD_MM256_SETR_EPI32(0,2,4,6,1,3,5,7);
const __m256i ZERO = SIMD_MM256_SET1_EPI8(0);
const size_t X = 72;
const size_t Y = 72;
const size_t Z = 20;

void cumsum3d(const void * srcin, void * sumin)
{
    uint8_t* src = (uint8_t*) srcin;
    uint32_t* sum = (uint32_t*) sumin;
    size_t Zp = Z + 1;
    size_t YpZp = (Y+1)*(Z+1);
    sum += 1; // skip all-zeros section

    for(size_t x = 0; x < X; x++){
        for(size_t y = 0; y < Y; y++){
            __m256i row_sums = ZERO;
            for(size_t z = 0; z < Z; z += 4){
                __m256i _src = _mm256_and_si256(_mm256_set1_epi32(*(uint32_t*)(src+z)), MASK);
                row_sums = _mm256_add_epi32(row_sums, _mm256_sad_epu8(_src, ZERO));
                __m128i curr_row_sums = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(row_sums, PACK));
                curr_row_sums = _mm_add_epi32(curr_row_sums, _mm_loadu_si128((__m128i*)(sum+z + Zp)));
                curr_row_sums = _mm_add_epi32(curr_row_sums, _mm_loadu_si128((__m128i*)(sum+z + YpZp)));
                curr_row_sums = _mm_sub_epi32(curr_row_sums, _mm_loadu_si128((__m128i*)(sum+z)));
                _mm_storeu_si128((__m128i*)(sum+z+Zp+YpZp), curr_row_sums);
                row_sums = _mm256_permute4x64_epi64(row_sums, 0xFF);
            }
            src += Z;
            sum += Zp;
        }
        sum += Zp;
    }
}
