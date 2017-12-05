#ifndef FP16_CROP
#define FP16_CROP

#include <immintrin.h>
#include <cassert>
#include "io/logging.h"

/* we only mask 11 out of 22 bit to keep 11bit mantissa and don't deal with rounding */
#define EDGE_FP16_FP32_MANT_MASK 0xFFFFF800

// scales the stress (3D) before and after cropping
#define EDGE_FP16_STRESS_SCA 1.0E8

typedef union edge_intfloat {
  unsigned int ui;
  float f;
} edge_intfloat;

static void edge_fp16_crop( float* io_buf, size_t length, int mantissa_only) {
  if (mantissa_only) {
    edge_intfloat value;
    size_t i = 0;
    value.ui = 0;

    for ( i = 0 ; i < length ; ++i) {
      value.f = io_buf[i];
      value.ui = (value.ui & EDGE_FP16_FP32_MANT_MASK);
      io_buf[i] = value.f;
    }   
  } else {
    size_t loop_trips = (length/16)*16;
    size_t i = 0;
  
    for ( i = 0 ; i < loop_trips ; i+= 16 ) {
      __m512 fp32 = _mm512_load_ps( io_buf+i );
      __m256i fp16 = _mm512_cvt_roundps_ph( fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
      fp32 = _mm512_cvt_roundph_ps( fp16, _MM_FROUND_NO_EXC );
      _mm512_store_ps( io_buf+i, fp32 );
    }
    if (loop_trips < length) {
      size_t remain = length-loop_trips;
      unsigned short int_mask = ((0xFFFF) << remain);
      __mmask16 mask = _mm512_int2mask((unsigned short)(~(int_mask)) );
      __m512 zero = _mm512_setzero_ps();
      __m512 fp32 = _mm512_mask_load_ps( zero, mask, io_buf+loop_trips );
      __m256i fp16 = _mm512_cvt_roundps_ph( fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
      fp32 = _mm512_cvt_roundph_ps( fp16, _MM_FROUND_NO_EXC );
      _mm512_mask_store_ps( io_buf+loop_trips, mask, fp32 );
    }
  }
}

#endif
