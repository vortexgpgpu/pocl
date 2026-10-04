/* Single-precision exp in float arithmetic (libclc algorithm). The generic
 * builtin lowers to libm expf, which evaluates in double precision and on a
 * device without the D extension costs thousands of soft-float instructions.
 *
 * Cody-Waite reduction by ln2, rational approximation on [-ln2/2, ln2/2], then
 * scaling by 2^p through the exponent field. The device reports CL_FP_DENORM,
 * so results below the smallest normal are scaled in two steps rather than
 * flushed to zero. */

#include "../templates.h"

static float vx_expf(float x) {
  const float ln2HI = 0x1.62e300p-1f;
  const float ln2LO = 0x1.2fefa2p-17f;
  const float invln2 = 0x1.715476p+0f;

  float fhalF = x < 0.0f ? -0.5f : 0.5f;
  int p = (int)fma(x, invln2, fhalF);
  float fp = (float)p;
  float hi = fma(fp, -ln2HI, x);
  float lo = -fp * ln2LO;

  float t = hi + lo;
  float tt = t * t;
  float v = fma(tt,
                -fma(tt,
                     fma(tt,
                         fma(tt,
                             fma(tt, 0x1.637698p-25f, -0x1.bbd41cp-20f),
                             0x1.1566aap-14f),
                         -0x1.6c16c2p-9f),
                     0x1.555556p-3f),
                t);

  float y = 1.0f - (((-lo) - (t * v) / (2.0f - v)) - hi);

  int subnormal = p < -125;
  int pn = subnormal ? p + 64 : p;
  float scale = subnormal ? 0x1p-64f : 1.0f;
  float r = as_float(as_int(y) + (pn << 23)) * scale;

  const float ulim = 0x1.62e430p+6f;  /* ln(FLT_MAX) */
  const float llim = -0x1.9fe368p+6f; /* below this exp(x) rounds to +0 */
  r = x < llim ? 0.0f : r;
  r = x < ulim ? r : as_float(0x7f800000);
  return isnan(x) ? x : r;
}

float _CL_OVERLOADABLE exp(float a) { return vx_expf(a); }
IMPLEMENT_BUILTIN_V_V(exp, float2, NAME1_2(vx_expf))
IMPLEMENT_BUILTIN_V_V(exp, float3, NAME1_3(vx_expf))
IMPLEMENT_BUILTIN_V_V(exp, float4, NAME1_4(vx_expf))
IMPLEMENT_BUILTIN_V_V(exp, float8, NAME1_8(vx_expf))
IMPLEMENT_BUILTIN_V_V(exp, float16, NAME1_16(vx_expf))

__IF_FP16(
half _CL_OVERLOADABLE exp(half a) { return (half)vx_expf((float)a); }
IMPLEMENT_BUILTIN_V_V(exp, half2, NAME1_2(exp))
IMPLEMENT_BUILTIN_V_V(exp, half3, NAME1_3(exp))
IMPLEMENT_BUILTIN_V_V(exp, half4, NAME1_4(exp))
IMPLEMENT_BUILTIN_V_V(exp, half8, NAME1_8(exp))
IMPLEMENT_BUILTIN_V_V(exp, half16, NAME1_16(exp)))

__IF_FP64(
double _CL_OVERLOADABLE exp(double a) { return __builtin_exp(a); }
IMPLEMENT_BUILTIN_V_V(exp, double2, NAME1_2(__builtin_exp))
IMPLEMENT_BUILTIN_V_V(exp, double3, NAME1_3(__builtin_exp))
IMPLEMENT_BUILTIN_V_V(exp, double4, NAME1_4(__builtin_exp))
IMPLEMENT_BUILTIN_V_V(exp, double8, NAME1_8(__builtin_exp))
IMPLEMENT_BUILTIN_V_V(exp, double16, NAME1_16(__builtin_exp)))
