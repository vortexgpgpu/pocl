/* libm entry points the Vortex device needs in the kernel bitcode library.
 *
 * Clang lowers some __builtin_* math calls (used by the generic OpenCL
 * builtin sources) into plain libm calls rather than inline code when the
 * target has no instruction for them. Those symbols have to resolve inside
 * kernel-riscv32.bc, so they are implemented here in terms of bit
 * manipulation only -- never via the corresponding __builtin_, which would
 * lower straight back into a call to this function.
 */

typedef unsigned int uint32_t;

static inline __attribute__((always_inline)) uint32_t f2b(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;
  return v.u;
}

static inline __attribute__((always_inline)) float b2f(uint32_t u) {
  union { float f; uint32_t u; } v;
  v.u = u;
  return v.f;
}

/* IEEE-754 nextafter for binary32: step one representable value from x
 * towards y. Magnitude ordering of the IEEE bit pattern makes this a simple
 * increment/decrement of the integer encoding. */
float nextafterf(float x, float y) {
  uint32_t ux = f2b(x), uy = f2b(y);

  /* NaN operand propagates. */
  if (((ux & 0x7f800000u) == 0x7f800000u && (ux & 0x007fffffu) != 0u) ||
      ((uy & 0x7f800000u) == 0x7f800000u && (uy & 0x007fffffu) != 0u))
    return x + y;

  if (x == y)
    return y; /* also covers +0 vs -0: returns y, per the spec */

  if (x == 0.0f) {
    /* Smallest subnormal, carrying the sign of the direction. */
    return b2f((uy & 0x80000000u) | 1u);
  }

  /* Step the magnitude: away from zero when moving away from zero. */
  if ((x < y) == (x > 0.0f))
    ux += 1u; /* towards larger magnitude */
  else
    ux -= 1u; /* towards smaller magnitude */

  return b2f(ux);
}

/* The double form is only reachable when the device advertises fp64; keep it
 * available so a doubled-up generic source still links. */
#ifdef cl_khr_fp64
typedef unsigned long long uint64_t;

static inline __attribute__((always_inline)) uint64_t d2b(double d) {
  union { double d; uint64_t u; } v;
  v.d = d;
  return v.u;
}

static inline __attribute__((always_inline)) double b2d(uint64_t u) {
  union { double d; uint64_t u; } v;
  v.u = u;
  return v.d;
}

double nextafter(double x, double y) {
  uint64_t ux = d2b(x), uy = d2b(y);

  if (((ux & 0x7ff0000000000000ull) == 0x7ff0000000000000ull &&
       (ux & 0x000fffffffffffffull) != 0ull) ||
      ((uy & 0x7ff0000000000000ull) == 0x7ff0000000000000ull &&
       (uy & 0x000fffffffffffffull) != 0ull))
    return x + y;

  if (x == y)
    return y;

  if (x == 0.0) {
    return b2d((uy & 0x8000000000000000ull) | 1ull);
  }

  if ((x < y) == (x > 0.0))
    ux += 1ull;
  else
    ux -= 1ull;

  return b2d(ux);
}
#endif
