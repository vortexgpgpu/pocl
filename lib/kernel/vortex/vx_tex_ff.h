/* Vortex fixed-function TEX intrinsics for the OpenCL image builtins.
 *
 * The canonical intrinsic (sw/kernel/include/vx_graphics.h) lives in a C++
 * namespace gated on __VORTEX__ and is not usable from OpenCL-C. It is only a
 * raw CUSTOM1 `.insn` encoding, though, so we re-declare it as an OpenCL-C
 * static inline with the identical encoding. This needs no backend support
 * (the PoCL kernel bitcode is plain rv32imaf, no +xvortex): a `.insn` emits
 * the raw instruction word directly. The FF op traps as illegal unless the
 * device was configured with VX_CFG_EXT_TEX_ENABLE, which is exactly the
 * Tier-A eligibility gate — a non-TEX build never reaches this path (the host
 * leaves _tex_stage < 0, see pocl-vortex.c), so it is never emitted there.
 *
 * ABI (VX_types.toml / vx_graphics.h):
 *   RISCV_CUSTOM1 = 0x2B
 *   vx_tex : .insn r4 0x2B, funct3=5, funct2=stage, rd=texel, rs1=u, rs2=v,
 *            rs3=lod
 * The unit takes its operands in registers and returns the texel in rd — no
 * window staging and no completion handle. `stage` rides funct2 and so must
 * be a compile-time constant; the kernel therefore branches on the (runtime)
 * bound stage into a fixed-stage arm. u/v are s32 fixed-point with
 * TEX_FXD_FRAC = 23 fractional bits, normalized to [0,1); lod is an integer
 * mip level (0 — OpenCL images are single-LOD in Phase 1). The sampled texel
 * returns as a packed 8-bit ARGB word.
 */

#ifndef VX_TEX_FF_H
#define VX_TEX_FF_H

/* TEX_FXD_FRAC = VX_TEX_DIM_BITS(15) + VX_TEX_SUBPIXEL_BITS(8) = 23. */
#define VX_TEXFF_FXD_FRAC 23

/* Stage 0 / stage 1 texture samples (stage is funct2: compile-time). */
_CL_READNONE static uint vx_texff_sample_stage0(uint u_fx, uint v_fx) {
  uint texel;
  __asm__ volatile (".insn r4 %1, 5, %2, %0, %3, %4, %5"
      : "=r"(texel)
      : "i"(0x2B), "i"(0), "r"(u_fx), "r"(v_fx), "r"((uint)0));
  return texel;
}
_CL_READNONE static uint vx_texff_sample_stage1(uint u_fx, uint v_fx) {
  uint texel;
  __asm__ volatile (".insn r4 %1, 5, %2, %0, %3, %4, %5"
      : "=r"(texel)
      : "i"(0x2B), "i"(1), "r"(u_fx), "r"(v_fx), "r"((uint)0));
  return texel;
}

/* One FF sample on the bound stage. `u_fx`/`v_fx` are TEX_FXD_FRAC fixed-point
 * normalized coords; the texel is the packed ARGB8 word. */
_CL_READNONE static uint vx_texff_sample(int stage, uint u_fx, uint v_fx) {
  return (stage == 0) ? vx_texff_sample_stage0(u_fx, v_fx)
                      : vx_texff_sample_stage1(u_fx, v_fx);
}

/* Convert a normalized coordinate (possibly outside [0,1) for repeat/mirror)
 * to the s32 fixed-point the FF unit expects. */
_CL_READNONE static uint vx_texff_fixed(float norm) {
  return (uint)(int)(norm * (float)(1 << VX_TEXFF_FXD_FRAC));
}

#endif /* VX_TEX_FF_H */
