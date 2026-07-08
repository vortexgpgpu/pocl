/* Vortex fixed-function TEX intrinsics for the OpenCL image builtins.
 *
 * The canonical intrinsics (sw/kernel/include/vx_graphics.h) live in a C++
 * namespace gated on __VORTEX__ and are not usable from OpenCL-C. They are only
 * raw CUSTOM1 `.insn` encodings, though, so we re-declare the two we need as
 * OpenCL-C static inlines with the identical encoding. This needs no backend
 * support (the PoCL kernel bitcode is plain rv32imaf, no +xvortex): a `.insn`
 * emits the raw instruction word directly. The FF op traps as illegal unless
 * the device was configured with VX_CFG_EXT_TEX_ENABLE, which is exactly the
 * Tier-A eligibility gate — a non-TEX build never reaches this path (the host
 * leaves _tex_stage < 0, see pocl-vortex.c), so it is never emitted there.
 *
 * ABI (VX_types.toml / vx_gfx_window.h / vx_graphics.h):
 *   RISCV_CUSTOM1 = 0x2B
 *   SETW : .insn r 0x2B, funct3=6, funct7=(slot<<2)|1, rd=x0, rs1=val, rs2=x0
 *   GETW+dep+barrier : .insn r 0x2B, funct3=6, funct7=(slot<<2)|3, rd, rs1=tok, rs2=x1
 *   vx_tex4_single   : .insn r 0x2B, funct3=5, funct7=(out_slot<<2)|(stage<<1),
 *                       rd=handle, rs1=lod, rs2=in_slot
 * `stage` and `out_slot` ride funct7 and so must be compile-time constants; the
 * kernel therefore branches on the (runtime) bound stage into a fixed-stage arm.
 * u/v are staged as s32 fixed-point with TEX_FXD_FRAC = 23 fractional bits,
 * normalized to [0,1); the sampled texel returns as a packed 8-bit ARGB word.
 */

#ifndef VX_TEX_FF_H
#define VX_TEX_FF_H

/* Window slot convention for the image path — u@0, v@1, texel@OUT. No RTU/raster
 * payload is in flight in a compute kernel, so the low slots are free (matches
 * tests/graphics/gfx_tex4). */
#define VX_TEXFF_IN_SLOT  0
#define VX_TEXFF_OUT_SLOT 4

/* TEX_FXD_FRAC = VX_TEX_DIM_BITS(15) + VX_TEX_SUBPIXEL_BITS(8) = 23. */
#define VX_TEXFF_FXD_FRAC 23

/* Stage 0 / stage 1 texture samples (out_slot=4, in_slot base=0). lod is 0
 * (OpenCL images are single-LOD in Phase 1). Returns the scoreboard handle. */
_CL_READNONE static uint vx_texff_sample_stage0(void) {
  uint h;
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, %4"
      : "=r"(h)
      : "i"(0x2B), "i"(((VX_TEXFF_OUT_SLOT) << 2) | ((0) << 1)),
        "r"((uint)0), "r"((uint)VX_TEXFF_IN_SLOT));
  return h;
}
_CL_READNONE static uint vx_texff_sample_stage1(void) {
  uint h;
  __asm__ volatile (".insn r %1, 5, %2, %0, %3, %4"
      : "=r"(h)
      : "i"(0x2B), "i"(((VX_TEXFF_OUT_SLOT) << 2) | ((1) << 1)),
        "r"((uint)0), "r"((uint)VX_TEXFF_IN_SLOT));
  return h;
}

/* Stage the u/v coordinates into the window (SETW slots 0 and 1). */
_CL_READNONE static void vx_texff_set_uv(uint u_fx, uint v_fx) {
  __asm__ volatile (".insn r %0, 6, %1, x0, %2, x0"
      :: "i"(0x2B), "i"(((VX_TEXFF_IN_SLOT) << 2) | 1), "r"(u_fx));
  __asm__ volatile (".insn r %0, 6, %1, x0, %2, x0"
      :: "i"(0x2B), "i"(((VX_TEXFF_IN_SLOT + 1) << 2) | 1), "r"(v_fx));
}

/* Read the sampled texel back (GETW slot 4, chained on the tex4 handle + a
 * memory barrier so the in-order read observes the FF unit's write). */
_CL_READNONE static uint vx_texff_get_texel(uint handle) {
  uint v;
  __asm__ volatile (".insn r %1, 6, %2, %0, %3, x1"
      : "=r"(v)
      : "i"(0x2B), "i"(((VX_TEXFF_OUT_SLOT) << 2) | 3), "r"(handle)
      : "memory");
  return v;
}

/* One FF sample: stage u/v, fire vx_tex4 on the bound stage, read the ARGB8
 * texel. `u_fx`/`v_fx` are TEX_FXD_FRAC fixed-point normalized coords. */
_CL_READNONE static uint vx_texff_sample(int stage, uint u_fx, uint v_fx) {
  vx_texff_set_uv(u_fx, v_fx);
  uint h = (stage == 0) ? vx_texff_sample_stage0() : vx_texff_sample_stage1();
  return vx_texff_get_texel(h);
}

/* Convert a normalized coordinate (possibly outside [0,1) for repeat/mirror)
 * to the s32 fixed-point the FF unit expects. */
_CL_READNONE static uint vx_texff_fixed(float norm) {
  return (uint)(int)(norm * (float)(1 << VX_TEXFF_FXD_FRAC));
}

#endif /* VX_TEX_FF_H */
