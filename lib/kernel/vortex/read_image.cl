/* OpenCL image read builtins — Vortex scalar implementation.
 *
 * The generic pocl read_image.cl operates on OpenCL vector types (uint4/float4)
 * with genuine vector arithmetic. The Vortex LLVM backend (+xvortex) has no
 * fixed-length vector datapath, so vector arithmetic fails to select. This
 * implementation keeps every computation scalar and only assembles the vector
 * result type at the return boundary (which lowers to scalar inserts), so it
 * compiles for the scalar SIMT target while remaining bit-faithful to the
 * OpenCL image conversion rules. 2D images, the CL 1.2 required format set,
 * nearest/linear filtering and all address modes are supported; other image
 * dimensionalities fall through undefined (unused by the current tests).
 */

#include "../templates.h"
#include "vx_tex_ff.h"

/* Channel data types and orders reuse the CLK_* values from opencl-c-base.h,
 * which are numerically equal to the CL_* enums stored in dev_image_t. */

// --- half -> float (scalar) ---------------------------------------------------
_CL_READNONE static float vx_half_to_float(ushort h) {
  uint sign = (uint)(h & 0x8000u) << 16;
  uint exp  = (h >> 10) & 0x1fu;
  uint man  = h & 0x3ffu;
  uint bits;
  if (exp == 0) {
    if (man == 0) { bits = sign; }
    else {
      // subnormal: normalize
      int e = -1;
      do { man <<= 1; e++; } while ((man & 0x400u) == 0);
      man &= 0x3ffu;
      bits = sign | ((uint)(127 - 15 - e) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  return as_float(bits);
}

// --- read one raw channel scalar as float, per data type ----------------------
_CL_READNONE static float vx_chan_to_f(global const char* p, int ctype, int idx) {
  switch (ctype) {
  case CLK_UNORM_INT8:   return ((global const uchar*)p)[idx] * (1.0f / 255.0f);
  case CLK_SNORM_INT8:   { int v = ((global const char*)p)[idx];
                           return fmax(v * (1.0f / 127.0f), -1.0f); }
  case CLK_UNORM_INT16:  return ((global const ushort*)p)[idx] * (1.0f / 65535.0f);
  case CLK_SNORM_INT16:  { int v = ((global const short*)p)[idx];
                           return fmax(v * (1.0f / 32767.0f), -1.0f); }
  case CLK_HALF_FLOAT:   return vx_half_to_float(((global const ushort*)p)[idx]);
  case CLK_FLOAT:        return ((global const float*)p)[idx];
  default:               return ((global const uchar*)p)[idx] * (1.0f / 255.0f);
  }
}

_CL_READNONE static int vx_chan_to_i(global const char* p, int ctype, int idx) {
  switch (ctype) {
  case CLK_SIGNED_INT8:  return ((global const char*)p)[idx];
  case CLK_SIGNED_INT16: return ((global const short*)p)[idx];
  case CLK_SIGNED_INT32: return ((global const int*)p)[idx];
  default:               return ((global const char*)p)[idx];
  }
}

_CL_READNONE static uint vx_chan_to_ui(global const char* p, int ctype, int idx) {
  switch (ctype) {
  case CLK_UNSIGNED_INT8:  return ((global const uchar*)p)[idx];
  case CLK_UNSIGNED_INT16: return ((global const ushort*)p)[idx];
  case CLK_UNSIGNED_INT32: return ((global const uint*)p)[idx];
  default:                 return ((global const uchar*)p)[idx];
  }
}

_CL_READNONE static int vx_elem_size(int ctype) {
  switch (ctype) {
  case CLK_UNORM_INT8: case CLK_SNORM_INT8:
  case CLK_SIGNED_INT8: case CLK_UNSIGNED_INT8:   return 1;
  case CLK_UNORM_INT16: case CLK_SNORM_INT16:
  case CLK_SIGNED_INT16: case CLK_UNSIGNED_INT16:
  case CLK_HALF_FLOAT:                            return 2;
  default:                                        return 4;   // *_INT32 / FLOAT
  }
}

_CL_READNONE static int vx_num_chan(int order) {
  switch (order) {
  case CLK_R: case CLK_A: case CLK_INTENSITY: case CLK_LUMINANCE: return 1;
  case CLK_RG: case CLK_RA:                                       return 2;
  case CLK_RGB:                                                   return 3;
  default:                                                        return 4; // RGBA/BGRA/ARGB
  }
}

// Byte offset of texel (x,y) within the image data.
_CL_READNONE static size_t vx_texel_off(global dev_image_t* img, int x, int y,
                                        int nchan, int esize) {
  return (size_t)y * img->_row_pitch + (size_t)x * nchan * esize;
}

// --- addressing (returns clamped/wrapped integer coord, or -1 for border) -----
_CL_READNONE static int vx_wrap(int c, int size, int addr) {
  switch (addr) {
  case CLK_ADDRESS_REPEAT: { int m = c % size; return m < 0 ? m + size : m; }
  case CLK_ADDRESS_MIRRORED_REPEAT: {
    int p = size ? (c % (2 * size)) : 0; if (p < 0) p += 2 * size;
    return p < size ? p : (2 * size - 1 - p);
  }
  case CLK_ADDRESS_CLAMP:              // clamp to border
    return (c < 0 || c >= size) ? -1 : c;
  default:                             // CLAMP_TO_EDGE / NONE
    return c < 0 ? 0 : (c >= size ? size - 1 : c);
  }
}

// Map the four RGBA lanes for a given channel order onto stored-channel indices;
// missing channels default to 0 (rgb) / 1 (a).
_CL_READNONE static void vx_load_rgba_f(global dev_image_t* img, int x, int y,
                                        float* r, float* g, float* b, float* a) {
  int order = img->_order, ctype = img->_data_type;
  int nchan = vx_num_chan(order), esize = vx_elem_size(ctype);
  global const char* p = (global const char*)(size_t)img->_data + vx_texel_off(img, x, y, nchan, esize);
  float c0 = vx_chan_to_f(p, ctype, 0);
  float c1 = nchan > 1 ? vx_chan_to_f(p, ctype, 1) : 0.0f;
  float c2 = nchan > 2 ? vx_chan_to_f(p, ctype, 2) : 0.0f;
  float c3 = nchan > 3 ? vx_chan_to_f(p, ctype, 3) : 0.0f;
  switch (order) {
  case CLK_A:         *r = 0; *g = 0; *b = 0; *a = c0; return;
  case CLK_R:         *r = c0; *g = 0; *b = 0; *a = 1; return;
  case CLK_INTENSITY: *r = c0; *g = c0; *b = c0; *a = c0; return;
  case CLK_LUMINANCE: *r = c0; *g = c0; *b = c0; *a = 1; return;
  case CLK_RG:        *r = c0; *g = c1; *b = 0; *a = 1; return;
  case CLK_RA:        *r = c0; *g = 0; *b = 0; *a = c1; return;
  case CLK_RGB:       *r = c0; *g = c1; *b = c2; *a = 1; return;
  case CLK_BGRA:      *r = c2; *g = c1; *b = c0; *a = c3; return;
  case CLK_ARGB:      *r = c1; *g = c2; *b = c3; *a = c0; return;
  default:            *r = c0; *g = c1; *b = c2; *a = c3; return; // RGBA
  }
}

// ============================ Tier A — fixed-function TEX =====================
// Sample the FF TEX unit (Tier A) and convert the packed 8-bit ARGB texel to a
// float4 in the image's channel order. The host bound the stage (VX_TEX_FORMAT
// A8R8G8B8) with this image's base/dims/filter/wrap, so the FF decode reads the
// stored bytes as (b,g,r,a) = (byte0,byte1,byte2,byte3). CL_BGRA stores B,G,R,A
// (direct); CL_RGBA stores R,G,B,A (swap r<->b). u/v are normalized [0,1).
_CL_READNONE static float4 vx_read_imagef_2d_ff(global dev_image_t* img, int stage,
                                                float nu, float nv) {
  uint texel = vx_texff_sample(stage, vx_texff_fixed(nu), vx_texff_fixed(nv));
  float wa = ((texel >> 24) & 0xffu) * (1.0f / 255.0f);
  float wr = ((texel >> 16) & 0xffu) * (1.0f / 255.0f);
  float wg = ((texel >>  8) & 0xffu) * (1.0f / 255.0f);
  float wb = ((texel      ) & 0xffu) * (1.0f / 255.0f);
  if (img->_order == CLK_RGBA) return (float4)(wb, wg, wr, wa);
  return (float4)(wr, wg, wb, wa); // CLK_BGRA
}

// Tier-A gate: the image was FF-bound this launch AND the runtime sampler is the
// exact one the host bound the stage for (so filter/wrap/normalization match the
// programmed DCRs). Anything else falls through to the software sampler.
_CL_READNONE static int vx_tex_ff_eligible(global dev_image_t* img, int smp) {
  return img->_tex_stage >= 0 && smp == img->_tex_sampler;
}

// ============================ read_imagef ====================================
_CL_READNONE static float4 vx_read_imagef_2d(global dev_image_t* img, int smp,
                                             float u, float v) {
  if (vx_tex_ff_eligible(img, smp)) {
    int norm = smp & 1;
    float nu = (norm == CLK_NORMALIZED_COORDS_TRUE) ? u : u / img->_width;
    float nv = (norm == CLK_NORMALIZED_COORDS_TRUE) ? v : v / img->_height;
    return vx_read_imagef_2d_ff(img, img->_tex_stage, nu, nv);
  }
  int w = img->_width, h = img->_height;
  int addr = smp & 0x0e, filt = smp & 0x30, norm = smp & 1;
  float fx = u, fy = v;
  if (norm == CLK_NORMALIZED_COORDS_TRUE) { fx = u * w; fy = v * h; }

  if (filt == CLK_FILTER_LINEAR) {
    float sx = fx - 0.5f, sy = fy - 0.5f;
    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    float ax = sx - x0, ay = sy - y0;
    int xi0 = vx_wrap(x0, w, addr), xi1 = vx_wrap(x0 + 1, w, addr);
    int yi0 = vx_wrap(y0, h, addr), yi1 = vx_wrap(y0 + 1, h, addr);
    float r00=0,g00=0,b00=0,a00=0, r10=0,g10=0,b10=0,a10=0;
    float r01=0,g01=0,b01=0,a01=0, r11=0,g11=0,b11=0,a11=0;
    if (xi0>=0&&yi0>=0) vx_load_rgba_f(img,xi0,yi0,&r00,&g00,&b00,&a00);
    if (xi1>=0&&yi0>=0) vx_load_rgba_f(img,xi1,yi0,&r10,&g10,&b10,&a10);
    if (xi0>=0&&yi1>=0) vx_load_rgba_f(img,xi0,yi1,&r01,&g01,&b01,&a01);
    if (xi1>=0&&yi1>=0) vx_load_rgba_f(img,xi1,yi1,&r11,&g11,&b11,&a11);
    float w00=(1-ax)*(1-ay), w10=ax*(1-ay), w01=(1-ax)*ay, w11=ax*ay;
    float r=r00*w00+r10*w10+r01*w01+r11*w11;
    float g=g00*w00+g10*w10+g01*w01+g11*w11;
    float b=b00*w00+b10*w10+b01*w01+b11*w11;
    float a=a00*w00+a10*w10+a01*w01+a11*w11;
    return (float4)(r,g,b,a);
  }
  // nearest
  int x = vx_wrap((int)floor(fx), w, addr);
  int y = vx_wrap((int)floor(fy), h, addr);
  float r=0,g=0,b=0,a=0;
  if (x>=0 && y>=0) vx_load_rgba_f(img,x,y,&r,&g,&b,&a);
  return (float4)(r,g,b,a);
}

float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  return vx_read_imagef_2d(img, (int)__builtin_astype(sampler, uintptr_t), (float)coord.x, (float)coord.y);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_t image, sampler_t sampler, float2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  return vx_read_imagef_2d(img, (int)__builtin_astype(sampler, uintptr_t), coord.x, coord.y);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_t image, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  // sampler-less reads: unnormalized, clamp-to-edge, nearest
  return vx_read_imagef_2d(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE
                                | CLK_FILTER_NEAREST, (float)coord.x, (float)coord.y);
}

// ============================ read_imagei / ui (nearest, no filter) ===========
static void vx_int_coord(global dev_image_t* img, int smp, float u, float v,
                         int* ox, int* oy) {
  int w = img->_width, h = img->_height;
  int addr = smp & 0x0e, norm = smp & 1;
  float fx = u, fy = v;
  if (norm == CLK_NORMALIZED_COORDS_TRUE) { fx = u * w; fy = v * h; }
  int x = vx_wrap((int)floor(fx), w, addr);
  int y = vx_wrap((int)floor(fy), h, addr);
  *ox = x; *oy = y;
}

static int4 vx_read_imagei_2d(global dev_image_t* img, int smp, float u, float v) {
  int x,y; vx_int_coord(img,smp,u,v,&x,&y);
  int order=img->_order, ctype=img->_data_type;
  int nchan=vx_num_chan(order), esize=vx_elem_size(ctype);
  int r=0,g=0,b=0,a=1;
  if (x>=0 && y>=0) {
    global const char* p=(global const char*)(size_t)img->_data+vx_texel_off(img,x,y,nchan,esize);
    r=vx_chan_to_i(p,ctype,0);
    if(nchan>1) g=vx_chan_to_i(p,ctype,1);
    if(nchan>2) b=vx_chan_to_i(p,ctype,2);
    if(nchan>3) a=vx_chan_to_i(p,ctype,3);
  }
  return (int4)(r,g,b,a);
}
static uint4 vx_read_imageui_2d(global dev_image_t* img, int smp, float u, float v) {
  int x,y; vx_int_coord(img,smp,u,v,&x,&y);
  int order=img->_order, ctype=img->_data_type;
  int nchan=vx_num_chan(order), esize=vx_elem_size(ctype);
  uint r=0,g=0,b=0,a=1;
  if (x>=0 && y>=0) {
    global const char* p=(global const char*)(size_t)img->_data+vx_texel_off(img,x,y,nchan,esize);
    r=vx_chan_to_ui(p,ctype,0);
    if(nchan>1) g=vx_chan_to_ui(p,ctype,1);
    if(nchan>2) b=vx_chan_to_ui(p,ctype,2);
    if(nchan>3) a=vx_chan_to_ui(p,ctype,3);
  }
  return (uint4)(r,g,b,a);
}

int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image2d_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  return vx_read_imagei_2d(img,(int)__builtin_astype(sampler, uintptr_t),(float)coord.x,(float)coord.y);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image2d_t image, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  // sampler-less reads: unnormalized, clamp-to-edge, nearest
  return vx_read_imagei_2d(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE
                                | CLK_FILTER_NEAREST,(float)coord.x,(float)coord.y);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image2d_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  return vx_read_imageui_2d(img,(int)__builtin_astype(sampler, uintptr_t),(float)coord.x,(float)coord.y);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image2d_t image, int2 coord) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  // sampler-less reads: unnormalized, clamp-to-edge, nearest
  return vx_read_imageui_2d(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE
                                 | CLK_FILTER_NEAREST,(float)coord.x,(float)coord.y);
}

// ================= generalized shapes: 1D / 1D array / 1D buffer / ===========
// ================= 2D array / 3D (software path only; FF TEX is 2D) ==========

// Texel base pointer at (x,y,z) — z indexes slices (3D depth or array layer).
_CL_READNONE static global const char* vx_texel_ptr3(global dev_image_t* img,
                                                     int x, int y, int z,
                                                     int nchan, int esize) {
  return (global const char*)(size_t)img->_data
       + (size_t)z * img->_slice_pitch
       + (size_t)y * img->_row_pitch + (size_t)x * nchan * esize;
}

static void vx_load_rgba_f3(global dev_image_t* img, int x, int y, int z,
                            float* r, float* g, float* b, float* a) {
  int order = img->_order, ctype = img->_data_type;
  int nchan = vx_num_chan(order), esize = vx_elem_size(ctype);
  global const char* p = vx_texel_ptr3(img, x, y, z, nchan, esize);
  float c0 = vx_chan_to_f(p, ctype, 0);
  float c1 = nchan > 1 ? vx_chan_to_f(p, ctype, 1) : 0.0f;
  float c2 = nchan > 2 ? vx_chan_to_f(p, ctype, 2) : 0.0f;
  float c3 = nchan > 3 ? vx_chan_to_f(p, ctype, 3) : 0.0f;
  switch (order) {
  case CLK_A:         *r = 0; *g = 0; *b = 0; *a = c0; return;
  case CLK_R:         *r = c0; *g = 0; *b = 0; *a = 1; return;
  case CLK_INTENSITY: *r = c0; *g = c0; *b = c0; *a = c0; return;
  case CLK_LUMINANCE: *r = c0; *g = c0; *b = c0; *a = 1; return;
  case CLK_RG:        *r = c0; *g = c1; *b = 0; *a = 1; return;
  case CLK_RA:        *r = c0; *g = 0; *b = 0; *a = c1; return;
  case CLK_RGB:       *r = c0; *g = c1; *b = c2; *a = 1; return;
  case CLK_BGRA:      *r = c2; *g = c1; *b = c0; *a = c3; return;
  case CLK_ARGB:      *r = c1; *g = c2; *b = c3; *a = c0; return;
  default:            *r = c0; *g = c1; *b = c2; *a = c3; return; // RGBA
  }
}

// Array-layer selection: clamp(rint(coord), 0, n-1); never normalized.
_CL_READNONE static int vx_layer(float c, int n) {
  int l = (int)rint(c);
  return l < 0 ? 0 : (l >= n ? n - 1 : l);
}

// ---- float reads -------------------------------------------------------------

// 1D filtering along x within row (y,z) fixed.
static float4 vx_read_imagef_row(global dev_image_t* img, int smp, float u,
                                 int y, int z) {
  int w = img->_width;
  int addr = smp & 0x0e, filt = smp & 0x30, norm = smp & 1;
  float fx = (norm == CLK_NORMALIZED_COORDS_TRUE) ? u * w : u;
  if (filt == CLK_FILTER_LINEAR) {
    float sx = fx - 0.5f;
    int x0 = (int)floor(sx);
    float ax = sx - x0;
    int xi0 = vx_wrap(x0, w, addr), xi1 = vx_wrap(x0 + 1, w, addr);
    float r0=0,g0=0,b0=0,a0=0, r1=0,g1=0,b1=0,a1=0;
    if (xi0>=0) vx_load_rgba_f3(img,xi0,y,z,&r0,&g0,&b0,&a0);
    if (xi1>=0) vx_load_rgba_f3(img,xi1,y,z,&r1,&g1,&b1,&a1);
    return (float4)(r0*(1-ax)+r1*ax, g0*(1-ax)+g1*ax,
                    b0*(1-ax)+b1*ax, a0*(1-ax)+a1*ax);
  }
  int x = vx_wrap((int)floor(fx), w, addr);
  float r=0,g=0,b=0,a=0;
  if (x>=0) vx_load_rgba_f3(img,x,y,z,&r,&g,&b,&a);
  return (float4)(r,g,b,a);
}

// 2D filtering within slice z (3D slice or array layer).
static float4 vx_read_imagef_slice(global dev_image_t* img, int smp,
                                   float u, float v, int z) {
  int w = img->_width, h = img->_height;
  int addr = smp & 0x0e, filt = smp & 0x30, norm = smp & 1;
  float fx = u, fy = v;
  if (norm == CLK_NORMALIZED_COORDS_TRUE) { fx = u * w; fy = v * h; }
  if (filt == CLK_FILTER_LINEAR) {
    float sx = fx - 0.5f, sy = fy - 0.5f;
    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    float ax = sx - x0, ay = sy - y0;
    int xi0 = vx_wrap(x0, w, addr), xi1 = vx_wrap(x0 + 1, w, addr);
    int yi0 = vx_wrap(y0, h, addr), yi1 = vx_wrap(y0 + 1, h, addr);
    float r00=0,g00=0,b00=0,a00=0, r10=0,g10=0,b10=0,a10=0;
    float r01=0,g01=0,b01=0,a01=0, r11=0,g11=0,b11=0,a11=0;
    if (xi0>=0&&yi0>=0) vx_load_rgba_f3(img,xi0,yi0,z,&r00,&g00,&b00,&a00);
    if (xi1>=0&&yi0>=0) vx_load_rgba_f3(img,xi1,yi0,z,&r10,&g10,&b10,&a10);
    if (xi0>=0&&yi1>=0) vx_load_rgba_f3(img,xi0,yi1,z,&r01,&g01,&b01,&a01);
    if (xi1>=0&&yi1>=0) vx_load_rgba_f3(img,xi1,yi1,z,&r11,&g11,&b11,&a11);
    float w00=(1-ax)*(1-ay), w10=ax*(1-ay), w01=(1-ax)*ay, w11=ax*ay;
    return (float4)(r00*w00+r10*w10+r01*w01+r11*w11,
                    g00*w00+g10*w10+g01*w01+g11*w11,
                    b00*w00+b10*w10+b01*w01+b11*w11,
                    a00*w00+a10*w10+a01*w01+a11*w11);
  }
  int x = vx_wrap((int)floor(fx), w, addr);
  int y = vx_wrap((int)floor(fy), h, addr);
  float r=0,g=0,b=0,a=0;
  if (x>=0 && y>=0) vx_load_rgba_f3(img,x,y,z,&r,&g,&b,&a);
  return (float4)(r,g,b,a);
}

// Full 3D (trilinear when CLK_FILTER_LINEAR).
static float4 vx_read_imagef_3d(global dev_image_t* img, int smp,
                                float u, float v, float t) {
  int w = img->_width, h = img->_height, d = img->_depth;
  int addr = smp & 0x0e, filt = smp & 0x30, norm = smp & 1;
  float fz = (norm == CLK_NORMALIZED_COORDS_TRUE) ? t * d : t;
  if (filt == CLK_FILTER_LINEAR) {
    float sz = fz - 0.5f;
    int z0 = (int)floor(sz);
    float az = sz - z0;
    int zi0 = vx_wrap(z0, d, addr), zi1 = vx_wrap(z0 + 1, d, addr);
    float4 s0 = (float4)(0), s1 = (float4)(0);
    if (zi0 >= 0) s0 = vx_read_imagef_slice(img, smp, u, v, zi0);
    if (zi1 >= 0) s1 = vx_read_imagef_slice(img, smp, u, v, zi1);
    return s0 * (1 - az) + s1 * az;
  }
  int z = vx_wrap((int)floor(fz), d, addr);
  if (z < 0) return (float4)(0);
  return vx_read_imagef_slice(img, smp, u, v, z);
}

// ---- integer reads (nearest only, per spec) -----------------------------------

_CL_READNONE static int vx_int_coord1(int size, int smp, float c) {
  int addr = smp & 0x0e, norm = smp & 1;
  float fc = (norm == CLK_NORMALIZED_COORDS_TRUE) ? c * size : c;
  return vx_wrap((int)floor(fc), size, addr);
}

static int4 vx_read_imagei_at(global dev_image_t* img, int x, int y, int z) {
  int order=img->_order, ctype=img->_data_type;
  int nchan=vx_num_chan(order), esize=vx_elem_size(ctype);
  int r=0,g=0,b=0,a=1;
  if (x>=0 && y>=0 && z>=0) {
    global const char* p = vx_texel_ptr3(img,x,y,z,nchan,esize);
    r=vx_chan_to_i(p,ctype,0);
    if(nchan>1) g=vx_chan_to_i(p,ctype,1);
    if(nchan>2) b=vx_chan_to_i(p,ctype,2);
    if(nchan>3) a=vx_chan_to_i(p,ctype,3);
  }
  return (int4)(r,g,b,a);
}
static uint4 vx_read_imageui_at(global dev_image_t* img, int x, int y, int z) {
  int order=img->_order, ctype=img->_data_type;
  int nchan=vx_num_chan(order), esize=vx_elem_size(ctype);
  uint r=0,g=0,b=0,a=1;
  if (x>=0 && y>=0 && z>=0) {
    global const char* p = vx_texel_ptr3(img,x,y,z,nchan,esize);
    r=vx_chan_to_ui(p,ctype,0);
    if(nchan>1) g=vx_chan_to_ui(p,ctype,1);
    if(nchan>2) b=vx_chan_to_ui(p,ctype,2);
    if(nchan>3) a=vx_chan_to_ui(p,ctype,3);
  }
  return (uint4)(r,g,b,a);
}

#define VX_SMP_DEFAULT (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST)
#define VX_IMG(image) __builtin_astype(image, global dev_image_t*)
#define VX_SMPI(sampler) ((int)__builtin_astype(sampler, uintptr_t))

// ---- image1d_t ---------------------------------------------------------------
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_t image, sampler_t sampler, int coord) {
  return vx_read_imagef_row(VX_IMG(image), VX_SMPI(sampler), (float)coord, 0, 0);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_t image, sampler_t sampler, float coord) {
  return vx_read_imagef_row(VX_IMG(image), VX_SMPI(sampler), coord, 0, 0);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_t image, int coord) {
  return vx_read_imagef_row(VX_IMG(image), VX_SMP_DEFAULT, (float)coord, 0, 0);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_t image, sampler_t sampler, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), (float)coord), 0, 0);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_t image, sampler_t sampler, float coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), coord), 0, 0);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_t image, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord), 0, 0);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_t image, sampler_t sampler, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), (float)coord), 0, 0);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_t image, sampler_t sampler, float coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), coord), 0, 0);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_t image, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord), 0, 0);
}

// ---- image1d_buffer_t (samplerless only) --------------------------------------
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_buffer_t image, int coord) {
  return vx_read_imagef_row(VX_IMG(image), VX_SMP_DEFAULT, (float)coord, 0, 0);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_buffer_t image, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord), 0, 0);
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_buffer_t image, int coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord), 0, 0);
}

// ---- image1d_array_t (coord.y = layer) -----------------------------------------
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_array_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_row(img, VX_SMPI(sampler), (float)coord.x, 0,
                            vx_layer((float)coord.y, img->_image_array_size));
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_array_t image, sampler_t sampler, float2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_row(img, VX_SMPI(sampler), coord.x, 0,
                            vx_layer(coord.y, img->_image_array_size));
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image1d_array_t image, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_row(img, VX_SMP_DEFAULT, (float)coord.x, 0,
                            vx_layer((float)coord.y, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_array_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), (float)coord.x),
                           0, vx_layer((float)coord.y, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_array_t image, sampler_t sampler, float2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), coord.x),
                           0, vx_layer(coord.y, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image1d_array_t image, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                           0, vx_layer((float)coord.y, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_array_t image, sampler_t sampler, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), (float)coord.x),
                            0, vx_layer((float)coord.y, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_array_t image, sampler_t sampler, float2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMPI(sampler), coord.x),
                            0, vx_layer(coord.y, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image1d_array_t image, int2 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                            0, vx_layer((float)coord.y, img->_image_array_size));
}

// ---- image2d_array_t (coord.z = layer) ------------------------------------------
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_array_t image, sampler_t sampler, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_slice(img, VX_SMPI(sampler), (float)coord.x, (float)coord.y,
                              vx_layer((float)coord.z, img->_image_array_size));
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_array_t image, sampler_t sampler, float4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_slice(img, VX_SMPI(sampler), coord.x, coord.y,
                              vx_layer(coord.z, img->_image_array_size));
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image2d_array_t image, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagef_slice(img, VX_SMP_DEFAULT, (float)coord.x, (float)coord.y,
                              vx_layer((float)coord.z, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image2d_array_t image, sampler_t sampler, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, smp, (float)coord.x),
                           vx_int_coord1(img->_height, smp, (float)coord.y),
                           vx_layer((float)coord.z, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image2d_array_t image, sampler_t sampler, float4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, smp, coord.x),
                           vx_int_coord1(img->_height, smp, coord.y),
                           vx_layer(coord.z, img->_image_array_size));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image2d_array_t image, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                           vx_int_coord1(img->_height, VX_SMP_DEFAULT, (float)coord.y),
                           vx_layer((float)coord.z, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image2d_array_t image, sampler_t sampler, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, smp, (float)coord.x),
                            vx_int_coord1(img->_height, smp, (float)coord.y),
                            vx_layer((float)coord.z, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image2d_array_t image, sampler_t sampler, float4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, smp, coord.x),
                            vx_int_coord1(img->_height, smp, coord.y),
                            vx_layer(coord.z, img->_image_array_size));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image2d_array_t image, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                            vx_int_coord1(img->_height, VX_SMP_DEFAULT, (float)coord.y),
                            vx_layer((float)coord.z, img->_image_array_size));
}

// ---- image3d_t ------------------------------------------------------------------
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image3d_t image, sampler_t sampler, int4 coord) {
  return vx_read_imagef_3d(VX_IMG(image), VX_SMPI(sampler),
                           (float)coord.x, (float)coord.y, (float)coord.z);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image3d_t image, sampler_t sampler, float4 coord) {
  return vx_read_imagef_3d(VX_IMG(image), VX_SMPI(sampler), coord.x, coord.y, coord.z);
}
float4 _CL_OVERLOADABLE read_imagef(IMG_RO_AQ image3d_t image, int4 coord) {
  return vx_read_imagef_3d(VX_IMG(image), VX_SMP_DEFAULT,
                           (float)coord.x, (float)coord.y, (float)coord.z);
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image3d_t image, sampler_t sampler, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, smp, (float)coord.x),
                           vx_int_coord1(img->_height, smp, (float)coord.y),
                           vx_int_coord1(img->_depth, smp, (float)coord.z));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image3d_t image, sampler_t sampler, float4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, smp, coord.x),
                           vx_int_coord1(img->_height, smp, coord.y),
                           vx_int_coord1(img->_depth, smp, coord.z));
}
int4 _CL_OVERLOADABLE read_imagei(IMG_RO_AQ image3d_t image, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imagei_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                           vx_int_coord1(img->_height, VX_SMP_DEFAULT, (float)coord.y),
                           vx_int_coord1(img->_depth, VX_SMP_DEFAULT, (float)coord.z));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image3d_t image, sampler_t sampler, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, smp, (float)coord.x),
                            vx_int_coord1(img->_height, smp, (float)coord.y),
                            vx_int_coord1(img->_depth, smp, (float)coord.z));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image3d_t image, sampler_t sampler, float4 coord) {
  global dev_image_t* img = VX_IMG(image);
  int smp = VX_SMPI(sampler);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, smp, coord.x),
                            vx_int_coord1(img->_height, smp, coord.y),
                            vx_int_coord1(img->_depth, smp, coord.z));
}
uint4 _CL_OVERLOADABLE read_imageui(IMG_RO_AQ image3d_t image, int4 coord) {
  global dev_image_t* img = VX_IMG(image);
  return vx_read_imageui_at(img, vx_int_coord1(img->_width, VX_SMP_DEFAULT, (float)coord.x),
                            vx_int_coord1(img->_height, VX_SMP_DEFAULT, (float)coord.y),
                            vx_int_coord1(img->_depth, VX_SMP_DEFAULT, (float)coord.z));
}
