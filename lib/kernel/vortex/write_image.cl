/* OpenCL image write builtins — Vortex scalar implementation.
 *
 * Companion to vortex/read_image.cl: all encoding is scalar (the float4/int4
 * color argument is only decomposed via extractelement, never vector-arithmetic)
 * so it selects on the no-vector Vortex backend. 2D images, CL 1.2 required
 * formats.
 */

#include "../templates.h"

_CL_READNONE static ushort vx_float_to_half(float f) {
  uint x = as_uint(f);
  uint sign = (x >> 16) & 0x8000u;
  int  exp  = (int)((x >> 23) & 0xffu) - 127 + 15;
  uint man  = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return (ushort)sign;
    man |= 0x800000u;
    int shift = 14 - exp;
    uint h = man >> shift;
    if ((man >> (shift - 1)) & 1u) h += 1;   // round to nearest
    return (ushort)(sign | h);
  } else if (exp >= 31) {
    return (ushort)(sign | 0x7c00u);
  }
  uint h = ((uint)exp << 10) | (man >> 13);
  if (man & 0x1000u) h += 1;                 // round to nearest
  return (ushort)(sign | h);
}

static void vx_store_chan_f(global char* p, int ctype, int idx, float v) {
  switch (ctype) {
  case CLK_UNORM_INT8:  ((global uchar*)p)[idx]  = (uchar)(clamp(v,0.0f,1.0f)*255.0f+0.5f); break;
  case CLK_SNORM_INT8:  ((global char*)p)[idx]   = (char)(clamp(v,-1.0f,1.0f)*127.0f+(v<0?-0.5f:0.5f)); break;
  case CLK_UNORM_INT16: ((global ushort*)p)[idx] = (ushort)(clamp(v,0.0f,1.0f)*65535.0f+0.5f); break;
  case CLK_SNORM_INT16: ((global short*)p)[idx]  = (short)(clamp(v,-1.0f,1.0f)*32767.0f+(v<0?-0.5f:0.5f)); break;
  case CLK_HALF_FLOAT:  ((global ushort*)p)[idx] = vx_float_to_half(v); break;
  case CLK_FLOAT:       ((global float*)p)[idx]  = v; break;
  default:              ((global uchar*)p)[idx]  = (uchar)(clamp(v,0.0f,1.0f)*255.0f+0.5f); break;
  }
}

_CL_READNONE static int vx_wnum_chan(int order) {
  switch (order) {
  case CLK_R: case CLK_A: case CLK_INTENSITY: case CLK_LUMINANCE: return 1;
  case CLK_RG: case CLK_RA:                                       return 2;
  case CLK_RGB:                                                   return 3;
  default:                                                        return 4;
  }
}
_CL_READNONE static int vx_welem_size(int ctype) {
  switch (ctype) {
  case CLK_UNORM_INT8: case CLK_SNORM_INT8:
  case CLK_SIGNED_INT8: case CLK_UNSIGNED_INT8:   return 1;
  case CLK_UNORM_INT16: case CLK_SNORM_INT16:
  case CLK_SIGNED_INT16: case CLK_UNSIGNED_INT16:
  case CLK_HALF_FLOAT:                            return 2;
  default:                                        return 4;
  }
}

static global char* vx_wbase(global dev_image_t* img, int x, int y, int* nchan) {
  int order=img->_order, ctype=img->_data_type;
  int nc=vx_wnum_chan(order), es=vx_welem_size(ctype);
  *nchan=nc;
  return (global char*)(size_t)img->_data + (size_t)y*img->_row_pitch + (size_t)x*nc*es;
}

void _CL_OVERLOADABLE write_imagef(IMG_WO_AQ image2d_t image, int2 coord, float4 color) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  int nchan; global char* p = vx_wbase(img, coord.x, coord.y, &nchan);
  int ctype=img->_data_type, order=img->_order;
  // Map RGBA lanes back to stored order (only the common orders that write_image
  // supports); scalar extraction, no vector math.
  float c0,c1,c2,c3;
  if (order==CLK_BGRA) { c0=color.z; c1=color.y; c2=color.x; c3=color.w; }
  else if (order==CLK_ARGB) { c0=color.w; c1=color.x; c2=color.y; c3=color.z; }
  else if (order==CLK_A) { c0=color.w; c1=c2=c3=0; }
  else { c0=color.x; c1=color.y; c2=color.z; c3=color.w; }
  vx_store_chan_f(p,ctype,0,c0);
  if(nchan>1) vx_store_chan_f(p,ctype,1,c1);
  if(nchan>2) vx_store_chan_f(p,ctype,2,c2);
  if(nchan>3) vx_store_chan_f(p,ctype,3,c3);
}

static void vx_store_chan_i(global char* p, int ctype, int idx, int v) {
  switch (ctype) {
  case CLK_SIGNED_INT8:  ((global char*)p)[idx]  = (char)v; break;
  case CLK_SIGNED_INT16: ((global short*)p)[idx] = (short)v; break;
  default:               ((global int*)p)[idx]   = v; break;
  }
}
static void vx_store_chan_ui(global char* p, int ctype, int idx, uint v) {
  switch (ctype) {
  case CLK_UNSIGNED_INT8:  ((global uchar*)p)[idx]  = (uchar)v; break;
  case CLK_UNSIGNED_INT16: ((global ushort*)p)[idx] = (ushort)v; break;
  default:                 ((global uint*)p)[idx]   = v; break;
  }
}

void _CL_OVERLOADABLE write_imagei(IMG_WO_AQ image2d_t image, int2 coord, int4 color) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  int nchan; global char* p = vx_wbase(img, coord.x, coord.y, &nchan);
  int ctype=img->_data_type;
  vx_store_chan_i(p,ctype,0,color.x);
  if(nchan>1) vx_store_chan_i(p,ctype,1,color.y);
  if(nchan>2) vx_store_chan_i(p,ctype,2,color.z);
  if(nchan>3) vx_store_chan_i(p,ctype,3,color.w);
}
void _CL_OVERLOADABLE write_imageui(IMG_WO_AQ image2d_t image, int2 coord, uint4 color) {
  global dev_image_t* img = __builtin_astype(image, global dev_image_t*);
  int nchan; global char* p = vx_wbase(img, coord.x, coord.y, &nchan);
  int ctype=img->_data_type;
  vx_store_chan_ui(p,ctype,0,color.x);
  if(nchan>1) vx_store_chan_ui(p,ctype,1,color.y);
  if(nchan>2) vx_store_chan_ui(p,ctype,2,color.z);
  if(nchan>3) vx_store_chan_ui(p,ctype,3,color.w);
}
