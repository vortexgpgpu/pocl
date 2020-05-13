/* newlib.h - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2016 Pekka Jääskeläinen / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/
/**
 * @file newlib.h
 *
 * The purpose of the 'basic' device driver is to serve as an example of
 * a minimalistic (but still working) device driver for pocl.
 *
 * It is a "native device" without multithreading and uses the malloc
 * directly for buffer allocation etc. It also executes work groups in
 * their sequential (increasing) order, thus makes it useful as a test
 * device.
 */

#ifndef POCL_NEWLIB_H
#define POCL_NEWLIB_H

#include "pocl_cl.h"
#include "pocl_icd.h"
#include "config.h"

#include "prototypes.inc"
GEN_PROTOTYPES (newlib)

struct pocl_context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  uint8_t *printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

typedef void (*pocl_wg_func) (
  void * /* args */,
  void * /* pocl_context */,
  uint32_t /* group_x */,
  uint32_t /* group_y */,
  uint32_t /* group_z */
);

#endif /* POCL_NEWLIB_H */
