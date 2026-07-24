/* OpenCL runtime library: clCreateContextFromType()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include "devices/devices.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include <stdlib.h>
#include <string.h>

CL_API_ENTRY cl_context CL_API_CALL
POname(clCreateContextFromType)(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                        void *user_data,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  int errcode;

  /* device_type must be CL_DEVICE_TYPE_ALL or a non-empty combination of
   * the defined type bits. */
  POCL_GOTO_ERROR_COND (
      (device_type == 0
       || (device_type != CL_DEVICE_TYPE_ALL
           && (device_type
               & ~(CL_DEVICE_TYPE_DEFAULT | CL_DEVICE_TYPE_CPU
                   | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
                   | CL_DEVICE_TYPE_CUSTOM))
                  != 0)),
      CL_INVALID_DEVICE_TYPE);

  errcode = pocl_init_devices();
  /* see clCreateContext.c for explanation */
  POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), CL_INVALID_DEVICE,
                      "Could not initialize devices\n");

  unsigned num_devices = pocl_get_device_type_count (device_type);

  /* No matching device: the spec requires a NULL context with
   * CL_DEVICE_NOT_FOUND. (An old workaround returned a dummy 0-device
   * context for 2012-era AMD SDK samples; the CTS rejects that.) */
  POCL_GOTO_ERROR_COND ((num_devices == 0), CL_DEVICE_NOT_FOUND);

  cl_device_id *devs
      = (cl_device_id *)alloca (num_devices * sizeof (cl_device_id));

  pocl_get_devices (device_type, devs, num_devices);

  return POname (clCreateContext) (properties, num_devices, devs, pfn_notify,
                                   user_data, errcode_ret);

ERROR:
  if (errcode_ret != NULL)
    {
      *errcode_ret = errcode;
    }

  return NULL;
}
POsym(clCreateContextFromType)
