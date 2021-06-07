#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <assert.h>
#include <unistd.h>
#include <CL/opencl.h>
#include <string.h>

#define SIZE 64
#define NUM_WORK_GROUPS 1
#define KERNEL_NAME "vecadd"
//#define KERNEL_NAME "_Z6vecaddPfS_S_"

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
	 cleanup();			                                                     \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                       \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   cleanup();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

int exitcode = 0;
char* kernel_file = NULL;
char* kernel_bin = NULL;

cl_context context = NULL;
cl_command_queue cmd_queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;  
cl_int *A = NULL;
cl_int *B = NULL;
cl_int *C = NULL;

static int read_kernel_file(const char* filename, char** source_str, size_t* source_size) {
  assert(filename && source_str && source_size);

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *source_str = (char*)malloc(fsize);
  *source_size = fread(*source_str, 1, fsize, fp);
  
  fclose(fp);
  
  return 0;
}

static void cleanup() {
  if (kernel_bin) free(kernel_bin);
  if (cmd_queue) clReleaseCommandQueue(cmd_queue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);  
  if (context) clReleaseContext(context);
  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
}

static int find_device(char* name, cl_platform_id platform_id, cl_device_id *device_id) {
  cl_device_id device_ids[64];
  cl_uint num_devices = 0;

  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 64, device_ids, &num_devices));

  for (int i=0; i<num_devices; i++) 	{
		char buffer[1024];
		cl_uint buf_uint;
		cl_ulong buf_ulong;

		CL_CHECK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		
    if (name == NULL ||  0 == strncmp(buffer, name, strlen(name))) {
      *device_id = device_ids[i];
      printf("Using OpenCL Device: %s\n", buffer);
      return 0;
    }
	}

  return 1;
}

static void show_usage() {
  printf("Vecadd Test.\n");
  printf("Usage: -k:<kernel-file> [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h?")) != -1) {
    switch (c) {
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  if (NULL == kernel_file) {
    show_usage();
    exit(-1);
  }
}

int main (int argc, char **argv) {  
  cl_platform_id platform_id;
  cl_device_id device_id;  
  cl_int binary_status = 0;
  size_t kernel_size = 0;
  int i;

  parse_args(argc, argv);

  // Open kernel file  
  if (0 != read_kernel_file(kernel_file, &kernel_bin, &kernel_size))
    return -1;
  
  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(find_device(NULL, platform_id, &device_id));
  //CL_CHECK(find_device("vortex", platform_id, &device_id));

  // Creating context.
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  // Memory buffers for each array
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(cl_int), NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(cl_int), NULL, &_err));

  // Allocate memories for input arrays and output arrays.  
  A = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  B = (cl_int*)malloc(sizeof(cl_int)*SIZE);
  C = (cl_int*)malloc(sizeof(cl_int)*SIZE);	
	
  // Initialize values for array members.  
  for (i=0; i<SIZE; ++i) {
    A[i] = i*2+0;
    B[i] = i*2+1;
  }

  // Create program from kernel source
  program = CL_CHECK2(clCreateProgramWithBinary(context, 1, &device_id, &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &_err));	

  // Build program
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
  
  // Create kernel
  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  // Set arguments for kernel
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));	
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_memobj));

  // Creating command queue
  cmd_queue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

	// Copy lists to memory buffers
  CL_CHECK(clEnqueueWriteBuffer(cmd_queue, a_memobj, CL_TRUE, 0, SIZE * sizeof(float), A, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(cmd_queue, b_memobj, CL_TRUE, 0, SIZE * sizeof(float), B, 0, NULL, NULL));

  // Execute the kernel
  size_t globalItemSize = SIZE;
  size_t localItemSize = SIZE/NUM_WORK_GROUPS;
  CL_CHECK(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL));
  CL_CHECK(clFinish(cmd_queue));

  // Read from device back to host.
  CL_CHECK(clEnqueueReadBuffer(cmd_queue, c_memobj, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL));

  // Test if correct answer
  int exitcode = 0;
  for (i=0; i<SIZE; ++i) {
    printf("C: %d A: %d B:%d\n", C[i], A[i], B[i]);
    if (C[i] != (A[i] + B[i])) {
      printf("Failed!\n");
      exitcode = 1;
      break;
    }
  }
  if (i == SIZE) {
    printf("Ok!\n");
  }

  // Clean up		
  cleanup();  

  return exitcode;
}
