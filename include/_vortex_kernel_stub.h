#include <inttypes.h>
#include <vx_intrinsics.h>

#define NUM_CORES_MAX 32

#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  char * printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

typedef void (*pfn_workgroup_func) (
  const void * /* args */,
	const struct context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef struct {
  struct context_t * ctx;
  pfn_workgroup_func wg_func;
  const void * args;
  int  offset; 
  int  N;
  int  R;  
  bool isXYpow2;
  bool isXpow2;
  char log2XY;
  char log2X;
} wspawn_args_t;

wspawn_args_t* g_wspawn_args[NUM_CORES_MAX];

void kernel_spawn_callback() {  
  vx_tmc(vx_num_threads());

  int core_id = vx_core_id();
  int wid     = vx_warp_id();
  int tid     = vx_thread_id(); 
  int NT      = vx_num_threads();
  
  wspawn_args_t* p_wspawn_args = g_wspawn_args[core_id];

  int wK = (p_wspawn_args->N * wid) + MIN(p_wspawn_args->R, wid);
  int tK = p_wspawn_args->N + (wid < p_wspawn_args->R);
  int offset = p_wspawn_args->offset + (wK * NT) + (tid * tK);

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  for (int wg_id = offset, N = wg_id + tK; wg_id < N; ++wg_id) {    
    int k = wg_id / XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d / X;
    int i = wg_2d - j * X;

    int gid0 = p_wspawn_args->ctx->global_offset[0] + i;
    int gid1 = p_wspawn_args->ctx->global_offset[1] + j;
    int gid2 = p_wspawn_args->ctx->global_offset[2] + k;

    (p_wspawn_args->wg_func)(p_wspawn_args->args, p_wspawn_args->ctx, gid0, gid1, gid2);
  }

  vx_tmc(0 == wid);
}

void kernel_spawn_remaining_callback(int nthreads) {    
  vx_tmc(nthreads);

  int core_id = vx_core_id(); 
  int tid = vx_thread_gid();

  wspawn_args_t* p_wspawn_args = g_wspawn_args[core_id];

  int wg_id = p_wspawn_args->offset + tid;

  int X = p_wspawn_args->ctx->num_groups[0];
  int Y = p_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;
  
  int k = wg_id / XY;
  int wg_2d = wg_id - k * XY;
  int j = wg_2d / X;
  int i = wg_2d - j * X;

  int gid0 = p_wspawn_args->ctx->global_offset[0] + i;
  int gid1 = p_wspawn_args->ctx->global_offset[1] + j;
  int gid2 = p_wspawn_args->ctx->global_offset[2] + k;

  (p_wspawn_args->wg_func)(p_wspawn_args->args, p_wspawn_args->ctx, gid0, gid1, gid2);

  vx_tmc(1);
}

void vx_spawn_kernel(struct context_t * ctx, pfn_workgroup_func wg_func, const void * args) {  
  // total number of WGs
  int X = ctx->num_groups[0];
  int Y = ctx->num_groups[1];
  int Z = ctx->num_groups[2];
  int Q = X * Y * Z;
  
  // device specs
  int NC = vx_num_cores();
  int NW = vx_num_warps();
  int NT = vx_num_threads();

  // current core id
  int core_id = vx_core_id();  
  if (core_id >= NUM_CORES_MAX)
    return;

  // calculate necessary active cores
  int WT = NW * NT;
  int nC = (Q > WT) ? (Q / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate extra cores

  // number of workgroups per core
  int wgs_per_core = Q / nc;
  int wgs_per_core0 = wgs_per_core;  
  if (core_id == (NC-1)) {    
    int QC_r = Q - (nc * wgs_per_core0); 
    wgs_per_core0 += QC_r; // last core executes remaining WGs
  }

  // number of workgroups per warp
  int nW = wgs_per_core0 / NT;              // total warps per core
  int rT = wgs_per_core0 - (nW * NT);       // remaining threads
  int fW = (nW >= NW) ? (nW / NW) : 0;      // full warps iterations
  int rW = (fW != 0) ? (nW - fW * NW) : 0;  // reamining full warps
  if (0 == fW)
    fW = 1;

  //--
  wspawn_args_t wspawn_args = { ctx, wg_func, args, core_id * wgs_per_core, fW, rW };
  g_wspawn_args[core_id] = &wspawn_args;

  //--
	if (nW >= 1)	{ 
    int nw = MIN(nW, NW);    
	  vx_wspawn(nw, (unsigned)&kernel_spawn_callback);
    kernel_spawn_callback();
	}  

  //--    
  if (rT != 0) {
    wspawn_args.offset = wgs_per_core0 - rT;
    kernel_spawn_remaining_callback(rT);
  }
}