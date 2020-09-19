#include <inttypes.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

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

typedef void (*vx_pocl_workgroup_func) (
  const void * /* args */,
	const struct context_t * /* context */,
	uint32_t /* group_x */,
	uint32_t /* group_y */,
	uint32_t /* group_z */
);

typedef struct {
  struct context_t * ctx;
  vx_pocl_workgroup_func pfn;
  const void * args;
  int wg_offset; 
  int K;
  int R;
} wspawn_args_t;

wspawn_args_t* g_wspawn_args;

void kernel_spawn_run_warp() {  
  vx_tmc(vx_num_threads());

  int NT = vx_num_threads();

  int X = g_wspawn_args->ctx->num_groups[0];
  int Y = g_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  int wid = vx_warp_id();
  int tid = vx_thread_id();

  int wK = (g_wspawn_args->K * wid) + MIN(g_wspawn_args->R, wid);
  int tK = g_wspawn_args->K + (wid < g_wspawn_args->R);
  int wg_base = g_wspawn_args->wg_offset + (wK * NT) + (tid * tK);

  /*vx_printv("wid=", wid);
  vx_printv("tid=", tid);
  vx_printv("wK=", wK);  
  vx_printv("tK=", tK);
  vx_printv("wg_base=", wg_base);*/

  for (int i = 0; i < tK; ++i) {
    int wg_id = wg_base + i;    
    
    int k = wg_id / XY;
    int wg_2d = wg_id - k * XY;
    int j = wg_2d / X;
    int i = wg_2d - j * X;

    int gid0 = g_wspawn_args->ctx->global_offset[0] + i;
    int gid1 = g_wspawn_args->ctx->global_offset[1] + j;
    int gid2 = g_wspawn_args->ctx->global_offset[2] + k;

    /*vx_printv("wg_id=", wg_id);
    vx_printv("wg_2d=", wg_2d);    
    vx_printv("k=", k);
    vx_printv("j=", j);
    vx_printv("i=", i);*/

    (g_wspawn_args->pfn)(g_wspawn_args->args, g_wspawn_args->ctx, gid0, gid1, gid2);
  }

  unsigned tmask = (0 == wid);
  vx_tmc(tmask);
}

void kernel_spawn_run_threads(int nthreads, int wg_offset) {    
  vx_tmc(nthreads);

  int X = g_wspawn_args->ctx->num_groups[0];
  int Y = g_wspawn_args->ctx->num_groups[1];
  int XY = X * Y;

  int tid = vx_thread_gid();

  int wg_id = wg_offset + tid;
  
  int k = wg_id / XY;
  int wg_2d = wg_id - k * XY;
  int j = wg_2d / X;
  int i = wg_2d - j * X;

  int gid0 = g_wspawn_args->ctx->global_offset[0] + i;
  int gid1 = g_wspawn_args->ctx->global_offset[1] + j;
  int gid2 = g_wspawn_args->ctx->global_offset[2] + k;

  /*vx_printv("-wg_id=", wg_id);
  vx_printv("-wg_2d=", wg_2d);    
  vx_printv("-k=", k);
  vx_printv("-j=", j);
  vx_printv("-i=", i);*/

  (g_wspawn_args->pfn)(g_wspawn_args->args, g_wspawn_args->ctx, gid0, gid1, gid2);

  vx_tmc(1);
}

void kernel_spawn(struct context_t * ctx, vx_pocl_workgroup_func pfn, const void * args) {  
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

  // calculate number of active cores
  int WT = NW * NT;
  int nC = (Q > WT) ? (Q / WT) : 1;
  int nc = MIN(nC, NC);
  if (core_id >= nc)
    return; // terminate unused cores

  // number of workgroups per core
  int QC_base = Q / nc;
  int QC = QC_base;  
  if (core_id == (NC-1)) {    
    int QC_r = Q - (nc * QC); 
    QC += QC_r; // last core executes remaining WGs
  }

  // number of workgroups per warp
  int nW = QC / NT;
  int qR = QC - (nW * NT);
  int K  = (nW >= NW) ? (nW / NW) : 0;
  int R  = (K != 0) ? (nW - K * NW) : 0; 
  if (0 == K)
    K = 1;

  //--
  wspawn_args_t wspawn_args = { ctx, pfn, args, core_id * QC_base, K, R};
  g_wspawn_args = &wspawn_args;

  /*vx_printv("X=", X);
  vx_printv("Y=", Y);
  vx_printv("Z=", Z);
  vx_printv("Q=", Q);
  vx_printv("NC=", NC);
  vx_printv("NW=", NW);
  vx_printv("NT=", NT);
  vx_printv("core_id=", core_id);
  vx_printv("nc=", nc);
  vx_printv("QC=", QC);
  vx_printv("nW=", nW);
  vx_printv("qR=", qR);
  vx_printv("K=", K);
  vx_printv("R=", R);*/

  //--
	if (nW > 1)	{ 
    int nw = MIN(nW, NW);    
	  vx_wspawn(nw, (unsigned)&kernel_spawn_run_warp);
    kernel_spawn_run_warp();
	}  

  //--    
  if (qR != 0) {
    kernel_spawn_run_threads(qR, QC - qR);
  }
} 