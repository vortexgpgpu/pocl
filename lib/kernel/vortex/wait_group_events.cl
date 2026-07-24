/* OpenCL built-in library: wait_group_events() -- Vortex implementation.
 *
 * The generic pocl version is an empty stub, which is only sound under the
 * work-item-loop execution model where every work-item of a group runs
 * sequentially on one thread, so a copy performed by work-item 0 is already
 * complete by the time any other work-item observes it.
 *
 * Vortex executes a work-group across several hardware warps that run
 * independently, so the stub lets the warps that did not perform the copy race
 * ahead and read memory the copying work-item has not written yet. The spec
 * makes wait_group_events a work-group synchronization point, so issue a real
 * barrier and fence both address spaces the async copies may have touched.
 */

void _CL_OVERLOADABLE wait_group_events (int num_events,
                                         event_t *event_list)
{
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
