#if !defined(OCS_AVAILABLE)
#include "vortex.h"
#include <stdarg.h> 
#endif

int
printf (const char *restrict fmt, ...)
{
#if !defined(OCS_AVAILABLE)
  va_list args; 
  va_start(args, fmt);
  vx_printf(fmt, args);
  va_end(args);
#endif
  
  return 0;
}
