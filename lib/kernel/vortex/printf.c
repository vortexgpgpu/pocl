#include <stdarg.h> 

// Make dummy printf for lowering 
int
printf (const char *restrict fmt, ...)
{
  return 0;
}
