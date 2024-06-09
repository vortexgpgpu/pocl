#include <vx_print.h>

int printf (const char *restrict fmt, ...) {
  int ret;
	va_list va;
	va_start(va, fmt);
	ret = vx_vprintf(fmt, va);
	va_end(va);
  return ret;
}
