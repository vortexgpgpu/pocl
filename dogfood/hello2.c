#include <pthread.h>

int main(int argc, char** argv)
{
  (void)argv;
#ifndef pthread_create
  return ((int*)(&pthread_create))[argc];
#else
  (void)argc;
  return 0;
#endif
}