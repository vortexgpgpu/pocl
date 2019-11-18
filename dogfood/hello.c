#include <stdio.h>

void _exit(int x) {}

int main(int argc, char **argv) { 
  return (argc * argc);
}