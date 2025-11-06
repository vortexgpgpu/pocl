static inline int _vx_atomic_add_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amoadd.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}


int _Z14_cl_atomic_addPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_add_asm(ptr, val);
}


int _Z14_cl_atomic_addPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, (int)val);
}


int _Z14_cl_atomic_addPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_add_asm(ptr, val);
}

int _Z14_cl_atomic_addPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, (int)val);
}

int _Z14_cl_atomic_incPU8CLglobalVj(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, 1);
}