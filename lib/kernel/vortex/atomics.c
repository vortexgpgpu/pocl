// atomic_add//atomic_inc/atomic_dec/atomic_sub
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

int _Z14_cl_atomic_addPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_add_asm(ptr, val);
}

unsigned int _Z14_cl_atomic_addPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, (int)val);
}

unsigned int _Z14_cl_atomic_addPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, (int)val);
}

// atomic_inc

int _Z14_cl_atomic_incPU8CLglobalVi(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, 1);
}

int _Z14_cl_atomic_incPU7CLlocalVi(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, 1);
}

unsigned int _Z14_cl_atomic_incPU8CLglobalVj(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, 1);
}

unsigned int _Z14_cl_atomic_incPU7CLlocalVj(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, 1);
}

// atomic_dec

int _Z14_cl_atomic_decPU8CLglobalVi(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, -1);
}

int _Z14_cl_atomic_decPU7CLlocalVi(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, -1);
}

unsigned int _Z14_cl_atomic_decPU7CLlocalVj(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, -1);
}

unsigned int _Z14_cl_atomic_decPU8CLglobalVj(volatile void *ptr)
{
    return _vx_atomic_add_asm(ptr, -1);
}

// atomic_sub

int _Z14_cl_atomic_subPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_add_asm(ptr, -val);
}

int _Z14_cl_atomic_subPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_add_asm(ptr, -val);
}

unsigned int _Z14_cl_atomic_subPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, -(int)val);
}

unsigned int _Z14_cl_atomic_subPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_add_asm(ptr, -(int)val);
}

// atomic_max

static inline int _vx_atomic_max_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amomax.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

int _Z14_cl_atomic_maxPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_max_asm(ptr, val);
}

int _Z14_cl_atomic_maxPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_max_asm(ptr, val);
}

// atomic_min

static inline int _vx_atomic_min_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amomin.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

int _Z14_cl_atomic_minPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_min_asm(ptr, val);
}

int _Z14_cl_atomic_minPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_min_asm(ptr, val);
}

// atomic_xor

static inline int _vx_atomic_xor_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amoxor.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

int _Z14_cl_atomic_xorPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_xor_asm(ptr, val);
}

int _Z14_cl_atomic_xorPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_xor_asm(ptr, val);
}

unsigned int _Z14_cl_atomic_xorPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_xor_asm(ptr, (int)val);
}

unsigned int _Z14_cl_atomic_xorPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_xor_asm(ptr, (int)val);
}
// atomic_or

static inline int _vx_atomic_or_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amoor.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

int _Z13_cl_atomic_orPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_or_asm(ptr, val);
}

int _Z13_cl_atomic_orPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_or_asm(ptr, val);
}

unsigned int _Z13_cl_atomic_orPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_or_asm(ptr, (int)val);
}

unsigned int _Z13_cl_atomic_orPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_or_asm(ptr, (int)val);
}

// atomic_and

static inline int _vx_atomic_and_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amoand.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

int _Z14_cl_atomic_andPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_and_asm(ptr, val);
}

int _Z14_cl_atomic_andPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_and_asm(ptr, val);
}

unsigned int _Z14_cl_atomic_andPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_and_asm(ptr, (int)val);
}

unsigned int _Z14_cl_atomic_andPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_and_asm(ptr, (int)val);
}

// atomic_xchg

static inline int _vx_atomic_xchg_asm(volatile void *addr, int value)
{
    int old_value;
    __asm__ volatile(
        "amoswap.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}
int _Z15_cl_atomic_xchgPU8CLglobalVii(volatile void *ptr, int val)
{
    return _vx_atomic_xchg_asm(ptr, val);
}

int _Z15_cl_atomic_xchgPU7CLlocalVii(volatile void *ptr, int val)
{
    return _vx_atomic_xchg_asm(ptr, val);
}

unsigned int _Z15_cl_atomic_xchgPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_xchg_asm(ptr, (int)val);
}

unsigned int _Z15_cl_atomic_xchgPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_xchg_asm(ptr, (int)val);
}

// atomic_cmpxchg

static inline int _vx_atomic_cmpxchg_asm(volatile void *addr, int cmp_val, int new_val)
{
    int old_val;
    int success;
    __asm__ volatile(
        "1: lr.w %0, (%2)\n"
        "   bne %0, %3, 2f\n"
        "   sc.w %1, %4, (%2)\n"
        "   bnez %1, 1b\n"
        "2:"
        : "=&r"(old_val), "=&r"(success)
        : "r"(addr), "r"(cmp_val), "r"(new_val)
        : "memory");
    return old_val;
}

int _Z18_cl_atomic_cmpxchgPU7CLlocalViii(volatile void *ptr, int cmp_val, int new_val)
{
    return _vx_atomic_cmpxchg_asm(ptr, cmp_val, new_val);
}

int _Z18_cl_atomic_cmpxchgPU8CLglobalViii(volatile void *ptr, int cmp_val, int new_val)
{
    return _vx_atomic_cmpxchg_asm(ptr, cmp_val, new_val);
}

unsigned int _Z18_cl_atomic_cmpxchgPU7CLlocalVjjj(volatile void *ptr, unsigned int cmp_val, unsigned int new_val)
{
    return _vx_atomic_cmpxchg_asm(ptr, (int)cmp_val, (int)new_val);
}

unsigned int _Z18_cl_atomic_cmpxchgPU8CLglobalVjjj(volatile void *ptr, unsigned int cmp_val, unsigned int new_val)
{
    return _vx_atomic_cmpxchg_asm(ptr, (int)cmp_val, (int)new_val);
}

static inline unsigned int _vx_atomic_minu_asm(volatile void *addr, unsigned int value)
{
    unsigned int old_value;
    __asm__ volatile(
        "amominu.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

unsigned int _Z14_cl_atomic_minPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_minu_asm(ptr, val);
}

unsigned int _Z14_cl_atomic_minPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_minu_asm(ptr, val);
}

static inline unsigned int _vx_atomic_maxu_asm(volatile void *addr, unsigned int value)
{
    unsigned int old_value;
    __asm__ volatile(
        "amomaxu.w %0, %2, (%1)"
        : "=r"(old_value)
        : "r"(addr), "r"(value)
        : "memory");
    return old_value;
}

unsigned int _Z14_cl_atomic_maxPU7CLlocalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_maxu_asm(ptr, val);
}

unsigned int _Z14_cl_atomic_maxPU8CLglobalVjj(volatile void *ptr, unsigned int val)
{
    return _vx_atomic_maxu_asm(ptr, val);
}