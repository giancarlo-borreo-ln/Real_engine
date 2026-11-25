/**
 * @file harness_stub.h
 * @brief Compile-time shims so the CBMC harnesses also pass a plain
 *        clang -fsyntax-only check. CBMC provides the real __CPROVER_*
 *        builtins and nondet_* generators; these stubs exist only for
 *        editor/CI syntax validation and are compiled out under CBMC.
 */
#ifndef RE_HARNESS_STUB_H
#define RE_HARNESS_STUB_H

#ifndef __CPROVER__

#include <stddef.h>

#define __CPROVER_assert(cond, msg) \
    do { (void)(cond); (void)sizeof(msg); } while (0)
#define __CPROVER_assume(cond) ((void)(cond))
#define __CPROVER_havoc_object(x) ((void)sizeof(x))

static inline unsigned nondet_uint(void) { return 0u; }
static inline unsigned long nondet_ulong(void) { return 0u; }
static inline size_t nondet_size_t(void) { return 0u; }
static inline unsigned char nondet_uchar(void) { return 0u; }
static inline unsigned short nondet_ushort(void) { return 0u; }
static inline float nondet_float(void) { return 0.0f; }

#endif /* !__CPROVER__ */

#endif /* RE_HARNESS_STUB_H */
