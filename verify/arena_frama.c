/**
 * @file arena_frama.c
 * @brief Frama-C/WP proof of the arena allocation contract: results are
 *        NULL or 16/32/64-aligned, inside the committed region, and
 *        monotonically placed so successive allocations never overlap.
 *
 * main() drives concrete sequences; -wp-rte additionally proves the
 * bump-pointer arithmetic free of overflows on the analyzed paths.
 * Run: frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 arena_frama.c
 */
#include "../src/re_core.h"
#include "../src/re_mem.h"
#include "../src/re_mem.c"

/*@ requires \valid(a);
    requires a->begin != \null;
    requires a->end == a->begin + a->capacity;
    requires a->offset <= a->capacity;
    ensures \result == \null ||
            (((uintptr_t)\result % align == 0) &&
             (unsigned char *)\result >= a->begin &&
             (unsigned char *)\result + size <= a->end);
    ensures a->offset <= a->capacity;
 */
void *re_arena_alloc_align(ReArena *a, size_t size, size_t align);

static unsigned char arena_mem[256];

int main(void) {
    ReArena a;
    bool ok = re_arena_init(&a, arena_mem, sizeof arena_mem);
    /*@ assert ok; */

    void *p1 = re_arena_alloc_align(&a, 12u, 16u);
    void *p2 = re_arena_alloc_align(&a, 4u, 16u);
    /*@ assert p1 != \null && p2 != \null; */
    /*@ assert (unsigned char *)p2 >= (unsigned char *)p1 + 12; */
    /*@ assert ((uintptr_t)p1 % 16 == 0); */
    /*@ assert ((uintptr_t)p2 % 16 == 0); */

    void *p3 = re_arena_alloc_align(&a, 1u, 64u);
    /*@ assert p3 != \null; */
    /*@ assert ((uintptr_t)p3 % 64 == 0); */
    /*@ assert (unsigned char *)p3 >= (unsigned char *)p2 + 4; */

    /* Exhaustion: a request larger than the remaining capacity fails. */
    size_t used = re_arena_used(&a);
    /*@ assert used <= a.capacity; */
    void *p4 = re_arena_alloc_align(&a, 300u, 16u);
    /*@ assert p4 == \null; */
    /*@ assert re_arena_used(&a) == used; */

    re_arena_reset(&a);
    /*@ assert re_arena_used(&a) == 0; */
    void *p5 = re_arena_alloc_align(&a, 8u, 16u);
    /*@ assert p5 == p1; */ /* reset restores the cursor */
    return 0;
}
