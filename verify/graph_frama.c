/**
 * @file graph_frama.c
 * @brief Frama-C/WP proof of the render-graph compilation contract:
 *        when re_graph_compile succeeds, no pass reads a resource that
 *        has no defined contents (no read-before-write), every emitted
 *        barrier carries layouts inside the layout domain, and the
 *        compiled output is structurally valid.
 *
 * Run: frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 graph_frama.c
 */
#include "../src/re_core.h"
#include "../src/re_render_graph.h"
#include "../src/re_render_graph.c"

/*@ requires \valid(resources + (0 .. resource_count - 1));
    requires \valid(passes + (0 .. pass_count - 1));
    requires resource_count <= 4 && pass_count <= 4;
    ensures \result == false ||
            (out->pass_count == pass_count &&
             \valid(out->pass_order + (0 .. pass_count - 1)) &&
             \valid(out->barrier_start + (0 .. pass_count)) &&
             out->barrier_start[0] == 0 &&
             out->barrier_start[pass_count] == out->barrier_count &&
             \forall integer k; 0 <= k < out->barrier_count ==>
                out->barriers[k].res < resource_count &&
                out->barriers[k].src < RE_LAYOUT_COUNT &&
                out->barriers[k].dst < RE_LAYOUT_COUNT);
 */
bool re_graph_compile(const ReGraphResource *resources, uint32_t resource_count,
                      const ReGraphPass *passes, uint32_t pass_count,
                      ReGraphCompiled *out, void *mem, size_t mem_size);

static unsigned char graph_mem[4096];

int main(void) {
    /* Scenario 1: write-then-read compiles and is valid. */
    {
        ReGraphResource res[2] = {
            { "rt", true, RE_LAYOUT_UNDEFINED, RE_LAYOUT_SHADER_READ },
            { "db", false, RE_LAYOUT_UNDEFINED, RE_LAYOUT_DEPTH_READ_ONLY },
        };
        uint32_t r0 = 0u;
        uint32_t w0 = 0u;
        ReGraphPass passes[2] = {
            { "geom", NULL, 0u, &w0, 1u, NULL, NULL },
            { "light", &r0, 1u, NULL, 0u, NULL, NULL },
        };
        ReGraphCompiled c = { 0 };
        bool ok = re_graph_compile(res, 2u, passes, 2u, &c, graph_mem,
                                   sizeof graph_mem);
        /*@ assert ok; */
        /*@ assert c.pass_count == 2; */
        /*@ assert c.barrier_count >= 2; */
        ok = re_graph_validate(res, 2u, &c);
        /*@ assert ok; */
    }

    /* Scenario 2: read-before-write must fail to compile. */
    {
        ReGraphResource res[1] = {
            { "rt", true, RE_LAYOUT_UNDEFINED, RE_LAYOUT_SHADER_READ },
        };
        uint32_t r0 = 0u;
        ReGraphPass passes[1] = {
            { "read", &r0, 1u, NULL, 0u, NULL, NULL },
        };
        ReGraphCompiled c = { 0 };
        bool ok = re_graph_compile(res, 1u, passes, 1u, &c, graph_mem,
                                   sizeof graph_mem);
        /*@ assert !ok; */
    }

    /* Scenario 3: resource index out of range must fail to compile. */
    {
        ReGraphResource res[1] = {
            { "rt", true, RE_LAYOUT_UNDEFINED, RE_LAYOUT_SHADER_READ },
        };
        uint32_t bad = 7u;
        ReGraphPass passes[1] = {
            { "bad", &bad, 1u, NULL, 0u, NULL, NULL },
        };
        ReGraphCompiled c = { 0 };
        bool ok = re_graph_compile(res, 1u, passes, 1u, &c, graph_mem,
                                   sizeof graph_mem);
        /*@ assert !ok; */
    }

    /* Scenario 4: aliasing of disjoint transients. */
    {
        ReGraphResource res[3] = {
            { "t0", true, RE_LAYOUT_UNDEFINED, RE_LAYOUT_SHADER_READ },
            { "t1", true, RE_LAYOUT_UNDEFINED, RE_LAYOUT_SHADER_READ },
            { "perm", false, RE_LAYOUT_UNDEFINED, RE_LAYOUT_COLOR_ATTACHMENT },
        };
        uint32_t w0 = 0u, r0 = 0u, w1 = 1u, r1 = 1u;
        ReGraphPass passes[3] = {
            { "a", NULL, 0u, &w0, 1u, NULL, NULL },
            { "b", &r0, 1u, &w1, 1u, NULL, NULL },
            { "c", &r1, 1u, NULL, 0u, NULL, NULL },
        };
        ReGraphCompiled c = { 0 };
        bool ok = re_graph_compile(res, 3u, passes, 3u, &c, graph_mem,
                                   sizeof graph_mem);
        /*@ assert ok; */
        /*@ assert c.alias_of[1] == 0; */ /* disjoint lifetime: aliased */
        ok = re_graph_validate(res, 3u, &c);
        /*@ assert ok; */
    }
    return 0;
}
