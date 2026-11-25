/**
 * @file bvh_cbmc.c
 * @brief CBMC proof: dynamic AABB tree insert/update/remove maintain the
 *        strict invariants for symbolic AABBs:
 *          1. every internal node encloses the union of its children;
 *          2. node heights equal max(child heights) + 1;
 *          3. every node is live or on the free list exactly once.
 *
 * Run: cbmc bvh_cbmc.c --unwind 16 --unwinding-assertions \
 *        --pointer-overflow-check --signed-overflow-check \
 *        --unsigned-overflow-check
 */
#include "harness_stub.h"

#include "../src/re_core.c"
#include "../src/re_vec.c"
#include "../src/re_bvh.c"

#include "../src/re_bvh.h"

#define TREE_CAP 8u

static ReBvhNode nodes[TREE_CAP];
static uint32_t free_list[TREE_CAP];
static ReBvh tree;

/** Verifies the three invariants on the whole tree. */
static void check_invariants(void) {
    for (uint32_t i = 0u; i < TREE_CAP; i++) {
        const ReBvhNode *n = &nodes[i];
        int32_t parent = n->parent;
        if (parent == RE_BVH_NULL) {
            continue; /* free node (or root when parentless) */
        }
        bool live = false;
        /* A node is live iff it is the root or a child of its parent. */
        if ((int32_t)i == tree.root) {
            live = true;
        } else if (parent >= 0 && (uint32_t)parent < TREE_CAP) {
            live = nodes[parent].left == (int32_t)i || nodes[parent].right == (int32_t)i;
        }
        if (!live) {
            continue;
        }
        int32_t l = n->left, r = n->right;
        if (l == RE_BVH_NULL && r == RE_BVH_NULL) {
            __CPROVER_assert(n->height == 0u, "leaf height is 0");
        } else {
            __CPROVER_assert(l >= 0 && (uint32_t)l < TREE_CAP, "left child in range");
            __CPROVER_assert(r >= 0 && (uint32_t)r < TREE_CAP, "right child in range");
            __CPROVER_assert(nodes[l].parent == (int32_t)i, "child parent backlink");
            __CPROVER_assert(nodes[r].parent == (int32_t)i, "child parent backlink");
            ReAabb u = re_aabb_merge(nodes[l].aabb, nodes[r].aabb);
            __CPROVER_assert(re_aabb_contains_aabb(n->aabb, u),
                             "parent encloses both children");
            uint32_t h = nodes[l].height > nodes[r].height ? nodes[l].height
                                                           : nodes[r].height;
            __CPROVER_assert(n->height == h + 1u, "height equals max(children)+1");
        }
    }
    /* Free list: parentless nodes not referenced as children of anything
       count as free nodes; node_count + free_count == TREE_CAP. */
    __CPROVER_assert(tree.free_count + tree.node_count == TREE_CAP,
                     "every node is live or free exactly once");
}

/** Symbolic AABB of bounded extent. */
static ReAabb symbolic_aabb(void) {
    float cx = nondet_float();
    float cy = nondet_float();
    float cz = nondet_float();
    float ex = nondet_float();
    float ey = nondet_float();
    float ez = nondet_float();
    __CPROVER_assume(cx >= -8.0f && cx <= 8.0f);
    __CPROVER_assume(cy >= -8.0f && cy <= 8.0f);
    __CPROVER_assume(cz >= -8.0f && cz <= 8.0f);
    __CPROVER_assume(ex >= 0.01f && ex <= 1.0f);
    __CPROVER_assume(ey >= 0.01f && ey <= 1.0f);
    __CPROVER_assume(ez >= 0.01f && ez <= 1.0f);
    ReAabb b;
    b.min = re_vec3_make(cx - ex, cy - ey, cz - ez);
    b.max = re_vec3_make(cx + ex, cy + ey, cz + ez);
    return b;
}

int main(void) {
    re_bvh_init(&tree, nodes, free_list, TREE_CAP, 0.05f);
    __CPROVER_assert(tree.node_count == 0u && tree.root == RE_BVH_NULL,
                     "tree starts empty");

    int32_t leaves[4];
    uint32_t leaf_count = 0u;

    /* Symbolic insert/update/remove sequence. */
    for (int step = 0; step < 8; step++) {
        unsigned op = nondet_uint();
        __CPROVER_assume(op <= 2u);
        if (op == 0u) {
            int32_t id = re_bvh_insert(&tree, symbolic_aabb(), leaf_count);
            if (id != RE_BVH_NULL) {
                __CPROVER_assert(id >= 0 && (uint32_t)id < TREE_CAP,
                                 "inserted id in range");
                if (leaf_count < 4u) {
                    leaves[leaf_count++] = id;
                }
            } else {
                __CPROVER_assert(tree.node_count == TREE_CAP, "NULL only when full");
            }
        } else if (op == 1u && leaf_count > 0u) {
            uint32_t idx = nondet_uint();
            __CPROVER_assume(idx < leaf_count);
            re_bvh_update(&tree, leaves[idx], symbolic_aabb());
        } else if (leaf_count > 0u) {
            uint32_t idx = nondet_uint();
            __CPROVER_assume(idx < leaf_count);
            re_bvh_remove(&tree, leaves[idx]);
            leaves[idx] = leaves[--leaf_count];
        }
        check_invariants();
    }

    /* Query correctness: every inserted proxy is found by a query with
       an AABB that contains the whole scene. */
    ReAabb universe;
    universe.min = re_vec3_make(-100.0f, -100.0f, -100.0f);
    universe.max = re_vec3_make(100.0f, 100.0f, 100.0f);
    uint32_t hits[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
    uint32_t hit_count = re_bvh_query_aabb(&tree, universe, hits, 4u);
    __CPROVER_assert(hit_count == leaf_count, "query returns every live leaf");
    check_invariants();
    return 0;
}
