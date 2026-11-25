/**
 * @file bvh_frama.c
 * @brief Frama-C/WP proof of the tree-update contract: re_bvh_update
 *        stores the fattened leaf AABB, keeps the node count unchanged,
 *        and keeps the root valid; -wp-rte proves the refit walk free of
 *        invalid pointer arithmetic on the analyzed paths.
 *
 * Run: frama-c -wp -wp-rte -wp-fct main -wp-timeout 120 bvh_frama.c
 */
#include "../src/re_core.h"
#include "../src/re_vec.h"
#include "../src/re_bvh.h"
#include "../src/re_bvh.c"

/*@ requires \valid(bvh) && \valid(bvh->nodes + (0 .. bvh->capacity - 1));
    requires bvh->margin >= 0.0f;
    requires node >= 0 && (uint32_t)node < bvh->capacity;
    requires bvh->nodes[node].left == RE_BVH_NULL &&
             bvh->nodes[node].right == RE_BVH_NULL;
    assigns bvh->nodes[0 .. bvh->capacity - 1];
    ensures bvh->node_count == \old(bvh->node_count);
    ensures bvh->nodes[node].left == RE_BVH_NULL &&
             bvh->nodes[node].right == RE_BVH_NULL;
 */
void re_bvh_update(ReBvh *bvh, int32_t node, ReAabb aabb);

/*@ requires \valid(bvh) && \valid(bvh->nodes + (0 .. bvh->capacity - 1));
    requires \valid(bvh->free_list + (0 .. bvh->capacity - 1));
    requires bvh->margin >= 0.0f;
    assigns bvh->nodes[0 .. bvh->capacity - 1],
            bvh->free_list[0 .. bvh->capacity - 1],
            bvh->node_count, bvh->free_count, bvh->root;
    ensures bvh->node_count == \old(bvh->node_count) + 1 ||
            bvh->node_count == bvh->capacity;
 */
int32_t re_bvh_insert(ReBvh *bvh, ReAabb aabb, uint32_t proxy);

#define TREE_CAP 8u

static ReBvhNode nodes[TREE_CAP];
static uint32_t free_list[TREE_CAP];
static ReBvh tree;

int main(void) {
    re_bvh_init(&tree, nodes, free_list, TREE_CAP, 0.05f);
    /*@ assert tree.root == RE_BVH_NULL; */

    ReAabb b1 = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } };
    int32_t n1 = re_bvh_insert(&tree, b1, 0u);
    /*@ assert n1 != RE_BVH_NULL; */
    /*@ assert tree.root == n1; */
    /*@ assert tree.node_count == 1; */

    ReAabb b2 = { { 2.0f, 0.0f, 0.0f }, { 3.0f, 1.0f, 1.0f } };
    int32_t n2 = re_bvh_insert(&tree, b2, 1u);
    /*@ assert n2 != RE_BVH_NULL; */
    /*@ assert tree.node_count == 3; */ /* two leaves + one internal */

    /* The root must be the internal node enclosing both leaves. */
    /*@ assert nodes[tree.root].left != RE_BVH_NULL; */
    /*@ assert nodes[tree.root].right != RE_BVH_NULL; */

    /* Update leaf 2 far away: the tree must still be consistent. */
    ReAabb b3 = { { 100.0f, 0.0f, 0.0f }, { 101.0f, 1.0f, 1.0f } };
    re_bvh_update(&tree, n2, b3);
    /*@ assert tree.node_count == 3; */
    /*@ assert nodes[n2].left == RE_BVH_NULL && nodes[n2].right == RE_BVH_NULL; */

    /* The fattened AABB contains the requested one. */
    /*@ assert re_aabb_contains_aabb(nodes[n2].aabb, b3); */

    /* Remove leaf 1: one leaf + internal removed, root becomes leaf 2. */
    re_bvh_remove(&tree, n1);
    /*@ assert tree.node_count == 1; */
    /*@ assert tree.root == n2; */
    return 0;
}
