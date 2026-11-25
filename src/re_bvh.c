/**
 * @file re_bvh.c
 * @brief Dynamic AABB tree broadphase (Box2D b2DynamicTree design).
 *
 * The tree is stored in caller-provided arrays, so no operation here
 * allocates: node ids are recycled through an explicit free list and
 * every query traversal uses a fixed-size stack or heap.
 *
 * Depth note: insert balances the path with AVL-style rotations, so a
 * tree of capacity 32768 stays below height 100. The query stacks are
 * sized well above that bound (1024 entries); RE_ASSERT fires if a
 * caller hands us a deeper tree than documented. The ray traversal is
 * a min-heap ordered by entry distance (the header promises increasing
 * order) and holds at most one entry per leaf that the ray touches.
 */
#include "re_bvh.h"

#include <string.h>

#include "re_core.h"

/** Node ids a single query traversal may keep on its fixed stack. */
#define RE_BVH_QUERY_STACK 1024u

/** Ordering entries the ray traversal may keep on its fixed heap. */
#define RE_BVH_RAY_STACK 1024u

/** Node pairs the pair traversal may keep on its fixed stack. */
#define RE_BVH_PAIR_STACK 1024u

/** One (entry distance, node) entry of the ray traversal min-heap. */
typedef struct ReBvhRayHit {
    float t;      /**< Entry distance along the ray. */
    int32_t node; /**< Node id. */
} ReBvhRayHit;

/** One pair of subtrees still to be cross-tested by query_pairs. */
typedef struct ReBvhPair {
    int32_t a; /**< First subtree root. */
    int32_t b; /**< Second subtree root. */
} ReBvhPair;

/**
 * @brief Clears one node to the "free" state.
 * @param node Node to clear.
 */
static void reset_node(ReBvhNode *node) {
    memset(node, 0, sizeof *node);
    node->parent = RE_BVH_NULL;
    node->left = RE_BVH_NULL;
    node->right = RE_BVH_NULL;
}

/**
 * @return True when the node is a leaf (the left child is absent).
 */
static bool is_leaf(const ReBvh *bvh, int32_t id) {
    return bvh->nodes[id].left == RE_BVH_NULL;
}

/**
 * @brief Adds the fat margin to a box, never shrinking it.
 * @return Box expanded by the tree margin.
 */
static ReAabb fatten(ReAabb box, float margin) {
    float m = margin > 0.0f ? margin : 0.0f;

    return re_aabb_expand(box, m);
}

/**
 * @return Fat AABB enclosing the two nodes' boxes.
 */
static ReAabb node_union(const ReBvh *bvh, int32_t a, int32_t b) {
    return fatten(re_aabb_merge(bvh->nodes[a].aabb, bvh->nodes[b].aabb), bvh->margin);
}

/**
 * @return SAH cost of enlarging child a to also hold box.
 */
static float merge_cost(ReAabb child, ReAabb box) {
    return re_aabb_surface_area(re_aabb_merge(child, box)) - re_aabb_surface_area(child);
}

/**
 * @brief Pops a free node id.
 * @return Fresh, zeroed node id, or RE_BVH_NULL when the pool is exhausted.
 */
static int32_t alloc_node(ReBvh *bvh) {
    if (bvh->free_count == 0u) {
        return RE_BVH_NULL;
    }
    bvh->free_count--;
    int32_t id = (int32_t)bvh->free_list[bvh->free_count];

    reset_node(&bvh->nodes[id]);
    bvh->node_count++;
    return id;
}

/**
 * @brief Returns a node id to the free list.
 */
static void free_node(ReBvh *bvh, int32_t id) {
    RE_ASSERT(bvh->free_count < bvh->capacity);
    reset_node(&bvh->nodes[id]);
    bvh->free_list[bvh->free_count] = (uint32_t)id;
    bvh->free_count++;
    RE_ASSERT(bvh->node_count > 0u);
    bvh->node_count--;
}

/**
 * @brief Recomputes an internal node's fat AABB and height from its children.
 */
static void refit(ReBvh *bvh, int32_t id) {
    ReBvhNode *node = &bvh->nodes[id];

    if (node->left == RE_BVH_NULL) {
        return;
    }
    uint32_t hl = bvh->nodes[node->left].height;
    uint32_t hr = bvh->nodes[node->right].height;

    node->aabb = node_union(bvh, node->left, node->right);
    node->height = (hl > hr ? hl : hr) + 1u;
}

/**
 * @brief Refits a node and every ancestor up to the root.
 */
static void refit_up(ReBvh *bvh, int32_t id) {
    while (id != RE_BVH_NULL) {
        refit(bvh, id);
        id = bvh->nodes[id].parent;
    }
}

/**
 * @brief Rewires parent/child links so id takes its parent's place.
 *
 * Handles the child transfer in either direction and fixes the grand
 * parent's child pointer (or the root). AABBs and heights are left to
 * the caller.
 */
static void rotate_links(ReBvh *bvh, int32_t id, int32_t parent, int32_t grand) {
    ReBvhNode *nodes = bvh->nodes;

    if (nodes[parent].left == id) {
        nodes[parent].left = nodes[id].right;
        if (nodes[id].right != RE_BVH_NULL) {
            nodes[nodes[id].right].parent = parent;
        }
        nodes[id].right = parent;
    } else {
        RE_ASSERT(nodes[parent].right == id);
        nodes[parent].right = nodes[id].left;
        if (nodes[id].left != RE_BVH_NULL) {
            nodes[nodes[id].left].parent = parent;
        }
        nodes[id].left = parent;
    }
    nodes[parent].parent = id;
    nodes[id].parent = grand;
    if (grand == RE_BVH_NULL) {
        bvh->root = id;
    } else if (nodes[grand].left == parent) {
        nodes[grand].left = id;
    } else {
        nodes[grand].right = id;
    }
}

/**
 * @brief Single AVL rotation: moves id into its parent's place.
 *
 * The parent moves down into the child slot id used to occupy. Both
 * affected nodes are refitted, so the caller only has to fix the
 * ancestors of the returned subtree.
 * @param bvh Tree.
 * @param id Node to rotate up (must have a parent).
 */
static void rotate_up(ReBvh *bvh, int32_t id) {
    int32_t parent = bvh->nodes[id].parent;

    RE_ASSERT(parent != RE_BVH_NULL);
    int32_t grand = bvh->nodes[parent].parent;

    rotate_links(bvh, id, parent, grand);
    refit(bvh, parent);
    refit(bvh, id);
}

/**
 * @brief Repairs a right-heavy subtree.
 * @return New subtree root.
 */
static int32_t balance_right(ReBvh *bvh, int32_t right) {
    int32_t inner = bvh->nodes[right].left;
    int32_t outer = bvh->nodes[right].right;

    if (bvh->nodes[inner].height > bvh->nodes[outer].height) {
        /* Double rotation: the inner grandchild climbs twice. */
        rotate_up(bvh, inner);
        rotate_up(bvh, inner);
        return inner;
    }
    rotate_up(bvh, right);
    return right;
}

/**
 * @brief Repairs a left-heavy subtree.
 * @return New subtree root.
 */
static int32_t balance_left(ReBvh *bvh, int32_t left) {
    int32_t outer = bvh->nodes[left].left;
    int32_t inner = bvh->nodes[left].right;

    if (bvh->nodes[inner].height > bvh->nodes[outer].height) {
        rotate_up(bvh, inner);
        rotate_up(bvh, inner);
        return inner;
    }
    rotate_up(bvh, left);
    return left;
}

/**
 * @brief Box2D Balance: restores the AVL height invariant at one node.
 * @param bvh Tree.
 * @param id Subtree root to examine.
 * @return New subtree root (id itself when no rotation happened).
 */
static int32_t balance(ReBvh *bvh, int32_t id) {
    if (id == RE_BVH_NULL) {
        return id;
    }
    ReBvhNode *node = &bvh->nodes[id];

    if (node->left == RE_BVH_NULL || node->height < 2u) {
        return id;
    }
    uint32_t hl = bvh->nodes[node->left].height;
    uint32_t hr = bvh->nodes[node->right].height;

    if (hr > hl + 1u) {
        return balance_right(bvh, node->right);
    }
    if (hl > hr + 1u) {
        return balance_left(bvh, node->left);
    }
    return id;
}

/**
 * @brief Balances and refits from a node up to the root.
 */
static void balance_up(ReBvh *bvh, int32_t id) {
    while (id != RE_BVH_NULL) {
        id = balance(bvh, id);
        refit(bvh, id);
        id = bvh->nodes[id].parent;
    }
}

/**
 * @brief SAH descent: picks the cheaper child until a leaf is reached.
 * @return Leaf that should become the new leaf's sibling.
 */
static int32_t descend(const ReBvh *bvh, ReAabb box) {
    int32_t id = bvh->root;

    while (!is_leaf(bvh, id)) {
        int32_t left = bvh->nodes[id].left;
        int32_t right = bvh->nodes[id].right;
        float cost_left = merge_cost(bvh->nodes[left].aabb, box);
        float cost_right = merge_cost(bvh->nodes[right].aabb, box);

        id = cost_left <= cost_right ? left : right;
    }
    return id;
}

/**
 * @brief Heap push keyed by entry distance.
 */
static void ray_heap_push(ReBvhRayHit *heap, uint32_t *count, ReBvhRayHit hit) {
    uint32_t i = *count;

    (*count)++;
    while (i > 0u) {
        uint32_t parent = (i - 1u) / 2u;

        if (heap[parent].t <= hit.t) {
            break;
        }
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = hit;
}

/**
 * @brief Heap pop of the smallest entry distance.
 */
static ReBvhRayHit ray_heap_pop(ReBvhRayHit *heap, uint32_t *count) {
    ReBvhRayHit top = heap[0u];
    uint32_t n = *count - 1u;

    *count = n;
    if (n == 0u) {
        return top;
    }
    ReBvhRayHit last = heap[n];
    uint32_t i = 0u;

    for (;;) {
        uint32_t left = 2u * i + 1u;
        uint32_t right = left + 1u;
        uint32_t best = i;

        if (left < n && heap[left].t < last.t) {
            best = left;
        }
        if (right < n && heap[right].t < (best == i ? last.t : heap[best].t)) {
            best = right;
        }
        if (best == i) {
            break;
        }
        heap[i] = heap[best];
        i = best;
    }
    heap[i] = last;
    return top;
}

/**
 * @brief Pushes a child node when the ray hits its fat AABB.
 */
static void ray_push_child(const ReBvh *bvh, int32_t child, ReRay ray, float t_max,
                           ReBvhRayHit *heap, uint32_t *count) {
    float t = 0.0f;

    if (!re_aabb_ray(bvh->nodes[child].aabb, ray, t_max, &t)) {
        return;
    }
    RE_ASSERT(*count < RE_BVH_RAY_STACK);
    ray_heap_push(heap, count, (ReBvhRayHit){.t = t, .node = child});
}

/**
 * @brief Pushes a pair of subtree roots onto the pair stack.
 */
static void pair_push(ReBvhPair *stack, uint32_t *count, int32_t a, int32_t b) {
    RE_ASSERT(*count < RE_BVH_PAIR_STACK);
    stack[*count].a = a;
    stack[*count].b = b;
    (*count)++;
}

/**
 * @brief Pushes a cross pair only when the two boxes overlap.
 */
static void pair_push_overlap(const ReBvh *bvh, ReBvhPair *stack, uint32_t *count, int32_t a,
                              int32_t b) {
    if (re_aabb_intersects(bvh->nodes[a].aabb, bvh->nodes[b].aabb)) {
        pair_push(stack, count, a, b);
    }
}

/**
 * @brief Expands a (subtree, subtree) self-pair into three disjoint pairs.
 */
static void pair_expand_self(const ReBvh *bvh, ReBvhPair *stack, uint32_t *count, int32_t id) {
    int32_t left = bvh->nodes[id].left;
    int32_t right = bvh->nodes[id].right;

    pair_push(stack, count, left, left);
    pair_push(stack, count, right, right);
    pair_push_overlap(bvh, stack, count, left, right);
}

/**
 * @brief Expands a cross pair by splitting the internal side.
 */
static void pair_expand_cross(const ReBvh *bvh, ReBvhPair *stack, uint32_t *count, int32_t a,
                              int32_t b) {
    if (!is_leaf(bvh, a)) {
        pair_push_overlap(bvh, stack, count, bvh->nodes[a].left, b);
        pair_push_overlap(bvh, stack, count, bvh->nodes[a].right, b);
    } else {
        pair_push_overlap(bvh, stack, count, a, bvh->nodes[b].left);
        pair_push_overlap(bvh, stack, count, a, bvh->nodes[b].right);
    }
}

/**
 * @brief Records one packed overlap pair.
 *
 * The larger proxy goes in the high 32 bits so the caller can sort the
 * packed ids without unpacking them.
 */
static void pair_emit(const ReBvh *bvh, int32_t a, int32_t b, uint64_t *out_pairs, uint32_t cap,
                      uint32_t *total) {
    uint32_t pa = bvh->nodes[a].proxy;
    uint32_t pb = bvh->nodes[b].proxy;
    uint32_t lo = pa < pb ? pa : pb;
    uint32_t hi = pa < pb ? pb : pa;

    if (*total < cap && out_pairs != NULL) {
        out_pairs[*total] = ((uint64_t)hi << 32u) | (uint64_t)lo;
    }
    (*total)++;
}

/**
 * @brief Creates the internal node that joins an old leaf and a new leaf.
 *
 * Replaces the old leaf in its parent's child slot and returns the new
 * internal node id, or RE_BVH_NULL when the node pool is exhausted.
 */
static int32_t link_internal(ReBvh *bvh, int32_t sibling, int32_t leaf) {
    int32_t internal = alloc_node(bvh);

    if (internal == RE_BVH_NULL) {
        return RE_BVH_NULL;
    }
    int32_t old_parent = bvh->nodes[sibling].parent;

    bvh->nodes[internal].aabb = node_union(bvh, sibling, leaf);
    bvh->nodes[internal].height = 1u; /* Both children are leaves. */
    bvh->nodes[internal].parent = old_parent;
    bvh->nodes[internal].left = sibling;
    bvh->nodes[internal].right = leaf;
    bvh->nodes[sibling].parent = internal;
    bvh->nodes[leaf].parent = internal;
    if (old_parent == RE_BVH_NULL) {
        bvh->root = internal;
    } else if (bvh->nodes[old_parent].left == sibling) {
        bvh->nodes[old_parent].left = internal;
    } else {
        bvh->nodes[old_parent].right = internal;
    }
    return internal;
}

void re_bvh_init(ReBvh *bvh, ReBvhNode *nodes, uint32_t *free_list, uint32_t capacity,
                 float margin) {
    RE_ASSERT(bvh != NULL);
    bvh->nodes = nodes;
    bvh->capacity = capacity;
    bvh->node_count = 0u;
    bvh->root = RE_BVH_NULL;
    bvh->free_list = free_list;
    bvh->free_count = 0u;
    bvh->margin = margin;
    re_bvh_clear(bvh);
}

int32_t re_bvh_insert(ReBvh *bvh, ReAabb aabb, uint32_t proxy) {
    RE_ASSERT(bvh != NULL);
    ReAabb fat = fatten(aabb, bvh->margin);
    int32_t leaf = alloc_node(bvh);

    if (leaf == RE_BVH_NULL) {
        return RE_BVH_NULL;
    }
    bvh->nodes[leaf].aabb = fat;
    bvh->nodes[leaf].proxy = proxy;
    if (bvh->root == RE_BVH_NULL) {
        bvh->root = leaf;
        return leaf;
    }
    int32_t sibling = descend(bvh, fat);
    int32_t internal = link_internal(bvh, sibling, leaf);

    if (internal == RE_BVH_NULL) {
        free_node(bvh, leaf); /* Roll back: the leaf is still unlinked. */
        return RE_BVH_NULL;
    }
    balance_up(bvh, internal);
    return leaf;
}

void re_bvh_remove(ReBvh *bvh, int32_t node) {
    RE_ASSERT(bvh != NULL);
    if (node < 0 || (uint32_t)node >= bvh->capacity || !is_leaf(bvh, node)) {
        return;
    }
    if (node == bvh->root) {
        re_bvh_clear(bvh);
        return;
    }
    int32_t parent = bvh->nodes[node].parent;

    if (parent == RE_BVH_NULL) {
        return; /* Not on the tree: refuse to corrupt the free list. */
    }
    int32_t grand = bvh->nodes[parent].parent;
    int32_t sibling = bvh->nodes[parent].left == node ? bvh->nodes[parent].right
                                                     : bvh->nodes[parent].left;

    bvh->nodes[sibling].parent = grand;
    if (grand == RE_BVH_NULL) {
        bvh->root = sibling;
    } else if (bvh->nodes[grand].left == parent) {
        bvh->nodes[grand].left = sibling;
    } else {
        bvh->nodes[grand].right = sibling;
    }
    free_node(bvh, parent);
    free_node(bvh, node);
    refit_up(bvh, grand);
}

void re_bvh_update(ReBvh *bvh, int32_t node, ReAabb aabb) {
    RE_ASSERT(bvh != NULL);
    if (node < 0 || (uint32_t)node >= bvh->capacity || !is_leaf(bvh, node)) {
        return;
    }
    bvh->nodes[node].aabb = fatten(aabb, bvh->margin);
    int32_t id = node;

    while (bvh->nodes[id].parent != RE_BVH_NULL) {
        int32_t parent = bvh->nodes[id].parent;

        if (re_aabb_contains_aabb(bvh->nodes[parent].aabb, bvh->nodes[id].aabb)) {
            break; /* Path compression: the ancestor still encloses us. */
        }
        refit(bvh, parent);
        id = parent;
    }
}

uint32_t re_bvh_query_aabb(const ReBvh *bvh, ReAabb q, uint32_t *out_proxies, uint32_t cap) {
    if (bvh == NULL || bvh->root == RE_BVH_NULL) {
        return 0u;
    }
    int32_t stack[RE_BVH_QUERY_STACK];
    uint32_t count = 0u;
    uint32_t hits = 0u;

    stack[count] = bvh->root;
    count++;
    while (count > 0u) {
        count--;
        int32_t id = stack[count];
        const ReBvhNode *node = &bvh->nodes[id];

        if (!re_aabb_intersects(node->aabb, q)) {
            continue;
        }
        if (node->left == RE_BVH_NULL) {
            if (hits < cap && out_proxies != NULL) {
                out_proxies[hits] = node->proxy;
            }
            hits++;
        } else {
            RE_ASSERT(count + 2u <= RE_BVH_QUERY_STACK);
            stack[count] = node->left;
            stack[count + 1u] = node->right;
            count += 2u;
        }
    }
    return hits;
}

uint32_t re_bvh_query_pairs(const ReBvh *bvh, uint64_t *out_pairs, uint32_t cap) {
    if (bvh == NULL || bvh->root == RE_BVH_NULL || is_leaf(bvh, bvh->root)) {
        return 0u;
    }
    ReBvhPair stack[RE_BVH_PAIR_STACK];
    uint32_t count = 0u;
    uint32_t total = 0u;

    pair_push(stack, &count, bvh->root, bvh->root);
    while (count > 0u) {
        count--;
        ReBvhPair pair = stack[count];

        if (pair.a == pair.b) {
            if (is_leaf(bvh, pair.a)) {
                continue; /* A leaf cannot overlap itself. */
            }
            pair_expand_self(bvh, stack, &count, pair.a);
            continue;
        }
        if (is_leaf(bvh, pair.a) && is_leaf(bvh, pair.b)) {
            if (re_aabb_intersects(bvh->nodes[pair.a].aabb, bvh->nodes[pair.b].aabb)) {
                pair_emit(bvh, pair.a, pair.b, out_pairs, cap, &total);
            }
            continue;
        }
        pair_expand_cross(bvh, stack, &count, pair.a, pair.b);
    }
    return total;
}

bool re_bvh_raycast(const ReBvh *bvh, ReRay ray, float t_max,
                    bool (*cb)(uint32_t proxy, float t, void *ud), void *ud) {
    if (bvh == NULL || cb == NULL || bvh->root == RE_BVH_NULL) {
        return false;
    }
    ReBvhRayHit heap[RE_BVH_RAY_STACK];
    uint32_t count = 0u;
    float t = 0.0f;

    if (!re_aabb_ray(bvh->nodes[bvh->root].aabb, ray, t_max, &t)) {
        return false;
    }
    ray_heap_push(heap, &count, (ReBvhRayHit){.t = t, .node = bvh->root});
    while (count > 0u) {
        ReBvhRayHit hit = ray_heap_pop(heap, &count);
        const ReBvhNode *node = &bvh->nodes[hit.node];

        if (node->left == RE_BVH_NULL) {
            if (cb(node->proxy, hit.t, ud)) {
                return true;
            }
            continue;
        }
        ray_push_child(bvh, node->left, ray, t_max, heap, &count);
        ray_push_child(bvh, node->right, ray, t_max, heap, &count);
    }
    return false;
}

uint32_t re_bvh_height(const ReBvh *bvh) {
    if (bvh == NULL || bvh->root == RE_BVH_NULL) {
        return 0u;
    }
    return bvh->nodes[bvh->root].height;
}

uint32_t re_bvh_node_count(const ReBvh *bvh) {
    return bvh != NULL ? bvh->node_count : 0u;
}

void re_bvh_clear(ReBvh *bvh) {
    if (bvh == NULL) {
        return;
    }
    bvh->root = RE_BVH_NULL;
    bvh->node_count = 0u;
    bvh->free_count = bvh->capacity;
    for (uint32_t i = 0u; i < bvh->capacity; i++) {
        bvh->free_list[i] = i;
        reset_node(&bvh->nodes[i]);
    }
}
