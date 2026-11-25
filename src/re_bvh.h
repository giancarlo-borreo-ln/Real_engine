/**
 * @file re_bvh.h
 * @brief Dynamic AABB tree broadphase with SAH-guided insertion and
 *        AVL-style rotations (Box2D b2DynamicTree design).
 *
 * Invariants maintained by every operation and proven by the harnesses:
 *  1. Every internal node's AABB strictly contains the union of its
 *     children's AABBs (plus margin).
 *  2. Node heights equal max(child heights) + 1.
 *  3. All node indices are either live (on the tree) or on the free
 *     list; insert/remove preserve both lists.
 *
 * Node storage and the free-list array are caller-provided; the physics
 * world carves them out of its arena.
 */
#ifndef RE_BVH_H
#define RE_BVH_H

#include <stdbool.h>
#include <stdint.h>

#include "re_vec.h"

/** Node id meaning "no node". */
#define RE_BVH_NULL (-1)

/**
 * @brief One tree node (64 bytes: a 32-byte ReAabb plus five 4-byte
 *        fields, rounded up to the Aabb's 16-byte alignment).
 */
typedef struct ReBvhNode {
    ReAabb aabb;     /**< Fat AABB (with margin). */
    int32_t parent;  /**< Parent node id or RE_BVH_NULL. */
    int32_t left;    /**< Left child or RE_BVH_NULL for leaves. */
    int32_t right;   /**< Right child or RE_BVH_NULL for leaves. */
    uint32_t proxy;  /**< Opaque id stored by the caller (leaves only). */
    uint32_t height; /**< Leaf = 0; internal = max(children) + 1. */
} ReBvhNode;

/**
 * @brief Dynamic AABB tree.
 */
typedef struct ReBvh {
    ReBvhNode *nodes;    /**< Node array (capacity entries). */
    uint32_t capacity;   /**< Node array size. */
    uint32_t node_count; /**< Live nodes. */
    int32_t root;        /**< Root node id or RE_BVH_NULL. */
    uint32_t *free_list; /**< Stack of free node ids (capacity entries). */
    uint32_t free_count; /**< Free node ids in free_list. */
    float margin;        /**< Fat margin added to every leaf AABB. */
} ReBvh;

/**
 * @brief Initializes an empty tree over caller-provided storage.
 * @param bvh Tree.
 * @param nodes Node array (capacity entries).
 * @param free_list Free-list array (capacity entries).
 * @param capacity Maximum node count.
 * @param margin Fat AABB margin (predicts motion).
 */
void re_bvh_init(ReBvh *bvh, ReBvhNode *nodes, uint32_t *free_list,
                 uint32_t capacity, float margin);

/**
 * @brief Inserts a leaf and returns its node id, or RE_BVH_NULL on OOM.
 *
 * Descends the tree picking, at each level, the child whose union with
 * the new AABB minimizes surface area (SAH greedy), then balances the
 * insertion path with rotations.
 * @param bvh Tree.
 * @param aabb Leaf AABB (the margin is added inside).
 * @param proxy Opaque id stored on the leaf.
 */
int32_t re_bvh_insert(ReBvh *bvh, ReAabb aabb, uint32_t proxy);

/**
 * @brief Removes a leaf; the parent is replaced by the sibling and the
 *        affected ancestors are refit.
 * @param bvh Tree.
 * @param node Leaf node id.
 */
void re_bvh_remove(ReBvh *bvh, int32_t node);

/**
 * @brief Moves a leaf: sets its fat AABB and refits the ancestors,
 *        stopping early when an ancestor already contains the child
 *        (path compression).
 * @param bvh Tree.
 * @param node Leaf node id.
 * @param aabb New leaf AABB (margin added inside).
 */
void re_bvh_update(ReBvh *bvh, int32_t node, ReAabb aabb);

/**
 * @brief Collects the proxies of every leaf whose fat AABB overlaps q.
 * @param bvh Tree.
 * @param q Query box.
 * @param out_proxies Output array.
 * @param cap Output capacity.
 * @return Total number of hits; at most cap are written. When the
 *         return value exceeds cap, call again with a larger array.
 */
uint32_t re_bvh_query_aabb(const ReBvh *bvh, ReAabb q, uint32_t *out_proxies, uint32_t cap);

/**
 * @brief Collects all pairs of overlapping leaves (self-overlap).
 *
 * Each pair is reported once with the smaller proxy in the low 32 bits
 * of the packed id. Proxies from distinct subtrees are tested pairwise.
 * @param bvh Tree.
 * @param out_pairs Output array of packed pairs (proxy_a << 32 | proxy_b).
 * @param cap Output capacity.
 * @return Total number of pairs; at most cap are written.
 */
uint32_t re_bvh_query_pairs(const ReBvh *bvh, uint64_t *out_pairs, uint32_t cap);

/**
 * @brief Ray-casts the tree, calling cb for every hit in increasing
 *        order of entry distance. cb returns true to stop early.
 * @param bvh Tree.
 * @param ray Ray.
 * @param t_max Maximum distance to accept.
 * @param cb Hit callback (proxy, entry t).
 * @param ud User data passed to cb.
 * @return True when cb stopped the traversal (found hit).
 */
bool re_bvh_raycast(const ReBvh *bvh, ReRay ray, float t_max,
                    bool (*cb)(uint32_t proxy, float t, void *ud), void *ud);

/**
 * @return Height of the root (0 for an empty tree).
 */
uint32_t re_bvh_height(const ReBvh *bvh);

/**
 * @return Number of live nodes.
 */
uint32_t re_bvh_node_count(const ReBvh *bvh);

/**
 * @brief Empties the tree (all nodes go to the free list).
 * @param bvh Tree.
 */
void re_bvh_clear(ReBvh *bvh);

#endif /* RE_BVH_H */
