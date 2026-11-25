/**
 * @file re_world.h
 * @brief Deterministic rigid-body world: fixed-timestep integration,
 *        broadphase via re_bvh, narrowphase via re_gjk, sequential
 *        impulse solving via re_solver.
 *
 * Bodies live in one contiguous array; removal swap-packs and fixes up
 * the swapped body's id. Simulation state is IEEE-754 float arithmetic
 * compiled with -ffp-contract=off (no FMA), which is bit-deterministic
 * for a given binary; see docs/ABI.md for the cross-platform contract.
 * Pairs and contacts are sorted by body ids every tick, so the solve
 * order never depends on scheduling.
 */
#ifndef RE_WORLD_H
#define RE_WORLD_H

#include <stdbool.h>
#include <stdint.h>

#include "re_bvh.h"
#include "re_core.h"
#include "re_gjk.h"
#include "re_mem.h"
#include "re_vec.h"

/** Body id meaning "no body". */
#define RE_BODY_NULL UINT32_MAX

/**
 * @brief Rigid body state.
 */
typedef struct ReBody {
    uint32_t id;        /**< Stable id (index in the world's array). */
    ReVec3 position;    /**< Center of mass (world). */
    ReQuat orientation; /**< Rotation (unit quaternion). */
    ReVec3 lin_vel;     /**< Linear velocity. */
    ReVec3 ang_vel;     /**< Angular velocity (world axes). */
    ReVec3 force;       /**< Force accumulator (world). */
    ReVec3 torque;      /**< Torque accumulator (world). */
    float inv_mass;     /**< 0 for static/infinite bodies. */
    ReVec3 inv_inertia; /**< Local-space diagonal inverse inertia. */
    float restitution;  /**< Coefficient of restitution [0, 1]. */
    float friction;     /**< Coulomb friction coefficient. */
    ReShapeKind kind;   /**< Shape kind. */
    float radius;       /**< Sphere/capsule radius. */
    ReVec3 half;        /**< Box half extents / capsule half length in x. */
    const ReVec3 *hull_points; /**< Hull vertices in local space. */
    uint32_t hull_count;       /**< Hull vertex count. */
    int32_t bvh_node;   /**< Broadphase leaf id or RE_BVH_NULL. */
    uint32_t sleep_timer; /**< Consecutive ticks below sleep velocity. */
    bool active;        /**< False while the body is asleep. */
    ReAabb aabb;        /**< Cached fat AABB (margin applied). */
} ReBody;

/**
 * @brief Per-pair contact constraint.
 */
typedef struct ReContact {
    ReVec3 point;   /**< Contact anchor (world). */
    ReVec3 normal;  /**< Contact normal, from B to A. */
    ReVec3 r_a;     /**< Anchor - center of A. */
    ReVec3 r_b;     /**< Anchor - center of B. */
    float depth;    /**< Penetration depth (>= 0). */
    float friction; /**< Combined friction (sqrt(fa * fb)). */
    float restitution; /**< Max of the two bodies' restitution. */
    uint32_t a;     /**< First body id. */
    uint32_t b;     /**< Second body id. */
} ReContact;

/**
 * @brief Persistent warm-start entry for one body pair.
 */
typedef struct ReWarmEntry {
    uint64_t pair;     /**< (min(a,b) << 32) | max(a,b). */
    float n_impulse;   /**< Accumulated normal impulse. */
    float t1_impulse;  /**< Accumulated tangent-1 impulse. */
    float t2_impulse;  /**< Accumulated tangent-2 impulse. */
} ReWarmEntry;

/**
 * @brief World configuration (clamped inside re_world_init).
 */
typedef struct ReWorldConfig {
    float gravity_y;        /**< Gravity along Y (negative = down). */
    uint32_t max_bodies;    /**< Body array capacity. */
    uint32_t max_contacts;  /**< Contact array capacity (per tick). */
    uint32_t max_pairs;     /**< Broadphase pair array capacity. */
    uint32_t solver_iterations; /**< PGS iterations per tick. */
    uint32_t bvh_capacity;  /**< Broadphase node capacity. */
    float broadphase_margin; /**< Fat margin added to leaf AABBs. */
    float sleep_lin_vel;    /**< Linear velocity below which a body may sleep. */
    float sleep_ang_vel;    /**< Angular velocity below which a body may sleep. */
    uint32_t sleep_time;    /**< Ticks below threshold before sleeping. */
    float damping;          /**< Linear/angular velocity damping per tick in [0,1). */
} ReWorldConfig;

/** Default world configuration (10k bodies, 120 Hz, 4 iterations). */
#define RE_WORLD_CONFIG_DEFAULT \
    { -9.81f, 16384u, 262144u, 262144u, 4u, 32768u, 0.05f, 0.05f, 0.05f, 120u, 0.0f }

/**
 * @brief Rigid-body world with arena-backed storage.
 */
typedef struct ReWorld {
    ReArena arena;          /**< Storage for everything below. */
    ReBody *bodies;         /**< Body array (max_bodies entries). */
    uint32_t body_count;    /**< Live bodies. */
    uint32_t body_cap;      /**< Body array capacity. */
    ReBvh bvh;              /**< Broadphase tree. */
    ReBvhNode *bvh_nodes;   /**< Node storage. */
    uint32_t *bvh_free;     /**< Free-list storage. */
    ReContact *contacts;    /**< This tick's contacts (arena reset per tick). */
    uint32_t contact_count; /**< Contacts this tick. */
    uint32_t contact_cap;   /**< Contact capacity. */
    ReWarmEntry *warm;      /**< Persistent warm-start cache, sorted by pair. */
    uint32_t warm_count;    /**< Warm entries. */
    uint32_t warm_cap;      /**< Warm capacity. */
    uint64_t *pairs;        /**< This tick's broadphase pairs (sorted). */
    uint32_t pair_count;    /**< Pairs this tick. */
    uint32_t pair_cap;      /**< Pair capacity. */
    ReWorldConfig cfg;      /**< Configuration. */
    float time;             /**< Accumulated simulation time (seconds). */
} ReWorld;

/**
 * @brief Per-tick statistics.
 */
typedef struct ReWorldStats {
    uint32_t broadphase_pairs; /**< Pairs tested this tick. */
    uint32_t contacts;         /**< Contacts solved this tick. */
    uint32_t awake_bodies;     /**< Bodies that participated this tick. */
} ReWorldStats;

/**
 * @brief Ray-cast hit record.
 */
typedef struct ReRayHit {
    uint32_t body;  /**< Hit body id. */
    float t;        /**< Entry distance. */
    ReVec3 point;   /**< Hit point. */
    ReVec3 normal;  /**< Surface normal. */
} ReRayHit;

/**
 * @brief Initializes a world over caller-provided memory.
 * @param w World.
 * @param cfg Configuration (clamped to the memory budget).
 * @param mem Backing memory block.
 * @param mem_size Size of mem in bytes.
 * @return True on success.
 */
bool re_world_init(ReWorld *w, const ReWorldConfig *cfg, void *mem, size_t mem_size);

/**
 * @brief Adds a body (copies init into the array).
 * @param w World.
 * @param init Initial body state (id and bvh_node are assigned).
 * @return Body id, or RE_BODY_NULL when full.
 */
uint32_t re_world_add_body(ReWorld *w, const ReBody *init);

/**
 * @brief Removes a body; the last body swaps into the hole and its id
 *        is updated (external references to the last body's id must be
 *        refreshed by re-reading the swapped slot).
 * @param w World.
 * @param id Body id.
 */
void re_world_remove_body(ReWorld *w, uint32_t id);

/**
 * @return Body pointer, or NULL for an invalid id.
 */
ReBody *re_world_body(ReWorld *w, uint32_t id);

/**
 * @brief Advances the simulation by one fixed timestep.
 *
 * Pipeline: integrate forces -> refit broadphase -> self-overlap pairs
 * (sorted by id) -> narrowphase (GJK/EPA) -> PGS solve with warm start
 * -> integrate velocities -> sleep management.
 * @param w World.
 * @param dt Timestep in seconds (> 0).
 */
void re_world_step(ReWorld *w, float dt);

/**
 * @brief Ray-casts the world; hits are sorted by distance.
 * @param w World.
 * @param ray Ray.
 * @param t_max Maximum distance.
 * @param out Hit array.
 * @param cap Hit array capacity.
 * @return Number of hits written (at most cap).
 */
uint32_t re_world_raycast(const ReWorld *w, ReRay ray, float t_max,
                          ReRayHit *out, uint32_t cap);

/**
 * @brief Fills stats with the last tick's counters.
 * @param w World.
 * @param stats Output.
 */
void re_world_stats(const ReWorld *w, ReWorldStats *stats);

/**
 * @return Number of live bodies.
 */
uint32_t re_world_body_count(const ReWorld *w);

/**
 * @brief True when two bodies overlap right now (GJK distance == 0).
 * @param w World.
 * @param a First body id.
 * @param b Second body id.
 */
bool re_world_overlap(const ReWorld *w, uint32_t a, uint32_t b);

#endif /* RE_WORLD_H */
