/**
 * @file re_engine.c
 * @brief Engine showcase: a fixed-timestep 120 Hz tick decoupled from the
 *        render rate, with an in-process GGPO-style rollback harness and
 *        deterministic packet-loss injection.
 *
 * Rollback harness scope: everything runs in one process. There is no
 * socket and no second "remote" world. Every simulated tick saves a
 * snapshot of the first `rollback_bodies` bodies plus a CRC-64 of their
 * state floats and the deterministic inputs of the tick. State packets
 * are queued with a fixed tick delay and a deterministic loss pattern;
 * when a packet is delivered the harness loads the frame back out of the
 * 64-frame ring and re-derives its checksum from the loaded bytes. A
 * mismatch means the checkpoint lost information, which is the invariant
 * the real GGPO client relies on. Fully re-simulating a frame against a
 * second world is the GGPO extension and is deliberately out of scope.
 *
 * The live world cannot be compared against a delayed snapshot (the
 * simulation has advanced by the packet delay), so the invariants checked
 * here are: loaded bytes == saved bytes, loaded body count == tracked
 * count, and stored inputs == inputs predicted from the frame number.
 *
 * Determinism: bodies are spawned from one splitmix64 stream, raycasts
 * from a second stream, and inputs are a pure function of the tick.
 * Nothing reads the clock except the pacing accumulator and telemetry.
 */
#include "re_engine.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "re_core.h"
#include "re_hash.h"
#include "re_mem.h"
#include "re_proto.h"
#include "re_renderer.h"
#include "re_rollback.h"
#include "re_vec.h"
#include "re_world.h"

/** One arena block for the whole run (world, rollback and checkpoints). */
#define RE_ENGINE_ARENA_BYTES ((size_t)(256u * 1024u * 1024u))

/** Slack on top of the exact world footprint (arena alignment, rounding). */
#define RE_ENGINE_WORLD_SLACK ((size_t)(1u << 20))

/** Rollback ring length in frames. */
#define RE_ENGINE_RING_CAP 64u

/** Fixed-timestep catch-up limit per render loop iteration. */
#define RE_ENGINE_MAX_STEPS 8u

/** Maximum hits recorded per raycast (first hit matters, extras ignored). */
#define RE_ENGINE_RAY_HIT_CAP 16u

/** Orbiting point-light count. */
#define RE_ENGINE_LIGHT_COUNT 4u

/** A state packet is emitted every this many ticks. */
#define RE_ENGINE_SEND_PERIOD 30u

/** Loss pattern period: (tick % 100) < loss_pct drops the packet. */
#define RE_ENGINE_LOSS_PERIOD 100u

/** Spare body slots above body_count (statics and future inserts). */
#define RE_ENGINE_BODY_HEADROOM 16u

/** Telemetry period in nanoseconds (one line per second). */
#define RE_ENGINE_TELEMETRY_NS UINT64_C(1000000000)

/** Camera orbit radius and look-at target height (world units). */
#define RE_ENGINE_CAMERA_RADIUS 30.0f
#define RE_ENGINE_CAMERA_TARGET_Y 2.0f

/** Pi as a float, used for the FOV in radians. */
#define RE_ENGINE_PI 3.14159265358979323846f

/**
 * @brief One queued state packet: the tick that delivers it and its frame.
 */
typedef struct RePacketSlot {
    uint32_t deliver_tick; /**< Tick at which the packet arrives. */
    uint32_t frame;        /**< Frame the packet describes. */
} RePacketSlot;

/**
 * @brief In-process rollback harness state.
 */
typedef struct ReRollbackHarness {
    ReRollback ring;          /**< Snapshot/input ring. */
    ReRollbackFrame *frames;  /**< Ring frame headers (arena). */
    ReBodySnap *snap_storage; /**< Ring snapshot slots (arena). */
    ReBodySnap *snap_load;    /**< Delivery staging buffer (arena). */
    ReBodySnap *snap_scratch; /**< Save staging buffer (arena). */
    RePacketSlot *queue;      /**< Packet delay queue (arena). */
    uint32_t queue_cap;       /**< Queue capacity in slots. */
    uint32_t queue_head;      /**< Oldest queued packet. */
    uint32_t queue_tail;      /**< Next free slot. */
    uint32_t tracked;         /**< Bodies checkpointed per frame. */
    uint32_t delay;           /**< Delivery delay in ticks (clamped). */
    uint32_t loss_pct;        /**< Injected loss percentage [0, 100]. */
    uint64_t packets_lost;    /**< Packets dropped by loss injection. */
    uint64_t rollbacks;       /**< Delivered packets that loaded a frame. */
    uint64_t desyncs;         /**< Checksum or input mismatches. */
} ReRollbackHarness;

/**
 * @brief Engine run state; all simulation memory hangs off one arena.
 */
typedef struct ReEngineState {
    ReArena arena;               /**< Arena over arena_block. */
    unsigned char *arena_block;  /**< malloc base (released in teardown). */
    ReWorld world;               /**< Rigid-body world. */
    ReRenderer *renderer;        /**< Renderer handle (opaque). */
    ReSceneInstance *instances;  /**< Render instance array (heap). */
    uint32_t instance_cap;       /**< Render instances submitted. */
    ReRng ray_rng;               /**< Deterministic raycast stream. */
    ReRollbackHarness rb;        /**< Rollback harness. */
    float dt;                    /**< Fixed timestep in seconds. */
    uint64_t ticks;              /**< Simulated ticks. */
    uint64_t frames;             /**< Rendered frames. */
    uint64_t tick_ns_total;      /**< Summed re_world_step durations. */
    uint64_t tick_ns_max;        /**< Worst re_world_step duration. */
    uint64_t frame_ns_total;     /**< Summed render durations. */
    uint64_t raycasts;           /**< Rays cast. */
    uint64_t ray_hits;           /**< Rays that hit at least one body. */
    uint64_t telemetry_ns;       /**< Timestamp of the last telemetry line. */
    uint64_t telemetry_ticks;    /**< Tick count of the last telemetry line. */
    ReWorldStats last_stats;     /**< Counters of the last simulated tick. */
} ReEngineState;

/* --------------------------- state checksums ------------------------- */

/**
 * @brief Folds count floats into a running CRC as raw IEEE-754 bytes.
 * @param crc Running CRC register.
 * @param values First float of a contiguous run.
 * @param count Floats in the run.
 * @return Updated CRC register.
 */
static uint64_t crc_floats(uint64_t crc, const float *values, uint32_t count)
{
    const unsigned char *bytes = (const unsigned char *)(const void *)values;
    const size_t n = (size_t)count * sizeof(float);

    for (size_t i = 0u; i < n; i++) {
        crc = re_crc64_update(crc, bytes[i]);
    }
    return crc;
}

/**
 * @brief Folds one snapshot's state into a running CRC.
 *
 * Hashes the 10 state floats (position, orientation, lin_vel) and skips
 * the 4 bytes of tail padding ReVec3 carries, because padding bytes are
 * indeterminate and would make the checksum build-dependent.
 * @param crc Running CRC register.
 * @param snap Snapshot to fold.
 * @return Updated CRC register.
 */
static uint64_t crc_snap_state(uint64_t crc, const ReBodySnap *snap)
{
    crc = crc_floats(crc, &snap->pos.x, 3u);
    crc = crc_floats(crc, &snap->quat.x, 4u);
    return crc_floats(crc, &snap->lin_vel.x, 3u);
}

/**
 * @brief CRC-64 of an array of body snapshots.
 * @param snaps Snapshot array.
 * @param count Snapshots to hash.
 * @return CRC-64/XZ checksum.
 */
static uint64_t snaps_checksum(const ReBodySnap *snaps, uint32_t count)
{
    uint64_t crc = RE_CRC64_INIT;

    for (uint32_t i = 0u; i < count; i++) {
        crc = crc_snap_state(crc, &snaps[i]);
    }
    return crc ^ UINT64_MAX;
}

/**
 * @brief CRC-64 of the first count bodies' state, read from the world.
 * @param w World.
 * @param count Bodies to hash (from id 0).
 * @return CRC-64/XZ checksum.
 */
static uint64_t bodies_checksum(const ReWorld *w, uint32_t count)
{
    uint64_t crc = RE_CRC64_INIT;

    for (uint32_t i = 0u; i < count; i++) {
        ReBodySnap snap;
        snap.pos = w->bodies[i].position;
        snap.quat = w->bodies[i].orientation;
        snap.lin_vel = w->bodies[i].lin_vel;
        crc = crc_snap_state(crc, &snap);
    }
    return crc ^ UINT64_MAX;
}

/* ------------------------------ inputs ------------------------------- */

/** @return Input button bits of a tick (pure function of the frame). */
static uint32_t tick_inputs(uint32_t tick)
{
    return tick & 0xFu;
}

/**
 * @brief Deterministic horizontal move of a tick (LCG high half).
 *
 * Computed without implementation-defined casts: the low 16 bits are
 * reinterpreted as two's-complement, then narrowed to int16_t.
 * @param tick Frame number.
 * @return Value in [-32768, 32767].
 */
static int16_t tick_move_x(uint32_t tick)
{
    const uint32_t raw = (tick * UINT32_C(2654435761)) >> 16u;
    int32_t value = (int32_t)(raw & UINT32_C(0xFFFF));

    if (value >= 32768) {
        value -= 65536;
    }
    return (int16_t)value;
}

/* ------------------------------- setup ------------------------------- */

/**
 * @brief Builds the world configuration for a run.
 * @param cfg Engine configuration.
 * @return World configuration with this run's capacities.
 */
static ReWorldConfig world_config(const ReEngineConfig *cfg)
{
    ReWorldConfig wc = (ReWorldConfig)RE_WORLD_CONFIG_DEFAULT;

    wc.max_bodies = cfg->body_count + RE_ENGINE_BODY_HEADROOM;
    wc.max_contacts = cfg->body_count * 8u;
    wc.max_pairs = cfg->body_count * 8u;
    wc.bvh_capacity = cfg->body_count * 4u;
    wc.solver_iterations = 4u;
    wc.sleep_time = 60u;
    return wc;
}

/**
 * @brief Exact arena footprint re_world_init will carve for a config.
 * @param wc World configuration.
 * @return Byte count, rounded up with slack.
 */
static size_t world_memory_bytes(const ReWorldConfig *wc)
{
    const size_t pairs = (size_t)wc->max_pairs;
    const size_t contacts = (size_t)wc->max_contacts;
    const size_t solver = contacts * 4u * sizeof(float);
    const size_t sort_scratch = pairs * sizeof(uint64_t);
    const size_t scratch = sort_scratch > solver ? sort_scratch : solver;
    size_t bytes = (size_t)wc->max_bodies * sizeof(ReBody);

    bytes += (size_t)wc->bvh_capacity * (sizeof(ReBvhNode) + sizeof(uint32_t));
    bytes += contacts * sizeof(ReContact) + pairs * sizeof(uint64_t);
    bytes += pairs * sizeof(ReWarmEntry) + re_align_up(scratch, 16u);
    return re_align_up(bytes + RE_ENGINE_WORLD_SLACK, 64u);
}

/**
 * @brief Allocates the single arena block and initializes the bump arena.
 * @param st Engine state.
 * @return True on success.
 */
static bool engine_alloc_arena(ReEngineState *st)
{
    st->arena_block = malloc(RE_ENGINE_ARENA_BYTES);
    if (st->arena_block == NULL) {
        re_log_error("engine: %zu MiB arena allocation failed",
                     (size_t)(RE_ENGINE_ARENA_BYTES >> 20u));
        return false;
    }
    if (!re_arena_init(&st->arena, st->arena_block, RE_ENGINE_ARENA_BYTES)) {
        re_log_error("engine: arena init failed");
        return false;
    }
    return true;
}

/**
 * @brief Carves the world storage out of the run arena and initializes it.
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @return True on success.
 */
static bool engine_init_world(ReEngineState *st, const ReEngineConfig *cfg)
{
    const ReWorldConfig wc = world_config(cfg);
    const size_t bytes = world_memory_bytes(&wc);
    void *mem = re_arena_alloc(&st->arena, bytes);

    if (mem == NULL) {
        re_log_error("engine: world needs %zu MiB, arena exhausted", bytes >> 20u);
        return false;
    }
    if (!re_world_init(&st->world, &wc, mem, bytes)) {
        re_log_error("engine: world init failed");
        return false;
    }
    return true;
}

/* ------------------------------ spawning ----------------------------- */

/**
 * @brief Uniform random unit quaternion from the deterministic stream.
 * @param rng PRNG state.
 * @return Unit quaternion.
 */
static ReQuat random_orientation(ReRng *rng)
{
    const ReVec3 axis = re_vec3_normalize(re_vec3_make(
        re_rng_f32(rng) * 2.0f - 1.0f, re_rng_f32(rng) * 2.0f - 1.0f,
        re_rng_f32(rng) * 2.0f - 1.0f));

    return re_quat_from_axis_angle(axis, re_rng_f32(rng) * 6.2831853f);
}

/** @brief Fills the inverse inertia of a unit-mass sphere. */
static ReVec3 sphere_inv_inertia(float radius)
{
    const float inv = 2.5f / (radius * radius); /* I = 2/5 m r^2, m = 1 */
    return re_vec3_make(inv, inv, inv);
}

/** @brief Fills the inverse inertia of a unit-mass box (local axes). */
static ReVec3 box_inv_inertia(ReVec3 half)
{
    const float ix = 3.0f / (half.y * half.y + half.z * half.z);
    const float iy = 3.0f / (half.x * half.x + half.z * half.z);
    const float iz = 3.0f / (half.x * half.x + half.y * half.y);
    return re_vec3_make(ix, iy, iz);
}

/**
 * @brief Spawns one dynamic body at a deterministic random pose.
 * @param w World.
 * @param rng PRNG state (70% spheres, 30% boxes).
 * @return True when the body was added.
 */
static bool spawn_dynamic(ReWorld *w, ReRng *rng)
{
    ReBody body;
    const float kind_roll = re_rng_f32(rng);

    memset(&body, 0, sizeof body);
    body.position = re_vec3_make(12.0f * (re_rng_f32(rng) - 0.5f),
                                 4.0f + 20.0f * re_rng_f32(rng),
                                 12.0f * (re_rng_f32(rng) - 0.5f));
    body.orientation = random_orientation(rng);
    body.inv_mass = 1.0f;
    body.friction = 0.5f;
    body.restitution = 0.2f;
    body.active = true;
    if (kind_roll < 0.7f) {
        body.kind = RE_SHAPE_SPHERE;
        body.radius = 0.3f + 0.3f * re_rng_f32(rng);
        body.inv_inertia = sphere_inv_inertia(body.radius);
    } else {
        body.kind = RE_SHAPE_BOX;
        body.half = re_vec3_make(0.25f + 0.25f * re_rng_f32(rng),
                                 0.25f + 0.25f * re_rng_f32(rng),
                                 0.25f + 0.25f * re_rng_f32(rng));
        body.inv_inertia = box_inv_inertia(body.half);
    }
    return re_world_add_body(w, &body) != RE_BODY_NULL;
}

/**
 * @brief Spawns one immovable box.
 * @param w World.
 * @param position Box center.
 * @param half Box half extents.
 * @return True when the body was added.
 */
static bool spawn_static_box(ReWorld *w, ReVec3 position, ReVec3 half)
{
    ReBody body;

    memset(&body, 0, sizeof body);
    body.position = position;
    body.orientation = re_quat_identity();
    body.inv_mass = 0.0f;
    body.friction = 0.5f;
    body.restitution = 0.5f;
    body.kind = RE_SHAPE_BOX;
    body.half = half;
    body.active = true;
    return re_world_add_body(w, &body) != RE_BODY_NULL;
}

/**
 * @brief Spawns the arena: one 40x40 floor slab at y = -1 and four walls.
 * @param w World.
 * @return True on success.
 */
static bool spawn_statics(ReWorld *w)
{
    if (!spawn_static_box(w, re_vec3_make(0.0f, -1.0f, 0.0f),
                          re_vec3_make(20.0f, 1.0f, 20.0f))) {
        return false;
    }
    if (!spawn_static_box(w, re_vec3_make(-12.0f, 6.0f, 0.0f),
                          re_vec3_make(1.0f, 6.0f, 20.0f)) ||
        !spawn_static_box(w, re_vec3_make(12.0f, 6.0f, 0.0f),
                          re_vec3_make(1.0f, 6.0f, 20.0f))) {
        return false;
    }
    if (!spawn_static_box(w, re_vec3_make(0.0f, 6.0f, -12.0f),
                          re_vec3_make(20.0f, 6.0f, 1.0f)) ||
        !spawn_static_box(w, re_vec3_make(0.0f, 6.0f, 12.0f),
                          re_vec3_make(20.0f, 6.0f, 1.0f))) {
        return false;
    }
    return true;
}

/**
 * @brief Spawns the statics and cfg->body_count dynamic bodies.
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @return True on success.
 */
static bool engine_spawn_bodies(ReEngineState *st, const ReEngineConfig *cfg)
{
    ReRng rng = re_rng_init(cfg->seed);

    if (!spawn_statics(&st->world)) {
        re_log_error("engine: static geometry insertion failed");
        return false;
    }
    for (uint32_t i = 0u; i < cfg->body_count; i++) {
        if (!spawn_dynamic(&st->world, &rng)) {
            re_log_error("engine: body %u insertion failed", i);
            return false;
        }
    }
    re_log_info("world: %u bodies (%u dynamic, 5 static), seed=0x%016llX",
                re_world_body_count(&st->world), cfg->body_count,
                (unsigned long long)cfg->seed);
    return true;
}

/* ------------------------------ rollback ----------------------------- */

/**
 * @brief Allocates the rollback ring, staging buffers and packet queue.
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @return True on success (also true when tracking is disabled).
 */
static bool engine_init_rollback(ReEngineState *st, const ReEngineConfig *cfg)
{
    ReRollbackHarness *h = &st->rb;
    const uint32_t tracked = re_min_u32(cfg->rollback_bodies,
                                        re_min_u32(RE_NET_MAX_BODIES, cfg->body_count));
    const uint32_t delay = re_min_u32(cfg->packet_delay, RE_ENGINE_RING_CAP - 2u);
    if (tracked == 0u) {
        re_log_warn("rollback: disabled (no bodies tracked)");
        return true;
    }
    h->frames = re_arena_alloc(&st->arena, (size_t)RE_ENGINE_RING_CAP * sizeof(ReRollbackFrame));
    h->snap_storage = re_arena_alloc(&st->arena,
                                     (size_t)RE_ENGINE_RING_CAP * tracked * sizeof(ReBodySnap));
    h->snap_load = re_arena_alloc(&st->arena, (size_t)tracked * sizeof(ReBodySnap));
    h->snap_scratch = re_arena_alloc(&st->arena, (size_t)tracked * sizeof(ReBodySnap));
    h->queue_cap = delay + 2u;
    h->queue = re_arena_alloc(&st->arena, (size_t)h->queue_cap * sizeof(RePacketSlot));
    if (h->frames == NULL || h->snap_storage == NULL || h->snap_load == NULL ||
        h->snap_scratch == NULL || h->queue == NULL) {
        re_log_error("engine: rollback storage allocation failed");
        return false;
    }
    h->tracked = tracked;
    h->delay = delay;
    h->loss_pct = re_min_u32(cfg->packet_loss_pct, 100u);
    re_rollback_init(&h->ring, h->frames, h->snap_storage, RE_ENGINE_RING_CAP, tracked);
    re_log_info("rollback: %u bodies, %u-frame ring, delay %u ticks, loss %u%%",
                tracked, RE_ENGINE_RING_CAP, h->delay, h->loss_pct);
    return true;
}

/**
 * @brief Copies the tracked body heads into the staging snapshots.
 * @param h Harness.
 * @param w World.
 */
static void harness_capture(const ReRollbackHarness *h, const ReWorld *w)
{
    for (uint32_t i = 0u; i < h->tracked; i++) {
        const ReBody *b = &w->bodies[i];
        h->snap_scratch[i].pos = b->position;
        h->snap_scratch[i].quat = b->orientation;
        h->snap_scratch[i].lin_vel = b->lin_vel;
    }
}

/**
 * @brief Saves the current tick's checkpoint into the ring.
 * @param h Harness.
 * @param w World.
 * @param tick Frame number.
 */
static void harness_save(ReRollbackHarness *h, const ReWorld *w, uint32_t tick)
{
    harness_capture(h, w);
    if (!re_rollback_save(&h->ring, tick, h->snap_scratch, h->tracked,
                          bodies_checksum(w, h->tracked), tick_inputs(tick),
                          tick_move_x(tick), 0)) {
        h->desyncs++; /* A rejected save breaks the checkpoint invariant. */
    }
}

/**
 * @brief Queues a state packet with the configured delay, or drops it.
 * @param h Harness.
 * @param tick Frame being sent.
 */
static void harness_send(ReRollbackHarness *h, uint32_t tick)
{
    const uint32_t next = (h->queue_tail + 1u) % h->queue_cap;

    if (tick % RE_ENGINE_SEND_PERIOD != 0u) {
        return;
    }
    if ((tick % RE_ENGINE_LOSS_PERIOD) < h->loss_pct || next == h->queue_head) {
        h->packets_lost++;
        return;
    }
    h->queue[h->queue_tail].deliver_tick = tick + h->delay;
    h->queue[h->queue_tail].frame = tick;
    h->queue_tail = next;
}

/**
 * @brief Verifies one delivered frame against its checkpoint.
 * @param h Harness.
 * @param frame Frame number carried by the packet.
 */
static void harness_consume(ReRollbackHarness *h, uint32_t frame)
{
    uint32_t count = 0u;
    uint32_t bits = 0u;
    int16_t move_x = 0;
    int16_t move_y = 0;
    uint64_t checksum = 0u;

    if (!re_rollback_load(&h->ring, frame, h->snap_load, &count, &checksum)) {
        return; /* Packet arrived after its frame left the window: dropped. */
    }
    h->rollbacks++;
    if (count != h->tracked || snaps_checksum(h->snap_load, count) != checksum) {
        h->desyncs++;
    }
    if (!re_rollback_inputs(&h->ring, frame, &bits, &move_x, &move_y) ||
        bits != tick_inputs(frame) || move_x != tick_move_x(frame) || move_y != 0) {
        h->desyncs++;
    }
}

/**
 * @brief Delivers every packet whose delay has elapsed.
 * @param h Harness.
 * @param tick Current tick.
 */
static void harness_deliver(ReRollbackHarness *h, uint32_t tick)
{
    while (h->queue_head != h->queue_tail &&
           h->queue[h->queue_head].deliver_tick <= tick) {
        const uint32_t frame = h->queue[h->queue_head].frame;
        h->queue_head = (h->queue_head + 1u) % h->queue_cap;
        harness_consume(h, frame);
    }
}

/**
 * @brief Runs one rollback step: save, send, then deliver due packets.
 * @param h Harness.
 * @param w World.
 * @param tick Current tick.
 */
static void harness_tick(ReRollbackHarness *h, const ReWorld *w, uint32_t tick)
{
    if (h->tracked == 0u) {
        return;
    }
    harness_save(h, w, tick);
    harness_send(h, tick);
    harness_deliver(h, tick);
}

/* ------------------------------ raycasts ----------------------------- */

/**
 * @brief Casts the configured number of deterministic rays into the world.
 * @param st Engine state.
 * @param rays Rays to cast this tick.
 */
static void engine_raycasts(ReEngineState *st, uint32_t rays)
{
    ReRayHit hits[RE_ENGINE_RAY_HIT_CAP];

    for (uint32_t i = 0u; i < rays; i++) {
        const ReVec3 origin = re_vec3_make(24.0f * (re_rng_f32(&st->ray_rng) - 0.5f),
                                           1.0f + 20.0f * re_rng_f32(&st->ray_rng),
                                           24.0f * (re_rng_f32(&st->ray_rng) - 0.5f));
        const ReVec3 dir = re_vec3_normalize(re_vec3_make(
            2.0f * re_rng_f32(&st->ray_rng) - 1.0f,
            2.0f * re_rng_f32(&st->ray_rng) - 1.0f,
            2.0f * re_rng_f32(&st->ray_rng) - 1.0f));
        const ReRay ray = {origin, dir};
        const uint32_t count = re_world_raycast(&st->world, ray, 100.0f, hits,
                                                RE_ENGINE_RAY_HIT_CAP);
        st->raycasts++;
        if (count > 0u) {
            st->ray_hits++;
        }
    }
}

/* ------------------------------- render ------------------------------ */

/**
 * @brief Fills one render instance from a body.
 * @param body Body to render.
 * @param out Instance to fill.
 */
static void fill_instance(const ReBody *body, ReSceneInstance *out)
{
    ReVec3 scale = re_vec3_make(1.0f, 1.0f, 1.0f);

    if (body->kind == RE_SHAPE_SPHERE) {
        /* The renderer's sphere mesh has radius 0.5, so scale by 2r. */
        scale = re_vec3_make(2.0f * body->radius, 2.0f * body->radius,
                             2.0f * body->radius);
    } else if (body->kind == RE_SHAPE_BOX) {
        scale = re_vec3_make(2.0f * body->half.x, 2.0f * body->half.y,
                             2.0f * body->half.z);
    }
    out->model = re_mat4_compose(body->position, body->orientation, scale);
    out->aabb_min_material = re_vec4_make(body->aabb.min.x, body->aabb.min.y,
                                          body->aabb.min.z,
                                          re_bits_to_f32((uint32_t)body->kind));
    out->aabb_max_flags = re_vec4_make(body->aabb.max.x, body->aabb.max.y,
                                       body->aabb.max.z,
                                       re_bits_to_f32(RE_INSTANCE_SHADOW_CASTER));
}

/**
 * @brief Builds the instance array for the frame.
 * @param w World.
 * @param out Instance storage.
 * @param cap Storage capacity.
 * @return Instances written.
 */
static uint32_t build_instances(const ReWorld *w, ReSceneInstance *out, uint32_t cap)
{
    const uint32_t count = re_min_u32(w->body_count, cap);

    for (uint32_t i = 0u; i < count; i++) {
        fill_instance(&w->bodies[i], &out[i]);
    }
    return count;
}

/**
 * @brief Orbits the camera around the arena.
 * @param cam Camera to fill.
 * @param t Simulation time in seconds.
 * @param aspect Width / height.
 */
static void fill_camera(ReSceneCamera *cam, float t, float aspect)
{
    const float angle = t * 0.1f;
    const ReVec3 eye = re_vec3_make(RE_ENGINE_CAMERA_RADIUS * sinf(angle), 12.0f,
                                    RE_ENGINE_CAMERA_RADIUS * cosf(angle));
    const ReVec3 target = re_vec3_make(0.0f, RE_ENGINE_CAMERA_TARGET_Y, 0.0f);

    cam->view = re_mat4_look_at(eye, target, re_vec3_make(0.0f, 1.0f, 0.0f));
    cam->proj = re_mat4_perspective(RE_ENGINE_PI / 3.0f, aspect, 0.1f, 500.0f);
    cam->position = eye;
    cam->far_plane = 500.0f;
}

/**
 * @brief Fills the sun and the four orbiting point lights.
 * @param scene Scene to fill.
 * @param lights Caller storage for the point lights.
 * @param t Simulation time in seconds.
 */
static void fill_lighting(ReSceneSnapshot *scene, RePointLight *lights, float t)
{
    static const ReVec3 colors[RE_ENGINE_LIGHT_COUNT] = {
        {1.0f, 0.35f, 0.25f}, {0.25f, 1.0f, 0.45f},
        {0.35f, 0.45f, 1.0f}, {1.0f, 0.8f, 0.3f}};

    scene->sun.direction = re_vec3_normalize(re_vec3_make(0.3f, -1.0f, 0.2f));
    scene->sun.color = re_vec3_make(1.0f, 0.95f, 0.9f);
    scene->sun.intensity = 3.0f;
    scene->sun.ambient = 0.03f;
    for (uint32_t i = 0u; i < RE_ENGINE_LIGHT_COUNT; i++) {
        const float angle = t * 0.4f + (float)i * (RE_ENGINE_PI * 0.5f);
        lights[i].position = re_vec3_make(14.0f * cosf(angle), 8.0f, 14.0f * sinf(angle));
        lights[i].radius = 24.0f;
        lights[i].color = colors[i];
        lights[i].intensity = 6.0f;
    }
    scene->point_lights = lights;
    scene->point_light_count = RE_ENGINE_LIGHT_COUNT;
}

/**
 * @brief Builds the frame snapshot and presents one render frame.
 * @param st Engine state.
 * @param cfg Engine configuration.
 */
static void engine_render(ReEngineState *st, const ReEngineConfig *cfg)
{
    RePointLight lights[RE_ENGINE_LIGHT_COUNT];
    ReSceneSnapshot scene;
    const float aspect = (float)cfg->width / (float)cfg->height;
    const uint64_t start = re_time_ns();

    memset(&scene, 0, sizeof scene);
    scene.instances = st->instances;
    scene.instance_count = build_instances(&st->world, st->instances, st->instance_cap);
    fill_camera(&scene.camera, st->world.time, aspect);
    fill_lighting(&scene, lights, st->world.time);
    scene.clear_color = re_vec3_make(0.05f, 0.06f, 0.1f);
    (void)re_renderer_render(st->renderer, &scene);
    st->frame_ns_total += re_time_ns() - start;
    st->frames++;
}

/* ----------------------------- telemetry ----------------------------- */

/**
 * @brief Prints one telemetry line with running averages.
 * @param st Engine state.
 */
static void log_telemetry(const ReEngineState *st)
{
    const double avg_tick = st->ticks != 0u
                                ? (double)st->tick_ns_total / (double)st->ticks / 1.0e6
                                : 0.0;
    const double avg_frame = st->frames != 0u
                                 ? (double)st->frame_ns_total / (double)st->frames / 1.0e6
                                 : 0.0;

    re_log_info("t=%6.1fs tick=%.3fms frame=%.2fms pairs=%u contacts=%u awake=%u "
                "ray_hits=%llu desync=%llu",
                (double)st->world.time, avg_tick, avg_frame,
                st->last_stats.broadphase_pairs, st->last_stats.contacts,
                st->last_stats.awake_bodies, (unsigned long long)st->ray_hits,
                (unsigned long long)st->rb.desyncs);
}

/**
 * @brief Emits telemetry at most once per second.
 * @param st Engine state.
 */
static void engine_telemetry(ReEngineState *st)
{
    const uint64_t now = re_time_ns();

    if (now - st->telemetry_ns < RE_ENGINE_TELEMETRY_NS) {
        return;
    }
    st->telemetry_ns = now;
    st->telemetry_ticks = st->ticks;
    log_telemetry(st);
}

/* ------------------------------ main loop ---------------------------- */

/**
 * @brief Simulates one fixed tick: physics, raycasts, rollback checkpoint.
 * @param st Engine state.
 * @param cfg Engine configuration.
 */
static void engine_tick(ReEngineState *st, const ReEngineConfig *cfg)
{
    const uint64_t start = re_time_ns();
    uint64_t duration;

    re_world_step(&st->world, st->dt);
    duration = re_time_ns() - start;
    st->tick_ns_total += duration;
    if (duration > st->tick_ns_max) {
        st->tick_ns_max = duration;
    }
    re_world_stats(&st->world, &st->last_stats);
    engine_raycasts(st, cfg->rays_per_tick);
    harness_tick(&st->rb, &st->world, (uint32_t)st->ticks);
    st->ticks++;
}

/**
 * @brief True when the configured simulation duration has elapsed.
 * @param st Engine state.
 * @param cfg Engine configuration.
 */
static bool sim_time_up(const ReEngineState *st, const ReEngineConfig *cfg)
{
    return cfg->max_seconds > 0.0 && (double)st->world.time >= cfg->max_seconds;
}

/**
 * @brief True when the run is over.
 * @param st Engine state.
 * @param cfg Engine configuration.
 */
static bool engine_done(const ReEngineState *st, const ReEngineConfig *cfg)
{
    return re_renderer_should_close(st->renderer) || sim_time_up(st, cfg);
}

/**
 * @brief Runs catch-up ticks until the accumulator or the step cap is hit.
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @param tick_dt_ns Nanoseconds per tick.
 * @param accumulator Remaining unsimulated time (updated in place).
 */
static void engine_catch_up(ReEngineState *st, const ReEngineConfig *cfg,
                            uint64_t tick_dt_ns, uint64_t *accumulator)
{
    uint32_t steps = 0u;

    while (*accumulator >= tick_dt_ns && steps < RE_ENGINE_MAX_STEPS &&
           !sim_time_up(st, cfg)) {
        engine_tick(st, cfg);
        *accumulator -= tick_dt_ns;
        steps++;
    }
}

/**
 * @brief Fixed-timestep accumulator loop with one render per iteration.
 * @param st Engine state.
 * @param cfg Engine configuration.
 */
static void engine_loop(ReEngineState *st, const ReEngineConfig *cfg)
{
    const uint64_t tick_dt_ns = RE_ENGINE_TELEMETRY_NS / (uint64_t)cfg->tick_hz;
    const uint64_t max_acc = tick_dt_ns * RE_ENGINE_MAX_STEPS;
    uint64_t accumulator = 0u;
    uint64_t last = re_time_ns();

    st->telemetry_ns = last;
    st->telemetry_ticks = UINT64_MAX; /* forces a closing line on short runs */
    for (;;) {
        const uint64_t now = re_time_ns();

        accumulator += now - last;
        last = now;
        if (accumulator > max_acc) {
            accumulator = max_acc;
        }
        engine_catch_up(st, cfg, tick_dt_ns, &accumulator);
        engine_render(st, cfg);
        engine_telemetry(st);
        if (engine_done(st, cfg)) {
            break;
        }
    }
    if (st->telemetry_ticks != st->ticks) {
        log_telemetry(st);
    }
}

/* --------------------------- setup / teardown ------------------------ */

/**
 * @brief Creates the renderer for the run.
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @return True on success.
 */
static bool engine_init_renderer(ReEngineState *st, const ReEngineConfig *cfg)
{
    const ReRenderConfig rc = {.windowed = cfg->windowed, .width = cfg->width,
                               .height = cfg->height, .title = "Real Engine"};

    st->renderer = re_renderer_create(&rc);
    if (st->renderer == NULL) {
        re_log_error("engine: renderer creation failed");
        return false;
    }
    return true;
}

/**
 * @brief Allocates the render instance array (heap, outside the arena).
 * @param st Engine state.
 * @param cfg Engine configuration.
 * @return True on success.
 */
static bool engine_alloc_instances(ReEngineState *st, const ReEngineConfig *cfg)
{
    st->instance_cap = re_min_u32(cfg->body_count, RE_MAX_INSTANCES);
    if (st->instance_cap < cfg->body_count) {
        re_log_warn("render: %u bodies exceed RE_MAX_INSTANCES, drawing the first %u",
                    cfg->body_count, st->instance_cap);
    }
    st->instances = calloc(st->instance_cap, sizeof(ReSceneInstance));
    if (st->instances == NULL) {
        re_log_error("engine: instance array allocation failed");
        return false;
    }
    return true;
}

/**
 * @brief Builds every subsystem the loop needs.
 * @param st Engine state (zeroed).
 * @param cfg Engine configuration.
 * @return True on success.
 */
static bool engine_setup(ReEngineState *st, const ReEngineConfig *cfg)
{
    if (!engine_alloc_arena(st)) {
        return false;
    }
    if (!engine_init_world(st, cfg) || !engine_spawn_bodies(st, cfg)) {
        return false;
    }
    if (!engine_init_renderer(st, cfg) || !engine_init_rollback(st, cfg)) {
        return false;
    }
    return engine_alloc_instances(st, cfg);
}

/**
 * @brief Releases the renderer, the instance array and the arena block.
 * @param st Engine state.
 */
static void engine_teardown(ReEngineState *st)
{
    if (st->renderer != NULL) {
        re_renderer_destroy(st->renderer);
        st->renderer = NULL;
    }
    free(st->instances);
    st->instances = NULL;
    free(st->arena_block);
    st->arena_block = NULL;
}

/**
 * @brief Publishes the run statistics, including the whole-world checksum.
 * @param st Engine state.
 * @param stats Receives the counters.
 */
static void fill_stats(const ReEngineState *st, ReEngineStats *stats)
{
    stats->ticks = st->ticks;
    stats->frames = st->frames;
    stats->avg_tick_ms = st->ticks != 0u
                             ? (double)st->tick_ns_total / (double)st->ticks / 1.0e6
                             : 0.0;
    stats->max_tick_ms = (double)st->tick_ns_max / 1.0e6;
    stats->avg_frame_ms = st->frames != 0u
                              ? (double)st->frame_ns_total / (double)st->frames / 1.0e6
                              : 0.0;
    stats->broadphase_pairs = st->last_stats.broadphase_pairs;
    stats->contacts = st->last_stats.contacts;
    stats->awake_bodies = st->last_stats.awake_bodies;
    stats->raycasts = st->raycasts;
    stats->ray_hits = st->ray_hits;
    stats->packets_lost = st->rb.packets_lost;
    stats->rollbacks = st->rb.rollbacks;
    stats->desyncs = st->rb.desyncs;
    stats->state_checksum = bodies_checksum(&st->world, st->world.body_count);
}

/**
 * @brief Validates a configuration before anything is allocated.
 * @param cfg Configuration to check.
 * @return True when the run can start.
 */
static bool config_valid(const ReEngineConfig *cfg)
{
    return cfg->body_count >= 1u && cfg->body_count <= UINT32_MAX / 8u &&
           cfg->tick_hz >= 1u && cfg->max_seconds >= 0.0 &&
           cfg->width >= 1u && cfg->height >= 1u;
}

/**
 * @brief Runs the engine showcase.
 *
 * Allocates a 256 MiB arena, builds the world, spawns the deterministic
 * body mix, executes the fixed-timestep loop with the rollback harness
 * and reports the final state checksum in stats.
 *
 * @param cfg Configuration (non-NULL, body_count >= 1, tick_hz >= 1).
 * @param stats Receives the final statistics.
 * @return 0 on success, 1 on fatal error.
 */
int re_engine_run(const ReEngineConfig *cfg, ReEngineStats *stats)
{
    ReEngineState st;

    if (cfg == NULL || stats == NULL) {
        return 1;
    }
    memset(stats, 0, sizeof *stats);
    if (!config_valid(cfg)) {
        re_log_error("engine: invalid configuration");
        return 1;
    }
    memset(&st, 0, sizeof st);
    st.dt = 1.0f / (float)cfg->tick_hz;
    st.ray_rng = re_rng_init(cfg->seed ^ UINT64_C(0x9E3779B97F4A7C15));
    if (!engine_setup(&st, cfg)) {
        engine_teardown(&st);
        return 1;
    }
    engine_loop(&st, cfg);
    fill_stats(&st, stats);
    engine_teardown(&st);
    return 0;
}
