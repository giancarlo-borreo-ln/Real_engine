/**
 * @file re_engine.h
 * @brief Main loop: fixed-timestep simulation tick (120 Hz default)
 *        decoupled from the variable-rate render tick, plus the
 *        10,000-body showcase with in-process GGPO rollback and
 *        packet-loss injection.
 */
#ifndef RE_ENGINE_H
#define RE_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Engine configuration.
 */
typedef struct ReEngineConfig {
    uint32_t body_count;      /**< Bodies spawned (default 10000). */
    uint32_t tick_hz;         /**< Simulation frequency (default 120). */
    double max_seconds;       /**< Run length; 0 = until window close. */
    bool windowed;            /**< Open a window and render. */
    uint32_t width;           /**< Window width. */
    uint32_t height;          /**< Window height. */
    uint32_t rollback_bodies; /**< Bodies tracked by rollback (<= 128). */
    uint32_t packet_loss_pct; /**< Simulated loss percentage [0, 100]. */
    uint32_t packet_delay;    /**< Simulated latency in ticks. */
    uint32_t rays_per_tick;   /**< Spatial raycasts per tick. */
    uint64_t seed;            /**< Simulation seed (deterministic). */
} ReEngineConfig;

/** Default configuration: 10k bodies, 120 Hz, headless 10 s. */
#define RE_ENGINE_CONFIG_DEFAULT \
    { 10000u, 120u, 10.0, false, 1280u, 720u, 64u, 5u, 3u, 64u, UINT64_C(0x5245414C454E47) }

/**
 * @brief Aggregated run statistics.
 */
typedef struct ReEngineStats {
    uint64_t ticks;           /**< Simulation ticks executed. */
    uint64_t frames;          /**< Render frames presented. */
    double avg_tick_ms;       /**< Mean tick duration. */
    double max_tick_ms;       /**< Worst tick duration. */
    double avg_frame_ms;      /**< Mean render frame duration. */
    uint32_t broadphase_pairs;/**< Pairs from the last tick. */
    uint32_t contacts;        /**< Contacts from the last tick. */
    uint32_t awake_bodies;    /**< Awake bodies from the last tick. */
    uint64_t raycasts;        /**< Raycasts issued. */
    uint64_t ray_hits;        /**< Raycast hits. */
    uint64_t packets_lost;    /**< Packets dropped by loss injection. */
    uint64_t rollbacks;       /**< Rollback re-simulations performed. */
    uint64_t desyncs;         /**< Checksum mismatches (must stay 0). */
    uint64_t state_checksum;  /**< CRC-64 of the final simulation state. */
} ReEngineStats;

/**
 * @brief Runs the engine: builds the world, spawns bodies, loops at
 *        tick_hz with a fixed-timestep accumulator, renders at the
 *        display rate, runs the rollback harness, and reports stats.
 * @param cfg Configuration.
 * @param stats Receives the final statistics.
 * @return 0 on success, non-zero on fatal error.
 */
int re_engine_run(const ReEngineConfig *cfg, ReEngineStats *stats);

#endif /* RE_ENGINE_H */
