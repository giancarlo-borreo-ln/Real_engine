/**
 * @file game.h
 * @brief Single-level 2.5D platformer simulation built on ReWorld.
 *
 * Gameplay happens on the z = 0 plane: the player runs along X, jumps
 * along Y and is locked in Z. The physics world is the engine's
 * deterministic rigid-body world, so the same input sequence always
 * produces the same trajectory on a given binary.
 *
 * Ownership: game_init stores the world inside the caller's memory
 * block and keeps no heap allocation of its own. game_reset only
 * rewinds the mutable gameplay state; the static level bodies stay
 * untouched.
 */
#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "re_world.h"

/** Maximum static platforms in a level. */
#define GAME_PLATFORM_CAP 32u

/** Maximum collectible coins in a level. */
#define GAME_COIN_CAP 32u

/** Simulation frequency of the game (Hz). */
#define GAME_TICK_HZ 120u

/** Player capsule radius (metres). */
#define GAME_PLAYER_RADIUS 0.35f

/** Player capsule half length of the segment (metres). */
#define GAME_PLAYER_HALF 0.45f

/** Player capsule total half height (half length + radius). */
#define GAME_PLAYER_HALF_HEIGHT (GAME_PLAYER_HALF + GAME_PLAYER_RADIUS)

/** Horizontal run speed (metres per second). */
#define GAME_MOVE_SPEED 6.5f

/** Ground acceleration (metres per second squared). */
#define GAME_GROUND_ACCEL 70.0f

/** Air acceleration (metres per second squared). */
#define GAME_AIR_ACCEL 32.0f

/** Initial jump speed (metres per second). */
#define GAME_JUMP_SPEED 9.6f

/** Extra downward acceleration while jump is released (jump cut). */
#define GAME_JUMP_CUT 45.0f

/** Maximum fall speed (metres per second). */
#define GAME_MAX_FALL 30.0f

/** Grace period after leaving a ledge during which a jump still works. */
#define GAME_COYOTE_TIME 0.09f

/** How long a jump press is remembered before landing. */
#define GAME_JUMP_BUFFER 0.10f

/** Height below which the player is considered fallen out of the level. */
#define GAME_KILL_Y (-9.0f)

/** Distance at which a coin is collected. */
#define GAME_COIN_RADIUS 0.95f

/** Distance at which the finish pillar triggers. */
#define GAME_GOAL_RADIUS 1.30f

/**
 * @brief One static platform (an axis-aligned box).
 */
typedef struct GamePlatform {
    float cx; /**< Centre X. */
    float cy; /**< Centre Y. */
    float cz; /**< Centre Z. */
    float hx; /**< Half extent X. */
    float hy; /**< Half extent Y. */
    float hz; /**< Half extent Z. */
} GamePlatform;

/**
 * @brief One collectible coin (purely gameplay data, no physics body).
 */
typedef struct GameCoin {
    float x;         /**< World X. */
    float y;         /**< World Y. */
    float z;         /**< World Z. */
    bool collected;  /**< True once picked up. */
} GameCoin;

/**
 * @brief Input for one fixed tick.
 */
typedef struct GameInput {
    float move;        /**< -1 (left) .. +1 (right). */
    bool jump_pressed; /**< Rising edge of the jump key this frame. */
    bool jump_held;    /**< Jump key currently held. */
} GameInput;

/**
 * @brief Full mutable game state.
 */
typedef struct Game {
    ReWorld world;                 /**< Engine physics world. */
    uint32_t player;               /**< Player body id. */
    uint32_t platform_count;       /**< Static level bodies. */
    GamePlatform platforms[GAME_PLATFORM_CAP]; /**< Level geometry. */
    uint32_t coin_count;           /**< Coins in the level. */
    GameCoin coins[GAME_COIN_CAP]; /**< Coin data. */
    float spawn_x;                 /**< Respawn point X. */
    float spawn_y;                 /**< Respawn point Y. */
    float goal_x;                  /**< Finish pillar X. */
    float goal_y;                  /**< Finish pillar Y. */
    bool grounded;                 /**< Player supported by a contact. */
    bool won;                      /**< Finish reached. */
    float coyote;                  /**< Remaining coyote time. */
    float jump_buffer;             /**< Remaining buffered jump time. */
    float cam_x;                   /**< Smoothed camera focus X. */
    float cam_y;                   /**< Smoothed camera focus Y. */
    float time;                    /**< Simulated seconds since reset. */
    float run_distance;            /**< Distance run since reset (metres). */
    float air_time;                /**< Time since leaving the ground. */
    float land_timer;              /**< Remaining landing-crouch time. */
    float roll_timer;              /**< Remaining landing-roll time. */
    float pre_step_vy;             /**< Vertical speed before the last step. */
    float facing;                  /**< +1 faces right, -1 faces left. */
    bool was_grounded;             /**< Grounded state of the previous tick. */
    uint32_t coins_collected;      /**< Collected coin count. */
    uint32_t deaths;               /**< Falls since the last reset. */
} Game;

/**
 * @brief Builds the level and spawns the player.
 * @param g Game state to initialise.
 * @param mem Backing memory for the physics world.
 * @param mem_size Size of mem in bytes (at least a few MiB).
 * @return True on success; false on NULL arguments or world init failure.
 */
bool game_init(Game *g, void *mem, size_t mem_size);

/**
 * @brief Rewinds the player, coins, timers and score to the spawn state.
 * @param g Game to reset.
 */
void game_reset(Game *g);

/**
 * @brief Advances the game by one fixed tick.
 * @param g Game state.
 * @param in Input for this tick.
 * @param dt Timestep in seconds (the game is tuned for 1/120).
 */
void game_step(Game *g, const GameInput *in, float dt);

/**
 * @brief Convenience wrapper around the physics world step count.
 * @return Simulated seconds since the last reset.
 */
float game_time(const Game *g);

/**
 * @brief Drives the player to the right and jumps whenever grounded.
 *
 * Used by the headless demo and by the integration test. No randomness,
 * so the resulting run is reproducible.
 * @param g Current game state.
 * @param out Receives the autopilot input.
 */
void game_autopilot(const Game *g, GameInput *out);

#endif /* GAME_H */
