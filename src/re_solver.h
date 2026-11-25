/**
 * @file re_solver.h
 * @brief Projected Gauss-Seidel sequential impulse solver with warm
 *        starting and Baumgarte stabilization.
 *
 * Solves normal impulses clamped to [0, +inf) (no sticking) and two
 * friction tangents clamped to +-mu * accumulated normal impulse.
 * Warm-start impulses persist per body pair in the world's cache and
 * are applied before the iteration loop, which makes resting stacks
 * stable. Iteration order is by contact index; contacts are sorted by
 * body id in the world, so the solve is deterministic for a given
 * contact set.
 */
#ifndef RE_SOLVER_H
#define RE_SOLVER_H

#include <stdint.h>

#include "re_world.h"

/**
 * @brief Solves one tick's contacts and writes velocities back.
 *
 * Also maintains the world's warm-start cache (sorted by pair key):
 * entries for pairs present this tick are carried over, stale entries
 * are dropped.
 *
 * @param w World.
 * @param contacts Contact array.
 * @param contact_count Number of contacts.
 * @param iterations PGS iterations.
 */
void re_solver_solve(ReWorld *w, ReContact *contacts, uint32_t contact_count,
                     uint32_t iterations);

/**
 * @brief Wakes bodies involved in contacts with enough impulse.
 *
 * A body whose accumulated impulse exceeded its mass times gravity
 * gets its sleep timer reset and becomes active again.
 * @param w World.
 * @param contacts Contact array.
 * @param contact_count Number of contacts.
 */
void re_solver_wake_bodies(ReWorld *w, const ReContact *contacts, uint32_t contact_count);

#endif /* RE_SOLVER_H */
