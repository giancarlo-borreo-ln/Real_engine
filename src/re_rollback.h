/**
 * @file re_rollback.h
 * @brief GGPO rollback state ring: per-frame snapshots and inputs.
 *
 * The host saves every simulated frame (state snapshot, checksum and
 * the inputs that produced it) into a ring window. When a remote
 * state packet arrives late, the client loads the matching snapshot
 * and re-simulates forward applying its own stored inputs; because
 * the simulation is deterministic, the re-simulation must reproduce
 * the remote checksum exactly, and any mismatch is a desync.
 */
#ifndef RE_ROLLBACK_H
#define RE_ROLLBACK_H

#include <stdbool.h>
#include <stdint.h>

#include "re_proto.h"

/**
 * @brief One saved frame.
 */
typedef struct ReRollbackFrame {
    uint32_t frame;         /**< Frame number. */
    uint64_t checksum;      /**< CRC-64 of the full state. */
    uint32_t body_count;    /**< Bodies in snapshots. */
    ReBodySnap *snapshots;  /**< Caller-provided storage. */
    uint32_t inputs_bits;   /**< Input flags of the frame. */
    int16_t move_x;         /**< Horizontal move of the frame. */
    int16_t move_y;         /**< Vertical move of the frame. */
} ReRollbackFrame;

/**
 * @brief Rollback ring buffer over caller-provided frame storage.
 */
typedef struct ReRollback {
    ReRollbackFrame *frames; /**< Ring of capacity frames. */
    ReBodySnap *snap_storage; /**< capacity * max_bodies snapshot slots. */
    uint32_t capacity;       /**< Ring size (frames). */
    uint32_t max_bodies;     /**< Snapshots per frame. */
    uint32_t head;           /**< Index of the oldest saved frame. */
    uint32_t count;          /**< Saved frames in the window. */
} ReRollback;

/**
 * @brief Initializes the ring.
 * @param rb Ring.
 * @param frames Frame array (capacity entries).
 * @param snap_storage Snapshot array (capacity * max_bodies entries).
 * @param capacity Ring size.
 * @param max_bodies Snapshots per frame.
 */
void re_rollback_init(ReRollback *rb, ReRollbackFrame *frames,
                      ReBodySnap *snap_storage, uint32_t capacity,
                      uint32_t max_bodies);

/**
 * @brief Saves a frame (copies snapshots; evicts the oldest frame when
 *        the ring is full). Frames must be saved in strictly increasing
 *        order; a frame <= the newest saved frame is rejected.
 * @param rb Ring.
 * @param frame Frame number (must be greater than any previously saved frame).
 * @param snapshots State to copy (body_count entries, <= max_bodies).
 * @param body_count Bodies to copy.
 * @param checksum State checksum.
 * @param inputs_bits Input flags.
 * @param move_x Horizontal move.
 * @param move_y Vertical move.
 * @return False when body_count exceeds max_bodies or frame regressed.
 */
bool re_rollback_save(ReRollback *rb, uint32_t frame, const ReBodySnap *snapshots,
                      uint32_t body_count, uint64_t checksum,
                      uint32_t inputs_bits, int16_t move_x, int16_t move_y);

/**
 * @brief Loads the exact snapshot of a frame.
 * @param rb Ring.
 * @param frame Frame number.
 * @param out Receives the snapshot (caller array of max_bodies entries).
 * @param body_count Receives the body count.
 * @param checksum Receives the checksum.
 * @return False when the frame fell out of the window.
 */
bool re_rollback_load(const ReRollback *rb, uint32_t frame, ReBodySnap *out,
                      uint32_t *body_count, uint64_t *checksum);

/**
 * @brief Returns the inputs stored for a frame (prediction replay).
 * @param rb Ring.
 * @param frame Frame number.
 * @param inputs_bits Receives the input flags.
 * @param move_x Receives the horizontal move.
 * @param move_y Receives the vertical move.
 * @return False when the frame fell out of the window.
 */
bool re_rollback_inputs(const ReRollback *rb, uint32_t frame, uint32_t *inputs_bits,
                        int16_t *move_x, int16_t *move_y);

/**
 * @return Oldest frame still in the window, or 0 when empty.
 */
uint32_t re_rollback_oldest(const ReRollback *rb);

/**
 * @return Newest frame in the window, or 0 when empty.
 */
uint32_t re_rollback_newest(const ReRollback *rb);

/**
 * @return Number of frames in the window.
 */
uint32_t re_rollback_count(const ReRollback *rb);

#endif /* RE_ROLLBACK_H */
