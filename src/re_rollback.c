/**
 * @file re_rollback.c
 * @brief GGPO rollback ring: per-frame snapshots and inputs.
 *
 * The ring is a FIFO window over caller-provided storage. `head` is the
 * oldest live slot and slots are visited as (head + i) % capacity, so a
 * save appends at (head + count) % capacity and a full ring evicts slot
 * `head` before appending. Frame numbers are strictly increasing, which
 * makes `newest` the frame of the slot just before the append point and
 * lets any lookup fail fast once the requested frame is older than the
 * window.
 *
 * Snapshots of slot i live at snap_storage + i * max_bodies: one fixed
 * stride per slot, so saving never shifts bytes and each frame keeps its
 * own body_count.
 */
#include "re_rollback.h"

#include <string.h>

/**
 * @brief Maps a logical ring offset to a physical slot index.
 * @param rb Ring.
 * @param offset Offset from head (0 .. count-1).
 * @return Slot index in [0, capacity).
 */
static uint32_t ring_slot(const ReRollback *rb, uint32_t offset) {
    return (uint32_t)(((size_t)rb->head + (size_t)offset) % (size_t)rb->capacity);
}

/**
 * @brief Finds the live frame with the given number.
 * @param rb Ring.
 * @param frame Frame number to look for.
 * @return Slot pointer, or NULL when the frame is not in the window.
 */
static ReRollbackFrame *find_frame(const ReRollback *rb, uint32_t frame) {
    uint32_t i;

    if (rb == NULL || rb->frames == NULL || rb->capacity == 0u) {
        return NULL;
    }
    for (i = 0u; i < rb->count && i < rb->capacity; ++i) {
        ReRollbackFrame *slot = &rb->frames[ring_slot(rb, i)];

        if (slot->frame == frame) {
            return slot;
        }
    }
    return NULL;
}

void re_rollback_init(ReRollback *rb, ReRollbackFrame *frames,
                      ReBodySnap *snap_storage, uint32_t capacity,
                      uint32_t max_bodies) {
    if (rb == NULL) {
        return;
    }
    rb->frames = frames;
    rb->snap_storage = snap_storage;
    rb->capacity = capacity;
    rb->max_bodies = max_bodies;
    rb->head = 0u;
    rb->count = 0u;
}

bool re_rollback_save(ReRollback *rb, uint32_t frame, const ReBodySnap *snapshots,
                      uint32_t body_count, uint64_t checksum,
                      uint32_t inputs_bits, int16_t move_x, int16_t move_y) {
    ReRollbackFrame *slot;
    uint32_t index;

    if (rb == NULL || rb->frames == NULL || rb->capacity == 0u) {
        return false;
    }
    if (body_count > rb->max_bodies) {
        return false;
    }
    if (body_count > 0u && (snapshots == NULL || rb->snap_storage == NULL)) {
        return false;
    }
    if (rb->count > 0u && frame <= rb->frames[ring_slot(rb, rb->count - 1u)].frame) {
        return false; /* Regression: frames enter the ring in increasing order. */
    }
    if (rb->count == rb->capacity) { /* Full: drop the oldest slot. */
        rb->head = ring_slot(rb, 1u);
        rb->count -= 1u;
    }
    index = ring_slot(rb, rb->count);
    slot = &rb->frames[index];
    slot->frame = frame;
    slot->checksum = checksum;
    slot->body_count = body_count;
    slot->inputs_bits = inputs_bits;
    slot->move_x = move_x;
    slot->move_y = move_y;
    slot->snapshots = rb->max_bodies > 0u && rb->snap_storage != NULL
                          ? rb->snap_storage + (size_t)index * rb->max_bodies
                          : NULL;
    if (body_count > 0u) {
        memcpy(slot->snapshots, snapshots, (size_t)body_count * sizeof *snapshots);
    }
    rb->count += 1u;
    return true;
}

bool re_rollback_load(const ReRollback *rb, uint32_t frame, ReBodySnap *out,
                      uint32_t *body_count, uint64_t *checksum) {
    const ReRollbackFrame *slot = find_frame(rb, frame);

    if (slot == NULL || out == NULL || body_count == NULL || checksum == NULL) {
        return false;
    }
    if (slot->body_count > 0u) {
        if (slot->snapshots == NULL) {
            return false;
        }
        memcpy(out, slot->snapshots, (size_t)slot->body_count * sizeof *out);
    }
    *body_count = slot->body_count;
    *checksum = slot->checksum;
    return true;
}

bool re_rollback_inputs(const ReRollback *rb, uint32_t frame, uint32_t *inputs_bits,
                        int16_t *move_x, int16_t *move_y) {
    const ReRollbackFrame *slot = find_frame(rb, frame);

    if (slot == NULL || inputs_bits == NULL || move_x == NULL || move_y == NULL) {
        return false;
    }
    *inputs_bits = slot->inputs_bits;
    *move_x = slot->move_x;
    *move_y = slot->move_y;
    return true;
}

uint32_t re_rollback_oldest(const ReRollback *rb) {
    if (rb == NULL || rb->frames == NULL || rb->capacity == 0u || rb->count == 0u) {
        return 0u;
    }
    return rb->frames[rb->head % rb->capacity].frame;
}

uint32_t re_rollback_newest(const ReRollback *rb) {
    if (rb == NULL || rb->frames == NULL || rb->capacity == 0u || rb->count == 0u) {
        return 0u;
    }
    return rb->frames[ring_slot(rb, rb->count - 1u)].frame;
}

uint32_t re_rollback_count(const ReRollback *rb) {
    if (rb == NULL || rb->capacity == 0u) {
        return 0u;
    }
    return rb->count;
}
