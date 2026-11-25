/**
 * @file character.h
 * @brief Procedural hooded-assassin rig for the platformer player.
 *
 * The renderer only draws boxes, so the character is a small kinematic
 * skeleton of boxes: torso, hooded head, cape, sash and two-bone arms
 * and legs. character_build_pose samples the current ReWorld body and
 * the game's animation timers into world-space parts; scene.c turns
 * those parts into renderer instances.
 */
#ifndef CHARACTER_H
#define CHARACTER_H

#include <stdint.h>

#include "game.h"
#include "re_vec.h"

/** Maximum number of boxes in one character pose. */
#define CHARACTER_MAX_PARTS 48u

/**
 * @brief One world-space box of the character.
 */
typedef struct CharacterPart {
    ReVec3 center;   /**< World centre. */
    ReVec3 half;     /**< Half extents in the part's local frame. */
    ReQuat rotation; /**< World orientation. */
    uint32_t material; /**< Palette id (see palette.h). */
} CharacterPart;

/**
 * @brief Complete character pose for one frame.
 */
typedef struct CharacterPose {
    CharacterPart parts[CHARACTER_MAX_PARTS]; /**< Emitted boxes. */
    uint32_t count;                           /**< Used parts. */
} CharacterPose;

/**
 * @brief Samples the player body and animation state into a box pose.
 * @param game Game state.
 * @param out Receives the pose (count is zeroed first).
 */
void character_build_pose(const Game *game, CharacterPose *out);

#endif /* CHARACTER_H */
