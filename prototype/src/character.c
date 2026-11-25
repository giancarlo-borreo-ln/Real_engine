/**
 * @file character.c
 * @brief Procedural hooded-assassin animation over the platformer body.
 *
 * The rig is sampled, not simulated: everything is derived from the
 * player's ReBody velocity/position, the run distance, air time and the
 * landing/roll timers. That keeps animation deterministic and free of
 * extra state while still giving run, leap, fall, landing and roll
 * poses. All rotations are around Z (the side-view plane); facing left
 * is handled by mirroring local X and the Z angles.
 */
#include "character.h"

#include <math.h>
#include <string.h>

#include "palette.h"
#include "re_core.h"

/** Full turn in radians. */
#define TAU 6.2831853072f

/** Torso lean pivot height in the character's local frame. */
#define LEAN_PIVOT_Y 0.10f

/** @brief Per-frame animation values shared by the part emitters. */
typedef struct Rig {
    ReVec3 root;  /**< World pelvis position. */
    ReQuat body;  /**< Whole-body roll. */
    float facing; /**< +1 right, -1 left. */
    float lean;   /**< Torso lean (negative = forward). */
    float run;    /**< Normalized run speed [0, 1]. */
    float phase;  /**< Cadence phase in radians. */
    float air;    /**< Normalized air time [0, 1]. */
    float time;   /**< World time for idle breathing. */
    bool grounded;/**< Contact with a platform. */
    bool rising;  /**< Ascending in a jump. */
} Rig;

/** @brief Rotates a local (x, y) vector around Z. */
static ReVec3 rotate_z(ReVec3 v, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return re_vec3_make(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

/**
 * @brief Moves an upper-body point around the lean pivot.
 * @param v Local point.
 * @param lean Torso lean angle.
 */
static ReVec3 upper_point(ReVec3 v, float lean) {
    ReVec3 pivot = re_vec3_make(0.0f, LEAN_PIVOT_Y, 0.0f);
    return re_vec3_add(pivot, rotate_z(re_vec3_sub(v, pivot), lean));
}

/**
 * @brief Appends one box, applying facing mirror and body roll.
 * @param pose Pose being built.
 * @param rig Rig state.
 * @param local Local-space centre before mirroring.
 * @param angle Local Z rotation.
 * @param half Half extents.
 * @param material Palette id.
 */
static void emit(CharacterPose *pose, const Rig *rig, ReVec3 local, float angle,
                 ReVec3 half, uint32_t material) {
    ReVec3 mirrored;
    CharacterPart *part;

    if (pose->count >= CHARACTER_MAX_PARTS) {
        return;
    }
    mirrored = re_vec3_make(local.x * rig->facing, local.y, local.z);
    part = &pose->parts[pose->count++];
    part->center = re_vec3_add(rig->root, re_quat_rotate_vec3(rig->body, mirrored));
    part->half = half;
    part->rotation = re_quat_mul(
        rig->body,
        re_quat_from_axis_angle(re_vec3_make(0.0f, 0.0f, 1.0f), angle * rig->facing));
    part->material = material;
}

/**
 * @brief Emits a two-bone arm, with the hidden blade on the right arm.
 * @param pose Pose being built.
 * @param rig Rig state.
 * @param z Lateral offset (+right, -left).
 * @param shoulder_angle Upper-arm swing.
 * @param elbow_bend Elbow flexion.
 * @param lean Torso lean.
 */
static void emit_arm(CharacterPose *pose, const Rig *rig, float z, float shoulder_angle,
                     float elbow_bend, float lean) {
    ReVec3 shoulder = upper_point(re_vec3_make(0.0f, 0.40f, z), lean);
    float a = shoulder_angle + lean * 0.45f;
    ReVec3 elbow;
    ReVec3 forearm_centre;
    float b;

    emit(pose, rig, shoulder, a, re_vec3_make(0.055f, 0.15f, 0.055f), GAME_MAT_ROBE);
    elbow = re_vec3_add(shoulder, rotate_z(re_vec3_make(0.0f, -0.30f, 0.0f), a));
    b = a + elbow_bend;
    forearm_centre =
        re_vec3_add(elbow, rotate_z(re_vec3_make(0.0f, -0.14f, 0.0f), b));
    emit(pose, rig, forearm_centre, b, re_vec3_make(0.05f, 0.14f, 0.05f),
         GAME_MAT_CREAM);
    emit(pose, rig,
         re_vec3_add(elbow, rotate_z(re_vec3_make(0.0f, -0.28f, 0.0f), b)), b,
         re_vec3_make(0.052f, 0.045f, 0.052f), GAME_MAT_LEATHER);
    if (z > 0.0f) {
        /* Hidden blade along the forearm. */
        emit(pose, rig,
             re_vec3_add(elbow, rotate_z(re_vec3_make(0.07f, -0.27f, 0.0f), b)), b,
             re_vec3_make(0.08f, 0.018f, 0.02f), GAME_MAT_GOLD);
    }
}

/**
 * @brief Emits a two-bone leg with a forward-pointing boot.
 * @param pose Pose being built.
 * @param rig Rig state.
 * @param z Lateral offset.
 * @param thigh_angle Forward swing of the thigh.
 * @param knee_bend Flexion subtracted from the shin angle.
 */
static void emit_leg(CharacterPose *pose, const Rig *rig, float z, float thigh_angle,
                     float knee_bend) {
    ReVec3 hip = re_vec3_make(0.0f, -0.08f, z);
    ReVec3 knee;
    ReVec3 ankle;
    float shin_angle = thigh_angle - knee_bend;

    emit(pose, rig,
         re_vec3_add(hip, rotate_z(re_vec3_make(0.0f, -0.17f, 0.0f), thigh_angle)),
         thigh_angle, re_vec3_make(0.075f, 0.17f, 0.075f), GAME_MAT_DARK);
    knee = re_vec3_add(hip, rotate_z(re_vec3_make(0.0f, -0.34f, 0.0f), thigh_angle));
    emit(pose, rig,
         re_vec3_add(knee, rotate_z(re_vec3_make(0.0f, -0.18f, 0.0f), shin_angle)),
         shin_angle, re_vec3_make(0.06f, 0.18f, 0.06f), GAME_MAT_DARK);
    ankle = re_vec3_add(knee, rotate_z(re_vec3_make(0.0f, -0.36f, 0.0f), shin_angle));
    emit(pose, rig,
         re_vec3_add(ankle, rotate_z(re_vec3_make(0.05f, -0.02f, 0.0f), shin_angle)),
         shin_angle, re_vec3_make(0.11f, 0.045f, 0.075f), GAME_MAT_LEATHER);
}

/** @brief Emits robe details that lean with the torso. */
static void emit_torso(CharacterPose *pose, const Rig *rig) {
    float lean = rig->lean;

    emit(pose, rig, upper_point(re_vec3_make(0.0f, 0.22f, 0.0f), lean), lean,
         re_vec3_make(0.16f, 0.26f, 0.13f), GAME_MAT_ROBE);
    emit(pose, rig, upper_point(re_vec3_make(0.02f, 0.22f, 0.145f), lean),
         lean + 0.55f, re_vec3_make(0.035f, 0.24f, 0.016f), GAME_MAT_RED);
    emit(pose, rig, upper_point(re_vec3_make(-0.16f, 0.42f, 0.0f), lean), lean,
         re_vec3_make(0.03f, 0.19f, 0.15f), GAME_MAT_ROBE);
    emit(pose, rig, upper_point(re_vec3_make(0.0f, 0.63f, 0.0f), lean), lean,
         re_vec3_make(0.10f, 0.11f, 0.10f), GAME_MAT_SKIN);
    emit(pose, rig, upper_point(re_vec3_make(0.0f, 0.66f, -0.02f), lean), lean,
         re_vec3_make(0.145f, 0.155f, 0.145f), GAME_MAT_ROBE);
    emit(pose, rig, upper_point(re_vec3_make(-0.03f, 0.79f, -0.02f), lean),
         lean + 0.60f, re_vec3_make(0.075f, 0.075f, 0.075f), GAME_MAT_ROBE);
    emit(pose, rig, upper_point(re_vec3_make(0.10f, 0.62f, 0.0f), lean), lean,
         re_vec3_make(0.025f, 0.07f, 0.06f), GAME_MAT_DARK);
}

/** @brief Emits the belt, pelvis and hanging sash tail. */
static void emit_waist(CharacterPose *pose, const Rig *rig) {
    emit(pose, rig, re_vec3_make(0.0f, -0.11f, 0.0f), 0.0f,
         re_vec3_make(0.14f, 0.10f, 0.12f), GAME_MAT_CREAM);
    emit(pose, rig, re_vec3_make(0.0f, -0.04f, 0.0f), 0.0f,
         re_vec3_make(0.17f, 0.045f, 0.14f), GAME_MAT_LEATHER);
    emit(pose, rig, re_vec3_make(-0.06f, -0.18f, -0.14f),
         0.15f + 0.12f * sinf(rig->phase), re_vec3_make(0.03f, 0.16f, 0.05f),
         GAME_MAT_RED);
}

/** @brief Emits the two cape segments behind the torso. */
static void emit_cape(CharacterPose *pose, const Rig *rig) {
    ReVec3 attach = upper_point(re_vec3_make(-0.17f, 0.42f, 0.0f), rig->lean);
    float base = -0.22f - 0.42f * rig->run - 0.55f * rig->air +
                 sinf(rig->time * 9.0f) * 0.06f * rig->air;
    ReVec3 joint;

    emit(pose, rig, re_vec3_add(attach, rotate_z(re_vec3_make(0.0f, -0.18f, 0.0f), base)),
         base, re_vec3_make(0.03f, 0.18f, 0.16f), GAME_MAT_ROBE);
    joint = re_vec3_add(attach, rotate_z(re_vec3_make(0.0f, -0.36f, 0.0f), base));
    emit(pose, rig,
         re_vec3_add(joint, rotate_z(re_vec3_make(0.0f, -0.18f, 0.0f),
                                     base + 0.20f + 0.25f * rig->air)),
         base + 0.20f + 0.25f * rig->air, re_vec3_make(0.03f, 0.18f, 0.15f),
         GAME_MAT_ROBE);
}

void character_build_pose(const Game *game, CharacterPose *out) {
    const ReBody *p;
    Rig rig;
    float speed;
    float swing;
    float crouch;
    float leg_l;
    float leg_r;
    float knee_l;
    float knee_r;
    float arm_l;
    float arm_r;
    float bend_l;
    float bend_r;
    float bob;

    if (game == NULL || out == NULL) {
        return;
    }
    memset(out, 0, sizeof *out);
    if (game->player >= game->world.body_count) {
        return;
    }
    p = &game->world.bodies[game->player];

    speed = fabsf(p->lin_vel.x);
    rig.facing = game->facing < 0.0f ? -1.0f : 1.0f;
    rig.run = re_clamp_f32(speed / GAME_MOVE_SPEED, 0.0f, 1.0f);
    rig.phase = game->run_distance * TAU / 1.55f;
    rig.air = re_clamp_f32(game->air_time / 0.35f, 0.0f, 1.0f);
    rig.grounded = game->grounded;
    rig.rising = !game->grounded && p->lin_vel.y > 1.0f;
    rig.time = game->time;
    swing = sinf(rig.phase);
    crouch = game->land_timer > 0.0f
                 ? re_clamp_f32(game->land_timer / 0.30f, 0.0f, 1.0f)
                 : 0.0f;

    bob = rig.grounded ? fabsf(swing) * 0.035f * rig.run +
                             sinf(rig.time * 2.0f) * 0.006f
                       : 0.0f;
    bob -= 0.10f * crouch;
    rig.root = re_vec3_add(p->position, re_vec3_make(0.0f, 0.025f + bob, 0.0f));
    rig.body = re_quat_identity();
    rig.lean = -0.10f * rig.run - 0.22f * crouch;

    if (!rig.grounded) {
        rig.lean += rig.rising ? -0.08f : 0.05f;
        if (rig.rising) {
            leg_l = 0.70f;
            leg_r = 0.20f;
            knee_l = 1.15f;
            knee_r = 1.35f;
            arm_l = -1.10f;
            arm_r = -0.85f;
            bend_l = 0.95f;
            bend_r = 0.80f;
        } else {
            leg_l = -0.12f;
            leg_r = 0.30f;
            knee_l = 0.25f;
            knee_r = 0.55f;
            arm_l = 0.70f;
            arm_r = -0.35f;
            bend_l = 0.45f;
            bend_r = 0.60f;
        }
    } else {
        leg_l = swing * 0.85f * rig.run;
        leg_r = -swing * 0.85f * rig.run;
        knee_l = re_max_f32(0.0f, -swing) * 1.05f * rig.run + 0.05f;
        knee_r = re_max_f32(0.0f, swing) * 1.05f * rig.run + 0.05f;
        arm_l = -swing * 0.75f * rig.run;
        arm_r = swing * 0.75f * rig.run;
        bend_l = 0.35f + re_max_f32(0.0f, -swing) * 0.45f * rig.run;
        bend_r = 0.35f + re_max_f32(0.0f, swing) * 0.45f * rig.run;
    }

    if (crouch > 0.0f) {
        leg_l += 0.35f * crouch;
        leg_r += 0.35f * crouch;
        knee_l += 0.85f * crouch;
        knee_r += 0.85f * crouch;
        arm_l += 0.40f * crouch;
        arm_r += 0.40f * crouch;
    }

    if (game->roll_timer > 0.0f) {
        float progress = 1.0f - game->roll_timer / 0.36f;
        rig.body = re_quat_from_axis_angle(re_vec3_make(0.0f, 0.0f, 1.0f),
                                           -rig.facing * TAU * progress);
        rig.root = re_vec3_add(rig.root,
                               re_vec3_make(0.0f, -0.10f * sinf(progress * 3.1415927f),
                                            0.0f));
        leg_l = 0.90f;
        leg_r = 0.90f;
        knee_l = 1.50f;
        knee_r = 1.50f;
        arm_l = 0.90f;
        arm_r = 0.90f;
        bend_l = 1.10f;
        bend_r = 1.10f;
    }

    emit_torso(out, &rig);
    emit_waist(out, &rig);
    emit_cape(out, &rig);
    emit_arm(out, &rig, 0.19f, arm_r, bend_r, rig.lean);
    emit_arm(out, &rig, -0.19f, arm_l, bend_l, rig.lean);
    emit_leg(out, &rig, 0.10f, leg_r, knee_r);
    emit_leg(out, &rig, -0.10f, leg_l, knee_l);
}
