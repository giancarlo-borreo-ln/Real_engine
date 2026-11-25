/**
 * @file palette.h
 * @brief Material ids shared by the scene builder and the PBR shader.
 *
 * Material 0 samples the engine's bindless checker texture. Every other
 * id maps to a flat palette colour in shaders/pbr_clustered.frag; the
 * character rig uses distinct ids for robe, leather, sash and skin.
 */
#ifndef PALETTE_H
#define PALETTE_H

/** Bindless checker texture. */
#define GAME_MAT_CHECKER 0u
/** White assassin robe. */
#define GAME_MAT_ROBE 1u
/** Cream under-tunic. */
#define GAME_MAT_CREAM 2u
/** Brown leather. */
#define GAME_MAT_LEATHER 3u
/** Red sash. */
#define GAME_MAT_RED 4u
/** Skin. */
#define GAME_MAT_SKIN 5u
/** Near-black cloth and boots. */
#define GAME_MAT_DARK 6u
/** Gold trim and coins. */
#define GAME_MAT_GOLD 7u
/** Green finish marker. */
#define GAME_MAT_GOAL 8u
/** Light stone platform. */
#define GAME_MAT_STONE 9u
/** Dark stone platform. */
#define GAME_MAT_STONE_DARK 10u

#endif /* PALETTE_H */
