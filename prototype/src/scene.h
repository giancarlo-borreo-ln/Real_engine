/**
 * @file scene.h
 * @brief Turns the platformer game state into a renderer scene snapshot.
 *
 * The renderer is stateless per frame: game_build_scene fills one
 * GameScene with instance records, a side-view camera, the sun and the
 * point lights that decorate coins and the finish pillar.
 */
#ifndef SCENE_H
#define SCENE_H

#include <stdint.h>

#include "game.h"
#include "re_renderer.h"

/** Maximum instances the level can submit. */
#define GAME_SCENE_MAX_INSTANCES 160u

/** Maximum point lights the level submits. */
#define GAME_SCENE_MAX_LIGHTS 40u

/**
 * @brief Everything the renderer needs for one frame of the platformer.
 */
typedef struct GameScene {
    ReSceneInstance instances[GAME_SCENE_MAX_INSTANCES]; /**< Opaque boxes. */
    uint32_t instance_count;                            /**< Used instances. */
    ReSceneCamera camera;                               /**< Side-view camera. */
    ReSceneLight sun;                                   /**< Key light. */
    RePointLight lights[GAME_SCENE_MAX_LIGHTS];         /**< Coin/goal glows. */
    uint32_t light_count;                               /**< Used lights. */
    ReVec3 clear_color;                                 /**< Sky colour. */
} GameScene;

/**
 * @brief Builds the renderable scene for the current game state.
 * @param g Game state.
 * @param width Framebuffer width in pixels.
 * @param height Framebuffer height in pixels.
 * @param out Receives the scene.
 */
void game_build_scene(const Game *g, uint32_t width, uint32_t height, GameScene *out);

#endif /* SCENE_H */
