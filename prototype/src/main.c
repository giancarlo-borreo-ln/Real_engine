/**
 * @file main.c
 * @brief Windowed/headless front end for the Real Engine platformer.
 *
 * The loop is a fixed 120 Hz accumulator decoupled from presentation:
 * keyboard state is sampled once per rendered frame, but every physics
 * tick sees the same input. When no Vulkan device or window is
 * available the engine falls back to its null renderer and the demo
 * plays itself with a deterministic autopilot.
 *
 * Controls: A/D or arrows move, Space/W/Up jumps, R restarts,
 * Escape closes the window.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game.h"
#include "re_core.h"
#include "re_renderer.h"
#include "scene.h"

/** Backing memory handed to the physics world (bytes). */
#define GAME_WORLD_MEMORY (8u * 1024u * 1024u)

/** Fixed simulation step in nanoseconds. */
#define GAME_TICK_NS (UINT64_C(1000000000) / GAME_TICK_HZ)

/** Largest frame delta fed into the accumulator (nanoseconds). */
#define GAME_MAX_ACCUM_NS UINT64_C(250000000)

/** Maximum catch-up ticks per rendered frame. */
#define GAME_MAX_STEPS 8u

/** Window title. */
#define GAME_TITLE "Real Engine Platformer"

/**
 * @brief Parsed command-line options.
 */
typedef struct Options {
    bool headless;  /**< Force the no-window path. */
    bool autopilot; /**< Force the demo autopilot. */
    double seconds; /**< Run length; 0 = until the window closes. */
    uint32_t width; /**< Window width. */
    uint32_t height; /**< Window height. */
} Options;

/** @brief Prints usage to the given stream. */
static void usage(FILE *out)
{
    fprintf(out,
            "usage: platformer [options]\n"
            "  --windowed        open a window (default)\n"
            "  --headless        run without a window (autopilot demo)\n"
            "  --autopilot       let the bot play, even in a window\n"
            "  --seconds T       stop after T seconds (0 = until close)\n"
            "  --width W         window width (default 1280)\n"
            "  --height H        window height (default 720)\n"
            "  --help            print this help and exit\n"
            "controls: A/D or arrows move, Space/W/Up jumps, R restarts\n");
}

/** @brief Parses a base-10 unsigned value in [lo, hi]. */
static bool parse_u32(const char *text, uint32_t lo, uint32_t hi, uint32_t *out)
{
    char *end = NULL;
    unsigned long value;

    if (text == NULL || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < lo || value > hi) {
        return false;
    }
    *out = (uint32_t)value;
    return true;
}

/** @brief Parses a finite non-negative duration. */
static bool parse_seconds(const char *text, double *out)
{
    char *end = NULL;
    double value;

    if (text == NULL || *text == '\0') {
        return false;
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !(value >= 0.0)) {
        return false;
    }
    *out = value;
    return true;
}

/**
 * @brief Parses the command line.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param opts Receives the options.
 * @return 0 on success, 1 for --help, -1 on error.
 */
static int parse_args(int argc, char **argv, Options *opts)
{
    int i;

    for (i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0) {
            return 1;
        } else if (strcmp(arg, "--headless") == 0) {
            opts->headless = true;
        } else if (strcmp(arg, "--windowed") == 0) {
            opts->headless = false;
        } else if (strcmp(arg, "--autopilot") == 0) {
            opts->autopilot = true;
        } else if (i + 1 < argc && strcmp(arg, "--seconds") == 0) {
            if (!parse_seconds(argv[++i], &opts->seconds)) {
                return -1;
            }
        } else if (i + 1 < argc && strcmp(arg, "--width") == 0) {
            if (!parse_u32(argv[++i], 64u, 16384u, &opts->width)) {
                return -1;
            }
        } else if (i + 1 < argc && strcmp(arg, "--height") == 0) {
            if (!parse_u32(argv[++i], 64u, 16384u, &opts->height)) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

/**
 * @brief Samples the keyboard into a game input.
 * @param keys Renderer keyboard snapshot.
 * @param jump_prev Jump state from the previous frame.
 * @param out Receives the input.
 * @return The current jump state (to store as jump_prev next frame).
 */
static bool read_human_input(const ReInput *keys, bool jump_prev, GameInput *out)
{
    bool jump = keys->down[RE_KEY_SPACE] || keys->down[RE_KEY_W] || keys->down[RE_KEY_UP];
    bool left = keys->down[RE_KEY_A] || keys->down[RE_KEY_LEFT];
    bool right = keys->down[RE_KEY_D] || keys->down[RE_KEY_RIGHT];

    out->move = (right ? 1.0f : 0.0f) - (left ? 1.0f : 0.0f);
    out->jump_pressed = jump && !jump_prev;
    out->jump_held = jump;
    return jump;
}

/** @brief Fills the per-frame scene snapshot from the game state. */
static void fill_snapshot(const GameScene *scene, ReSceneSnapshot *snap)
{
    snap->instances = scene->instances;
    snap->instance_count = scene->instance_count;
    snap->camera = scene->camera;
    snap->sun = scene->sun;
    snap->point_lights = scene->lights;
    snap->point_light_count = scene->light_count;
    snap->clear_color = scene->clear_color;
}

/**
 * @brief Runs the game until close, a time limit or a fatal error.
 * @return 0 on success, 1 on failure.
 */
static int run(const Options *opts)
{
    ReRenderConfig rc;
    ReRenderer *renderer;
    Game *game;
    GameScene scene;
    ReSceneSnapshot snap;
    void *mem;
    bool autopilot;
    bool jump_prev = false;
    bool reset_prev = false;
    bool won_reported = false;
    uint32_t coins_reported = 0u;
    uint64_t accumulator = 0u;
    uint64_t last_ns;
    uint32_t render_width = rc.width;
    uint32_t render_height = rc.height;
    uint32_t rendered = 0u;
    int status = 0;

    mem = malloc(GAME_WORLD_MEMORY);
    game = malloc(sizeof *game);
    if (mem == NULL || game == NULL) {
        fprintf(stderr, "fatal: out of memory\n");
        free(mem);
        free(game);
        return 1;
    }
    if (!game_init(game, mem, GAME_WORLD_MEMORY)) {
        fprintf(stderr, "fatal: game init failed\n");
        free(mem);
        free(game);
        return 1;
    }

    memset(&rc, 0, sizeof rc);
    rc.windowed = !opts->headless;
    rc.width = opts->width;
    rc.height = opts->height;
    rc.title = GAME_TITLE;
    rc.window_x = -1;
    rc.window_y = -1;
    renderer = opts->headless ? re_null_renderer_create(&rc) : re_renderer_create(&rc);
    if (renderer == NULL) {
        fprintf(stderr, "fatal: renderer creation failed\n");
        free(mem);
        free(game);
        return 1;
    }
    re_renderer_extent(renderer, &render_width, &render_height);
    if (render_width == 0u || render_height == 0u) {
        render_width = rc.width;
        render_height = rc.height;
    }
    autopilot = opts->autopilot || re_renderer_is_null(renderer);
    if (autopilot) {
        printf("autopilot enabled (bot plays the level)\n");
    }
    printf("platformer: %s, coins=%u, reach the green pillar at x=%.1f\n",
           re_renderer_is_null(renderer) ? "null renderer" : "vulkan",
           game->coin_count, (double)game->goal_x);
    fflush(stdout);

    last_ns = re_time_ns();
    while (!re_renderer_should_close(renderer)) {
        ReInput keys;
        GameInput input;
        uint64_t now = re_time_ns();
        uint64_t elapsed = now - last_ns;
        uint32_t steps = 0u;
        bool reset_pressed;

        if (elapsed > GAME_MAX_ACCUM_NS) {
            elapsed = GAME_MAX_ACCUM_NS;
        }
        accumulator += elapsed;
        if (accumulator > GAME_MAX_ACCUM_NS) {
            accumulator = GAME_MAX_ACCUM_NS;
        }
        last_ns = now;

        re_renderer_poll_input(renderer, &keys);
        reset_pressed = keys.down[RE_KEY_R] && !reset_prev;
        reset_prev = keys.down[RE_KEY_R];
        if (reset_pressed) {
            game_reset(game);
            won_reported = false;
            coins_reported = 0u;
            printf("restart\n");
            fflush(stdout);
        }
        jump_prev = read_human_input(&keys, jump_prev, &input);

        while (accumulator >= GAME_TICK_NS && steps < GAME_MAX_STEPS) {
            if (autopilot) {
                game_autopilot(game, &input);
            }
            game_step(game, &input, 1.0f / (float)GAME_TICK_HZ);
            input.jump_pressed = false;
            accumulator -= GAME_TICK_NS;
            steps++;
        }

        if (game->coins_collected != coins_reported) {
            coins_reported = game->coins_collected;
            printf("coin %u/%u\n", coins_reported, game->coin_count);
            fflush(stdout);
        }
        if (game->won && !won_reported) {
            won_reported = true;
            printf("LEVEL COMPLETE in %.2fs (%u/%u coins, %u falls)\n",
                   (double)game->time, game->coins_collected, game->coin_count,
                   game->deaths);
            fflush(stdout);
        }

        game_build_scene(game, render_width, render_height, &scene);
        fill_snapshot(&scene, &snap);
        if (!re_renderer_render(renderer, &snap)) {
            fprintf(stderr, "fatal: renderer lost the device\n");
            status = 1;
            break;
        }
        rendered++;
        if (opts->seconds > 0.0 && (double)game->time >= opts->seconds) {
            break;
        }
    }

    printf("session: %.2fs, %" PRIu32 " frames, %u/%u coins, falls=%u, won=%s\n",
           (double)game->time, rendered, game->coins_collected, game->coin_count,
           game->deaths, game->won ? "yes" : "no");
    re_renderer_destroy(renderer);
    free(game);
    free(mem);
    return status;
}

int main(int argc, char **argv)
{
    Options opts = {false, false, 0.0, 1280u, 720u};
    int parsed = parse_args(argc, argv, &opts);

    if (parsed == 1) {
        usage(stdout);
        return 0;
    }
    if (parsed != 0) {
        usage(stderr);
        return 2;
    }
    return run(&opts);
}
