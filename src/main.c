/**
 * @file main.c
 * @brief Command-line front end for the Real Engine showcase.
 *
 * Parses the run configuration, invokes re_engine_run and prints the
 * final statistics block. Unknown or malformed flags print the usage to
 * stderr and exit with status 2; a fatal engine error exits with 1.
 */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "re_core.h"
#include "re_engine.h"

/** Fixed simulation frequency of the showcase (Hz). */
#define CLI_TICK_HZ 120u

/** Default PRNG seed (same value as RE_ENGINE_CONFIG_DEFAULT). */
#define CLI_DEFAULT_SEED UINT64_C(0x5245414C454E47)

/**
 * @brief Parsed command line.
 */
typedef struct CliOptions {
    bool windowed;      /**< Open a window (default: headless). */
    bool seconds_given; /**< --seconds was supplied. */
    uint32_t bodies;    /**< Dynamic bodies to spawn. */
    double seconds;     /**< Run length, 0 = until close. */
    uint32_t rollback;  /**< Bodies checkpointed by rollback. */
    uint32_t loss;      /**< Injected packet loss percentage. */
    uint32_t delay;     /**< Simulated packet delay in ticks. */
    uint32_t rays;      /**< Raycasts per tick. */
    uint64_t seed;      /**< Simulation seed. */
    uint32_t width;     /**< Window width. */
    uint32_t height;    /**< Window height. */
} CliOptions;

/** Built-in defaults, matching RE_ENGINE_CONFIG_DEFAULT. */
static const CliOptions CLI_DEFAULTS = {
    .windowed = false,
    .seconds_given = false,
    .bodies = 10000u,
    .seconds = 10.0,
    .rollback = 64u,
    .loss = 5u,
    .delay = 3u,
    .rays = 64u,
    .seed = CLI_DEFAULT_SEED,
    .width = 1280u,
    .height = 720u,
};

/**
 * @brief Prints the command-line usage.
 * @param out Stream to print to (stdout for --help, stderr for errors).
 */
static void usage(FILE *out)
{
    fprintf(out,
            "usage: real_engine [options]\n"
            "  --headless          run without a window (default)\n"
            "  --windowed          open a window and render until closed\n"
            "  --bodies N          dynamic bodies to spawn (default 10000)\n"
            "  --seconds T         run length; 0 = until close (default 10 headless)\n"
            "  --rollback N        bodies checkpointed per frame (default 64)\n"
            "  --loss P            injected packet loss percentage, 0-100 (default 5)\n"
            "  --delay D           simulated packet delay in ticks (default 3)\n"
            "  --rays N            raycasts per tick (default 64)\n"
            "  --seed S            simulation seed in hex (default 0x5245414C454E47)\n"
            "  --width W           window width (default 1280)\n"
            "  --height H          window height (default 720)\n"
            "  --help              print this help and exit\n");
}

/**
 * @brief Parses a base-10 unsigned 32-bit value.
 * @param text Argument text.
 * @param out Receives the value.
 * @return False on empty, negative, trailing or out-of-range input.
 */
static bool parse_u32(const char *text, uint32_t *out)
{
    char *end = NULL;
    unsigned long long value;

    if (text == NULL || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return false;
    }
    *out = (uint32_t)value;
    return true;
}

/**
 * @brief Parses a finite non-negative duration in seconds.
 * @param text Argument text.
 * @param out Receives the value.
 * @return False on malformed or negative input.
 */
static bool parse_seconds(const char *text, double *out)
{
    char *end = NULL;
    double value;

    if (text == NULL || *text == '\0') {
        return false;
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value < 0.0) {
        return false;
    }
    *out = value;
    return true;
}

/**
 * @brief Parses a hexadecimal 64-bit seed.
 * @param text Argument text (0x prefix optional).
 * @param out Receives the value.
 * @return False on malformed input.
 */
static bool parse_seed(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;

    if (text == NULL || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    value = strtoull(text, &end, 16);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *out = (uint64_t)value;
    return true;
}

/**
 * @brief Applies one of the unsigned-value flags.
 * @param opts Options to update.
 * @param name Flag name.
 * @param value Flag value.
 * @return False when the flag is unknown or the value is malformed.
 */
static bool apply_u32_flag(CliOptions *opts, const char *name, const char *value)
{
    typedef struct Entry {
        const char *name;  /**< Flag spelling. */
        uint32_t *field;   /**< Destination field. */
    } Entry;
    const Entry entries[] = {
        {"--bodies", &opts->bodies}, {"--rollback", &opts->rollback},
        {"--loss", &opts->loss},     {"--delay", &opts->delay},
        {"--rays", &opts->rays},     {"--width", &opts->width},
        {"--height", &opts->height},
    };

    for (size_t i = 0u; i < RE_ARRAY_LEN(entries); i++) {
        if (strcmp(name, entries[i].name) == 0) {
            return parse_u32(value, entries[i].field);
        }
    }
    return false;
}

/**
 * @brief Applies a flag that takes one value.
 * @param opts Options to update.
 * @param name Flag name.
 * @param value Flag value.
 * @return False when the flag is unknown or the value is malformed.
 */
static bool apply_number_flag(CliOptions *opts, const char *name, const char *value)
{
    if (apply_u32_flag(opts, name, value)) {
        return true;
    }
    if (strcmp(name, "--seed") == 0) {
        return parse_seed(value, &opts->seed);
    }
    if (strcmp(name, "--seconds") == 0) {
        if (!parse_seconds(value, &opts->seconds)) {
            return false;
        }
        opts->seconds_given = true;
        return true;
    }
    return false;
}

/**
 * @brief Parses the full argument vector.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param opts Receives the options.
 * @return 0 on success, 1 for --help, -1 on error.
 */
static int parse_args(int argc, char **argv, CliOptions *opts)
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0) {
            return 1;
        }
        if (strcmp(arg, "--headless") == 0) {
            opts->windowed = false;
            continue;
        }
        if (strcmp(arg, "--windowed") == 0) {
            opts->windowed = true;
            continue;
        }
        if (i + 1 >= argc || !apply_number_flag(opts, arg, argv[i + 1])) {
            return -1;
        }
        i++;
    }
    return 0;
}

/**
 * @brief Rejects configurations the engine cannot run.
 * @param opts Parsed options.
 * @return True when every value is in range.
 */
static bool cli_valid(const CliOptions *opts)
{
    return opts->bodies >= 1u && opts->bodies <= UINT32_MAX / 8u &&
           opts->loss <= 100u && opts->width >= 1u && opts->height >= 1u;
}

/**
 * @brief Prints the startup banner.
 * @param opts Parsed options.
 */
static void print_banner(const CliOptions *opts)
{
    printf("REAL ENGINE v%u.%u.%u | C23 | bodies=%u tick=%uHz rollback=%u loss=%u%% "
           "delay=%u rays=%u\n",
           RE_VERSION_MAJOR, RE_VERSION_MINOR, RE_VERSION_PATCH, opts->bodies,
           CLI_TICK_HZ, opts->rollback, opts->loss, opts->delay, opts->rays);
    printf("mode=%s window=%ux%u seconds=%.1f seed=0x%016" PRIX64 "\n",
           opts->windowed ? "windowed" : "headless", opts->width, opts->height,
           opts->seconds, opts->seed);
}

/**
 * @brief Prints the final statistics block.
 * @param stats Completed run statistics.
 */
static void print_stats(const ReEngineStats *stats)
{
    printf("\n--- final stats ---\n"
           "ticks          : %" PRIu64 "\n"
           "frames         : %" PRIu64 "\n"
           "tick ms  avg   : %.3f\n"
           "tick ms  max   : %.3f\n"
           "frame ms avg   : %.2f\n"
           "pairs          : %u\n"
           "contacts       : %u\n"
           "awake bodies   : %u\n"
           "raycasts       : %" PRIu64 "\n"
           "ray hits       : %" PRIu64 "\n"
           "packets lost   : %" PRIu64 "\n"
           "rollbacks      : %" PRIu64 "\n"
           "desyncs        : %" PRIu64 "\n"
           "state checksum : 0x%016" PRIX64 "\n",
           stats->ticks, stats->frames, stats->avg_tick_ms, stats->max_tick_ms,
           stats->avg_frame_ms, stats->broadphase_pairs, stats->contacts,
           stats->awake_bodies, stats->raycasts, stats->ray_hits,
           stats->packets_lost, stats->rollbacks, stats->desyncs,
           stats->state_checksum);
}

/**
 * @brief Copies the CLI options into an engine configuration.
 * @param opts Parsed options.
 * @param cfg Receives the configuration.
 */
static void build_config(const CliOptions *opts, ReEngineConfig *cfg)
{
    cfg->body_count = opts->bodies;
    cfg->tick_hz = CLI_TICK_HZ;
    cfg->max_seconds = opts->seconds;
    cfg->windowed = opts->windowed;
    cfg->width = opts->width;
    cfg->height = opts->height;
    cfg->rollback_bodies = opts->rollback;
    cfg->packet_loss_pct = opts->loss;
    cfg->packet_delay = opts->delay;
    cfg->rays_per_tick = opts->rays;
    cfg->seed = opts->seed;
}

/**
 * @brief Runs the engine and prints the statistics block.
 * @param opts Parsed options (run length defaulted here).
 * @return Process exit status.
 */
static int run_and_report(CliOptions *opts)
{
    ReEngineConfig cfg;
    ReEngineStats stats;
    int rc;

    if (!opts->seconds_given) {
        opts->seconds = opts->windowed ? 0.0 : 10.0;
    }
    if (!opts->windowed && opts->seconds == 0.0) {
        re_log_warn("headless run with --seconds 0 runs until interrupted");
    }
    print_banner(opts);
    build_config(opts, &cfg);
    memset(&stats, 0, sizeof stats);
    rc = re_engine_run(&cfg, &stats);
    if (rc != 0) {
        fprintf(stderr, "fatal: engine run failed (%d)\n", rc);
        return rc;
    }
    print_stats(&stats);
    if (stats.desyncs > 0u) {
        printf("WARNING: DESYNC DETECTED (%" PRIu64 ")\n", stats.desyncs);
    }
    return 0;
}

/**
 * @brief Entry point: parse the CLI, run the engine, report statistics.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on a fatal engine error, 2 on bad arguments.
 */
int main(int argc, char **argv)
{
    CliOptions opts = CLI_DEFAULTS;
    const int parsed = parse_args(argc, argv, &opts);

    if (parsed == 1) {
        usage(stdout);
        return 0;
    }
    if (parsed != 0 || !cli_valid(&opts)) {
        usage(stderr);
        return 2;
    }
    return run_and_report(&opts);
}
