# Real Engine Platformer

A single-level, side-view platformer built directly on the Real Engine
sources in `../src`. Gameplay is 2D (run along X, jump along Y, locked to
Z = 0) with a perspective 3D camera and the engine's Vulkan forward
renderer, so it looks like a 2.5D platformer.

## Build and run

```sh
make                 # builds bin/platformer, bin/test_game and shaders
./bin/platformer     # windowed game
make test            # headless deterministic test suite
```

Run the binary from this directory: the renderer loads its SPIR-V blobs
from `bin/shaders/` (built by `make` from `../shaders/`).

Options:

```sh
./bin/platformer --headless --seconds 12   # autopilot demo, no window
./bin/platformer --windowed --autopilot    # watch the bot play
./bin/platformer --width 1920 --height 1080
```

Controls: `A`/`D` or left/right arrows move, `Space`/`W`/`Up` jumps,
`R` restarts, `Escape` closes the window. If Vulkan or a display is
unavailable, the engine falls back to its null renderer and the bot
plays the level automatically.

## Objective

Run and jump across the floating platforms, collect the coins, and reach
the green pillar at `x = 38`. Falling into a gap respawns you at the
start without resetting the coins you already collected.

## Design

| File | Responsibility |
|------|----------------|
| `src/game.c` | Level table, player control, ground probing, coins, win state |
| `src/character.c` | Procedural hooded-assassin rig and animation |
| `src/scene.c` | Instance/camera/scene construction for the renderer |
| `src/palette.h` | Material ids shared by the scene and the shader |
| `src/main.c` | CLI, fixed 120 Hz accumulator, input, window loop |
| `tests/test_game.c` | Headless movement, jump, coin, respawn, rig and full-route tests |

The player is a vertical capsule in the engine's `ReWorld`. Rotation is
pinned (`inv_inertia = 0`) so the character never tips over; horizontal
movement is velocity-driven with ground/air acceleration, gravity and
collision response stay owned by the physics world. Ground detection
reads the post-step contact list: a contact supports the player when its
normal points up towards the capsule centre. Coyote time and a jump
buffer smooth the controls.

Every visible object is one instance of the engine's unit cube, scaled by
a model matrix. Coins carry point lights for glow; the sun and two
cascaded shadow maps come from the renderer.

## Character and animation

The player is a kinematic box rig in the style of a hooded assassin: white
robe and hood, red sash, leather boots and gloves, a two-segment cape and a
hidden blade on the right forearm. `character.c` samples the `ReWorld` body
and the game's animation timers every frame; there is no extra simulation.

- **Run**: cadence phase from distance travelled, opposite arm/leg swing,
  knee flexion, torso lean, body bob and a trailing cape.
- **Jump / fall**: a tucked leap pose while rising and a spread, ready-to-land
  pose while falling.
- **Landing**: knee-bend crouch scaled by impact speed.
- **Parkour roll**: a full forward roll on hard running landings.
- **Idle**: breathing bob and relaxed arms.

Materials use a flat palette in `shaders/pbr_clustered.frag` (ids in
`palette.h`): material 0 keeps the engine's checker texture, while the
character, stone platforms, gold coins and green finish marker use solid
colours. The rig emits only boxes and is culled with conservative AABBs.

## Testing

`make test` runs `bin/test_game`, which is silent on success:

- settles and reports grounded on the starting platform;
- walks under horizontal input;
- jumps, leaves the ground and lands again;
- collects a coin once and only once;
- respawns after falling below the kill plane;
- resets score, flags and position;
- projects the player inside the camera frustum;
- validates the character rig emits finite, bounded parts while running and
  airborne;
- runs the autopilot through the whole level and requires the finish.
