# Real Engine — Memory Layout, Struct Alignment & ABI Specification

Target: x86-64 Linux, clang 23+, `-std=c23 -Wall -Wextra -Werror -pedantic -Wconversion -ffp-contract=off`.
Second-class: Win32 (thread + time paths compile; render backend needs X11).

## 1. Endianness and wire format

- In-memory layout is native (little-endian on all supported hosts).
- The network bitstream (`re_bitstream.h`) is **MSB-first and therefore endian-neutral**:
  floats cross the wire as their 32-bit pattern read bit-by-bit, so both peers always
  interpret the same value regardless of host endianness.
- Checksums (FNV-1a 64, CRC-64 ECMA-182) are computed over the same canonical byte
  stream used for replication.

## 2. Determinism contract

| Domain                | Guarantee                                                              |
|-----------------------|------------------------------------------------------------------------|
| `re_fixed` Q32.32     | Bit-exact on every platform and compiler: pure integer math, saturating, no UB. |
| `re_vec`/`re_physics` | Bit-exact for a given binary (IEEE-754, SSE2 scalar baseline, no FMA, no x87). |
| `re_ecs`, `re_bvh`    | Order-dependent only on input order; pairs sorted before narrowphase.   |
| `re_scheduler`        | Task order inside a wave is unspecified; callers sort results.          |
| `re_net`              | Lossless fixed-width codecs; rollback re-simulation is deterministic per the row above. |

The engine is compiled with `-ffp-contract=off` and never uses `-ffast-math` or FMA:
same binary => same bits => rollback re-simulation cannot diverge.

## 3. Core type sizes and alignments

| Type               | sizeof | align | Notes                                          |
|--------------------|-------:|------:|------------------------------------------------|
| `ReArena`          |     64 |     8 | manually padded to one cache line              |
| `RePool`           |     64 |     8 | manually padded to one cache line              |
| `ReMpmcSlot`       |     64 |     8 | `state` ticket word + `item` pointer per line  |
| `ReMpmcRing`       |     64 |     8 | `head`/`tail` atomic counters in one line      |
| `ReMask`           |     32 |     8 | 4 x uint64_t, bit i in word i/64, shift i%64   |
| `ReEntity`         |      8 |     8 | `uint32 index \| uint32 generation << 32`      |
| `ReVec3`           |     16 |    16 | padded w component for SIMD loads              |
| `ReVec4`           |     16 |    16 |                                                |
| `ReMat4`           |     64 |    16 | column-major, matches std140/std430            |
| `ReAabb`           |     32 |    16 | two 16-byte ReVec3 corners                     |
| `ReBvhNode`        |     64 |    16 | aabb 32 + parent/left/right/proxy/height       |
| `ReBody`           |    240 |     8 | see re_world.h (float state)                   |
| `ReContact`        |     96 |     8 | point/normal/r_a/r_b/depth/friction/restitution/a/b |
| `ReSceneInstance`  |     96 |     4 | mat4 + aabb + material + flags (std430)        |
| `ReBodySnap`       |     48 |     4 | in-memory (padded); wire format is 40 bytes (10 x 32 bits) |
| `ReMutex`/`ReCond` |     64/80 | 8 | inline storage >= pthread_mutex_t/cond_t       |

`static_assert`s in the headers pin every one of these down; changing a layout
without updating the assert is a compile error.

## 4. Allocation ownership

- `ReArena`/`RePool`/`ReMpmcRing`/`ReEcs`/`ReWorld`/`ReGraphCompiled` are **non-owning**:
  storage is caller-provided and reset/reused by the owner.
- `ReScheduler` and `ReRenderer` own their OS resources; `destroy` releases them.
- The engine process owns one arena for the ECS, one for the physics world, and
  stack buffers for per-frame data.

## 5. Component masks and scheduler conflict rule

Two jobs conflict iff `(a.write & b.write) | (a.write & b.read) | (b.write & a.read) != 0`.
`re_dag_plan` (Kahn) places every job in exactly one wave; same-wave jobs share no
conflict edge by construction — the Frama-C harness proves this on bounded inputs.

## 6. Vulkan binding contract (set 0)

| Binding | Type                | Contents                                  |
|--------:|---------------------|-------------------------------------------|
| 0       | uniform (std140)    | camera: view, proj, view_proj, 6 planes, position, params |
| 1       | uniform (std140)    | sun + ambient + cascade splits            |
| 2       | storage (std430)    | cluster grid: `uvec2 offset,count` x 2048 |
| 3       | sampler2DArray      | shadow cascades, `RE_SHADOW_CASCADES` layers |
| 4       | sampler2D array     | bindless textures, `RE_BINDLESS_COUNT` descriptors, update-after-bind, partially bound, variable count |
| 5       | storage (std430)    | `ReSceneInstance[RE_MAX_INSTANCES]`       |
| 6       | storage (std430)    | visible instance indices                  |
| 7       | storage (std430)    | indirect draw args + atomic drawCount     |
| 8       | storage (std430)    | point light list `ReRenderPointLightGpu`  |

All pipelines share set 0; per-pass data travels in push constants. Dynamic
rendering (VK 1.3) replaces render passes; synchronization2 emits every barrier.

## 7. Proof harness layout (`verify/`)

- `*_cbmc.c` — CBMC bounded model checking (`__CPROVER_assert`/`__CPROVER_assume`,
  symbolic inputs, unwound loops; `__CPROVER_r_ok`/`__CPROVER_w_ok` bounds proofs).
- `*_frama.c` — Frama-C/WP ACSL contracts (`requires/ensures/assigns/loop invariant/
  loop variant`, `\valid`, `\separated`, `\result` predicates).
- `*_genmc.c` — GenMC weak-memory model checking (`memory_order_acquire/release`,
  pthread threads, exhaustive behavior exploration).

`verify.sh` runs all three suites plus the strict build, GLSL validation and the
headless smoke run; tools that are not installed are reported as SKIP, never faked.
