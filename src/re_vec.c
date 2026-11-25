/**
 * @file re_vec.c
 * @brief Float linear algebra: vectors, matrices, quaternions, AABBs,
 *        rays and frustums.
 *
 * Determinism contract: scalar code uses plain IEEE-754 operations in a
 * fixed evaluation order and no fused multiply-add (build with
 * -ffp-contract=off). AVX2 paths are enabled only by RE_USE_AVX2 and are
 * alternative implementations of the same contract; they may differ
 * from the scalar reference in the last ulp.
 */
#include <stdint.h> /* re_vec.h uses int64_t without including it */

#include "re_vec.h"

#include <math.h>

#if defined(RE_USE_AVX2)
#include <immintrin.h>
#endif

/* ------------------------------ ReVec3 ------------------------------ */

/** @brief Zero vector. */
ReVec3 re_vec3_zero(void)
{
    return re_vec3_make(0.0f, 0.0f, 0.0f);
}

/** @brief Vector with the given components. */
ReVec3 re_vec3_make(float x, float y, float z)
{
    ReVec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

/** @brief Component-wise sum. */
ReVec3 re_vec3_add(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(a.x + b.x, a.y + b.y, a.z + b.z);
}

/** @brief Component-wise difference. */
ReVec3 re_vec3_sub(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** @brief Component-wise negation. */
ReVec3 re_vec3_neg(ReVec3 a)
{
    return re_vec3_make(-a.x, -a.y, -a.z);
}

/** @brief Vector scaled by s. */
ReVec3 re_vec3_scale(ReVec3 v, float s)
{
    return re_vec3_make(v.x * s, v.y * s, v.z * s);
}

/** @brief a + b * s, evaluated as a multiply followed by an add. */
ReVec3 re_vec3_add_scaled(ReVec3 a, ReVec3 b, float s)
{
    return re_vec3_make(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s);
}

/** @brief Component-wise product. */
ReVec3 re_vec3_mul(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(a.x * b.x, a.y * b.y, a.z * b.z);
}

/** @brief Dot product. */
float re_vec3_dot(ReVec3 a, ReVec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** @brief Cross product. */
ReVec3 re_vec3_cross(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

/** @brief Squared length. */
float re_vec3_length_sq(ReVec3 v)
{
    return re_vec3_dot(v, v);
}

/** @brief Length. */
float re_vec3_length(ReVec3 v)
{
    return sqrtf(re_vec3_dot(v, v));
}

/** @brief Normalized vector; zero and non-finite lengths map to zero. */
ReVec3 re_vec3_normalize(ReVec3 v)
{
    float len = sqrtf(re_vec3_dot(v, v));

    if (!(len > 0.0f) || !isfinite(len)) {
        return re_vec3_zero();
    }
    return re_vec3_scale(v, 1.0f / len);
}

/** @brief Component-wise minimum. */
ReVec3 re_vec3_min(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

/** @brief Component-wise maximum. */
ReVec3 re_vec3_max(ReVec3 a, ReVec3 b)
{
    return re_vec3_make(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

/** @brief Component-wise absolute value. */
ReVec3 re_vec3_abs(ReVec3 v)
{
    return re_vec3_make(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

/** @brief Linear interpolation a + (b - a) * t. */
ReVec3 re_vec3_lerp(ReVec3 a, ReVec3 b, float t)
{
    return re_vec3_add(a, re_vec3_scale(re_vec3_sub(b, a), t));
}

/** @brief Distance between two points. */
float re_vec3_distance(ReVec3 a, ReVec3 b)
{
    return re_vec3_length(re_vec3_sub(a, b));
}

/** @brief True when every component difference is below eps. */
bool re_vec3_equal_eps(ReVec3 a, ReVec3 b, float eps)
{
    return fabsf(a.x - b.x) < eps
        && fabsf(a.y - b.y) < eps
        && fabsf(a.z - b.z) < eps;
}

/* ------------------------------ ReVec4 ------------------------------ */

/** @brief Vector with the given components. */
ReVec4 re_vec4_make(float x, float y, float z, float w)
{
    ReVec4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

/** @brief Dot product. */
float re_vec4_dot(ReVec4 a, ReVec4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** @brief Component-wise sum. */
ReVec4 re_vec4_add(ReVec4 a, ReVec4 b)
{
    return re_vec4_make(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/** @brief Vector scaled by s. */
ReVec4 re_vec4_scale(ReVec4 v, float s)
{
    return re_vec4_make(v.x * s, v.y * s, v.z * s, v.w * s);
}

/* ------------------------------ ReMat4 ------------------------------ */

/* Returns component i (0-3) of a 4D vector. */
static float vec4_component(ReVec4 v, int i)
{
    if (i == 0) {
        return v.x;
    }
    if (i == 1) {
        return v.y;
    }
    return (i == 2) ? v.z : v.w;
}

#if !defined(RE_USE_AVX2)
/* Dot product of four components in left-to-right order (scalar path). */
static float dot4(ReVec4 a, ReVec4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
#endif

/* Adds two 4D vectors (plane extraction). */
static ReVec4 vec4_add(ReVec4 a, ReVec4 b)
{
    return re_vec4_make(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/* Subtracts two 4D vectors (plane extraction). */
static ReVec4 vec4_sub(ReVec4 a, ReVec4 b)
{
    return re_vec4_make(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/** @brief Identity matrix. */
ReMat4 re_mat4_identity(void)
{
    ReMat4 m;

    m.col[0] = re_vec4_make(1.0f, 0.0f, 0.0f, 0.0f);
    m.col[1] = re_vec4_make(0.0f, 1.0f, 0.0f, 0.0f);
    m.col[2] = re_vec4_make(0.0f, 0.0f, 1.0f, 0.0f);
    m.col[3] = re_vec4_make(0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}

#if defined(RE_USE_AVX2)
/*
 * a * b with 256-bit vectors: two output columns per accumulator.
 * b[k][j] is gathered by permuting the two loaded column pairs; lanes
 * 0-3 get b[k][0], lanes 4-7 get b[k][1] (and likewise for columns 2-3).
 * 128-bit loads/stores stay inside one ReVec4 subobject: the structs are
 * only 16-byte aligned, so aligned 256-bit moves would be invalid.
 */
static ReMat4 mat4_mul_avx2(ReMat4 a, ReMat4 b)
{
    ReMat4 r;
    __m256 b01 = _mm256_set_m128(_mm_loadu_ps(&b.col[1].x),
                                 _mm_loadu_ps(&b.col[0].x));
    __m256 b23 = _mm256_set_m128(_mm_loadu_ps(&b.col[3].x),
                                 _mm_loadu_ps(&b.col[2].x));
    __m256 r01 = _mm256_setzero_ps();
    __m256 r23 = _mm256_setzero_ps();

    for (int k = 0; k < 4; ++k) {
        __m256 ac = _mm256_broadcast_ps((const __m128 *)&a.col[k]);
        __m256i i01 = _mm256_setr_epi32(k, k, k, k, k + 4, k + 4, k + 4, k + 4);
        r01 = _mm256_add_ps(r01, _mm256_mul_ps(ac, _mm256_permutevar8x32_ps(b01, i01)));
        r23 = _mm256_add_ps(r23, _mm256_mul_ps(ac, _mm256_permutevar8x32_ps(b23, i01)));
    }
    _mm_storeu_ps(&r.col[0].x, _mm256_castps256_ps128(r01));
    _mm_storeu_ps(&r.col[1].x, _mm256_extractf128_ps(r01, 1));
    _mm_storeu_ps(&r.col[2].x, _mm256_castps256_ps128(r23));
    _mm_storeu_ps(&r.col[3].x, _mm256_extractf128_ps(r23, 1));
    return r;
}
#endif

/** @brief Matrix product a * b. */
ReMat4 re_mat4_mul(ReMat4 a, ReMat4 b)
{
#if defined(RE_USE_AVX2)
    return mat4_mul_avx2(a, b);
#else
    ReMat4 r;
    ReVec4 r0 = re_vec4_make(a.col[0].x, a.col[1].x, a.col[2].x, a.col[3].x);
    ReVec4 r1 = re_vec4_make(a.col[0].y, a.col[1].y, a.col[2].y, a.col[3].y);
    ReVec4 r2 = re_vec4_make(a.col[0].z, a.col[1].z, a.col[2].z, a.col[3].z);
    ReVec4 r3 = re_vec4_make(a.col[0].w, a.col[1].w, a.col[2].w, a.col[3].w);

    for (int j = 0; j < 4; ++j) {
        r.col[j] = re_vec4_make(dot4(r0, b.col[j]), dot4(r1, b.col[j]),
                                dot4(r2, b.col[j]), dot4(r3, b.col[j]));
    }
    return r;
#endif
}

/** @brief Transposed matrix. */
ReMat4 re_mat4_transpose(ReMat4 m)
{
    ReMat4 t;

    t.col[0] = re_vec4_make(m.col[0].x, m.col[1].x, m.col[2].x, m.col[3].x);
    t.col[1] = re_vec4_make(m.col[0].y, m.col[1].y, m.col[2].y, m.col[3].y);
    t.col[2] = re_vec4_make(m.col[0].z, m.col[1].z, m.col[2].z, m.col[3].z);
    t.col[3] = re_vec4_make(m.col[0].w, m.col[1].w, m.col[2].w, m.col[3].w);
    return t;
}

/**
 * @brief Right-handed reverse-Z perspective matrix.
 *
 * With w = -z_view and clip depth in [0, 1], near maps to z/w = 1 and
 * far to 0: m22 = n / (f - n), m23 = -1, m32 = n * f / (f - n), m33 = 0.
 * An infinite far plane degenerates to m22 = 0 and m32 = n.
 */
ReMat4 re_mat4_perspective(float fov_y, float aspect, float znear, float zfar)
{
    ReMat4 m = re_mat4_identity();
    float fn = zfar - znear;
    float f = 1.0f / tanf(fov_y * 0.5f);
    float m22 = znear / fn;
    float m32 = isinf(zfar) ? znear : (zfar * znear) / fn;

    m.col[0] = re_vec4_make(f / aspect, 0.0f, 0.0f, 0.0f);
    m.col[1] = re_vec4_make(0.0f, f, 0.0f, 0.0f);
    m.col[2] = re_vec4_make(0.0f, 0.0f, m22, -1.0f);
    m.col[3] = re_vec4_make(0.0f, 0.0f, m32, 0.0f);
    return m;
}

/**
 * @brief Right-handed reverse-Z orthographic matrix.
 *
 * With w = 1, z_view = -near maps to 1 and z_view = -far to 0, hence
 * m22 = 1 / (f - n) and m32 = f / (f - n).
 */
ReMat4 re_mat4_ortho(float left, float right, float bottom, float top,
                     float znear, float zfar)
{
    ReMat4 m = re_mat4_identity();
    float fn = zfar - znear;

    m.col[0].x = 2.0f / (right - left);
    m.col[1].y = 2.0f / (top - bottom);
    m.col[2].z = 1.0f / fn;
    m.col[3].x = -(right + left) / (right - left);
    m.col[3].y = -(top + bottom) / (top - bottom);
    m.col[3].z = isinf(zfar) ? 1.0f : zfar / fn;
    return m;
}

/** @brief Right-handed look-at view matrix. */
ReMat4 re_mat4_look_at(ReVec3 eye, ReVec3 target, ReVec3 up)
{
    ReMat4 m = re_mat4_identity();
    ReVec3 f = re_vec3_normalize(re_vec3_sub(eye, target));
    ReVec3 r = re_vec3_normalize(re_vec3_cross(up, f));
    ReVec3 u = re_vec3_cross(f, r);

    m.col[0] = re_vec4_make(r.x, u.x, f.x, 0.0f);
    m.col[1] = re_vec4_make(r.y, u.y, f.y, 0.0f);
    m.col[2] = re_vec4_make(r.z, u.z, f.z, 0.0f);
    m.col[3] = re_vec4_make(-re_vec3_dot(r, eye), -re_vec3_dot(u, eye),
                            -re_vec3_dot(f, eye), 1.0f);
    return m;
}

/** @brief Pure translation matrix. */
ReMat4 re_mat4_translate(ReVec3 t)
{
    ReMat4 m = re_mat4_identity();

    m.col[3] = re_vec4_make(t.x, t.y, t.z, 1.0f);
    return m;
}

/** @brief Pure rotation matrix around a normalized axis. */
ReMat4 re_mat4_rotate_axis(ReVec3 axis, float angle)
{
    return re_quat_to_mat4(re_quat_from_axis_angle(axis, angle));
}

/** @brief Pure scale matrix. */
ReMat4 re_mat4_scale(ReVec3 s)
{
    ReMat4 m = re_mat4_identity();

    m.col[0].x = s.x;
    m.col[1].y = s.y;
    m.col[2].z = s.z;
    return m;
}

/** @brief TRS matrix: translate * rotate * scale. */
ReMat4 re_mat4_compose(ReVec3 t, ReQuat r, ReVec3 s)
{
    ReMat4 m = re_quat_to_mat4(r);

    m.col[0] = re_vec4_scale(m.col[0], s.x);
    m.col[1] = re_vec4_scale(m.col[1], s.y);
    m.col[2] = re_vec4_scale(m.col[2], s.z);
    m.col[3] = re_vec4_make(t.x, t.y, t.z, 1.0f);
    return m;
}

/* Copies a column-major matrix into an augmented [4][8] row array. */
static void mat4_to_augmented(ReMat4 m, float a[4][8])
{
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r][c] = vec4_component(m.col[c], r);
            a[r][c + 4] = (r == c) ? 1.0f : 0.0f;
        }
    }
}

/* Copies the right half of an augmented array back to a matrix. */
static ReMat4 augmented_to_mat4(const float a[4][8])
{
    ReMat4 m;

    for (int c = 0; c < 4; ++c) {
        m.col[c] = re_vec4_make(a[0][c + 4], a[1][c + 4],
                                a[2][c + 4], a[3][c + 4]);
    }
    return m;
}

/* In-place Gauss-Jordan elimination on a [4][8] augmented matrix. */
static bool gauss_jordan_eliminate(float a[4][8])
{
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 4; ++r) {
            if (fabsf(a[r][col]) > fabsf(a[pivot][col])) {
                pivot = r;
            }
        }
        if (a[pivot][col] == 0.0f) {
            return false;
        }
        if (pivot != col) {
            for (int c = 0; c < 8; ++c) {
                float tmp = a[col][c];
                a[col][c] = a[pivot][c];
                a[pivot][c] = tmp;
            }
        }
        float inv = 1.0f / a[col][col];
        for (int c = 0; c < 8; ++c) {
            a[col][c] *= inv;
        }
        for (int r = 0; r < 4; ++r) {
            float factor = a[r][col];
            if (r == col || factor == 0.0f) {
                continue;
            }
            for (int c = 0; c < 8; ++c) {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    return true;
}

/**
 * @brief Full inverse via Gauss-Jordan with partial pivoting.
 *
 * A zero pivot means the matrix is singular; the identity is returned as
 * the documented degenerate result.
 */
ReMat4 re_mat4_inverse(ReMat4 m)
{
    float a[4][8];

    mat4_to_augmented(m, a);
    if (!gauss_jordan_eliminate(a)) {
        return re_mat4_identity();
    }
    return augmented_to_mat4(a);
}

/**
 * @brief Fast inverse for affine matrices.
 *
 * Inverts the upper-left 3x3 block (rotation/scale) through its adjugate
 * and negates its product with the translation. A zero determinant maps
 * to the identity as the documented degenerate result.
 */
ReMat4 re_mat4_affine_inverse(ReMat4 m)
{
    float a = m.col[0].x, b = m.col[1].x, c = m.col[2].x;
    float d = m.col[0].y, e = m.col[1].y, f = m.col[2].y;
    float g = m.col[0].z, h = m.col[1].z, i = m.col[2].z;
    float ei = e * i, fh = f * h, di = d * i, fg = f * g;
    float dh = d * h, eg = e * g;
    float det = a * (ei - fh) - b * (di - fg) + c * (dh - eg);

    if (det == 0.0f) {
        return re_mat4_identity();
    }
    float inv = 1.0f / det;
    float r00 = (ei - fh) * inv, r01 = (c * h - b * i) * inv;
    float r02 = (b * f - c * e) * inv;
    float r10 = (fg - di) * inv, r11 = (a * i - c * g) * inv;
    float r12 = (c * d - a * f) * inv;
    float r20 = (dh - eg) * inv, r21 = (b * g - a * h) * inv;
    float r22 = (a * e - b * d) * inv;
    float tx = m.col[3].x, ty = m.col[3].y, tz = m.col[3].z;
    ReMat4 out;

    out.col[0] = re_vec4_make(r00, r10, r20, 0.0f);
    out.col[1] = re_vec4_make(r01, r11, r21, 0.0f);
    out.col[2] = re_vec4_make(r02, r12, r22, 0.0f);
    out.col[3] = re_vec4_make(-(r00 * tx + r01 * ty + r02 * tz),
                              -(r10 * tx + r11 * ty + r12 * tz),
                              -(r20 * tx + r21 * ty + r22 * tz), 1.0f);
    return out;
}

/** @brief p transformed as a point (w = 1, perspective divide applied). */
ReVec3 re_mat4_transform_point(ReMat4 m, ReVec3 p)
{
    float x = m.col[0].x * p.x + m.col[1].x * p.y + m.col[2].x * p.z + m.col[3].x;
    float y = m.col[0].y * p.x + m.col[1].y * p.y + m.col[2].y * p.z + m.col[3].y;
    float z = m.col[0].z * p.x + m.col[1].z * p.y + m.col[2].z * p.z + m.col[3].z;
    float w = m.col[0].w * p.x + m.col[1].w * p.y + m.col[2].w * p.z + m.col[3].w;

    return re_vec3_make(x / w, y / w, z / w);
}

/** @brief v transformed as a direction (w = 0). */
ReVec3 re_mat4_transform_dir(ReMat4 m, ReVec3 v)
{
    return re_vec3_make(
        m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z,
        m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z,
        m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z);
}

/** @brief v transformed as a 4D point (w = 1), no divide. */
ReVec4 re_mat4_transform_vec4(ReMat4 m, ReVec4 v)
{
    ReVec4 r;

#if defined(RE_USE_AVX2)
    __m256 c01 = _mm256_set_m128(_mm_loadu_ps(&m.col[1].x),
                                 _mm_loadu_ps(&m.col[0].x));
    __m256 c23 = _mm256_set_m128(_mm_loadu_ps(&m.col[3].x),
                                 _mm_loadu_ps(&m.col[2].x));
    __m256 xy = _mm256_set_m128(_mm_set1_ps(v.y), _mm_set1_ps(v.x));
    __m256 zw = _mm256_set_m128(_mm_set1_ps(v.w), _mm_set1_ps(v.z));
    __m256 prod = _mm256_add_ps(_mm256_mul_ps(c01, xy),
                                _mm256_mul_ps(c23, zw));
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(prod),
                            _mm256_extractf128_ps(prod, 1));

    _mm_storeu_ps(&r.x, sum);
#else
    r.x = m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z + m.col[3].x * v.w;
    r.y = m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z + m.col[3].y * v.w;
    r.z = m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z + m.col[3].z * v.w;
    r.w = m.col[0].w * v.x + m.col[1].w * v.y + m.col[2].w * v.z + m.col[3].w * v.w;
#endif
    return r;
}

/** @brief Matrix built from a unit quaternion (see re_quat_to_mat4). */
ReMat4 re_mat4_from_quat(ReQuat q)
{
    return re_quat_to_mat4(q);
}

/* ------------------------------ ReQuat ------------------------------ */

/** @brief Identity quaternion. */
ReQuat re_quat_identity(void)
{
    ReQuat q;

    q.x = 0.0f;
    q.y = 0.0f;
    q.z = 0.0f;
    q.w = 1.0f;
    return q;
}

/** @brief Hamilton product a * b. */
ReQuat re_quat_mul(ReQuat a, ReQuat b)
{
    ReQuat q;

    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return q;
}

/** @brief Normalized quaternion; zero and non-finite lengths map to identity. */
ReQuat re_quat_normalize(ReQuat q)
{
    float len = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float inv;

    if (!(len > 0.0f) || !isfinite(len)) {
        return re_quat_identity();
    }
    inv = 1.0f / len;
    q.x *= inv;
    q.y *= inv;
    q.z *= inv;
    q.w *= inv;
    return q;
}

/** @brief Conjugate. */
ReQuat re_quat_conjugate(ReQuat q)
{
    ReQuat r;

    r.x = -q.x;
    r.y = -q.y;
    r.z = -q.z;
    r.w = q.w;
    return r;
}

/**
 * @brief Rotation quaternion from a normalized axis and an angle.
 *
 * The axis is defensively re-normalized; a zero (or non-finite) axis
 * yields the identity quaternion.
 */
ReQuat re_quat_from_axis_angle(ReVec3 axis, float angle)
{
    ReQuat q;
    ReVec3 n = re_vec3_normalize(axis);
    float half = angle * 0.5f;
    float s = sinf(half);

    if (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f) {
        return re_quat_identity();
    }
    q.x = n.x * s;
    q.y = n.y * s;
    q.z = n.z * s;
    q.w = cosf(half);
    return q;
}

/** @brief Vector rotated by the unit quaternion. */
ReVec3 re_quat_rotate_vec3(ReQuat q, ReVec3 v)
{
    ReVec3 u = re_vec3_make(q.x, q.y, q.z);
    ReVec3 t = re_vec3_scale(re_vec3_cross(u, v), 2.0f);

    return re_vec3_add(re_vec3_add(v, re_vec3_scale(t, q.w)),
                       re_vec3_cross(u, t));
}

/** @brief Rotation matrix from the unit quaternion. */
ReMat4 re_quat_to_mat4(ReQuat q)
{
    ReMat4 m = re_mat4_identity();
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    m.col[0] = re_vec4_make(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz),
                            2.0f * (xz - wy), 0.0f);
    m.col[1] = re_vec4_make(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz),
                            2.0f * (yz + wx), 0.0f);
    m.col[2] = re_vec4_make(2.0f * (xz + wy), 2.0f * (yz - wx),
                            1.0f - 2.0f * (xx + yy), 0.0f);
    return m;
}

/** @brief Normalized linear interpolation (nlerp). */
ReQuat re_quat_nlerp(ReQuat a, ReQuat b, float t)
{
    ReQuat q;

    q.x = a.x + (b.x - a.x) * t;
    q.y = a.y + (b.y - a.y) * t;
    q.z = a.z + (b.z - a.z) * t;
    q.w = a.w + (b.w - a.w) * t;
    return re_quat_normalize(q);
}

/** @brief Spherical linear interpolation; near-parallel inputs use nlerp. */
ReQuat re_quat_slerp(ReQuat a, ReQuat b, float t)
{
    float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    ReQuat q;
    float theta;
    float sa;
    float sb;

    dot = fminf(fmaxf(dot, -1.0f), 1.0f);
    if (dot < 0.0f) {
        b.x = -b.x;
        b.y = -b.y;
        b.z = -b.z;
        b.w = -b.w;
        dot = -dot;
    }
    if (dot > 0.9995f) {
        return re_quat_nlerp(a, b, t);
    }
    theta = acosf(dot);
    sa = sinf((1.0f - t) * theta);
    sb = sinf(t * theta);
    q.x = a.x * sa + b.x * sb;
    q.y = a.y * sa + b.y * sb;
    q.z = a.z * sa + b.z * sb;
    q.w = a.w * sa + b.w * sb;
    return re_quat_normalize(q);
}

/* ------------------------------ ReAabb ------------------------------ */

/** @brief Empty AABB (min = +inf, max = -inf). */
ReAabb re_aabb_empty(void)
{
    ReAabb b;

    b.min = re_vec3_make(INFINITY, INFINITY, INFINITY);
    b.max = re_vec3_make(-INFINITY, -INFINITY, -INFINITY);
    return b;
}

/** @brief AABB containing a single point. */
ReAabb re_aabb_from_point(ReVec3 p)
{
    ReAabb b;

    b.min = p;
    b.max = p;
    return b;
}

/** @brief AABB containing both points (either order). */
ReAabb re_aabb_from_two_points(ReVec3 p, ReVec3 q)
{
    ReAabb b;

    b.min = re_vec3_min(p, q);
    b.max = re_vec3_max(p, q);
    return b;
}

/** @brief True when the box is empty on any axis. */
bool re_aabb_is_empty(ReAabb b)
{
    return b.min.x > b.max.x || b.min.y > b.max.y || b.min.z > b.max.z;
}

/** @brief Union of the two boxes. */
ReAabb re_aabb_merge(ReAabb a, ReAabb b)
{
    ReAabb r;

    r.min = re_vec3_min(a.min, b.min);
    r.max = re_vec3_max(a.max, b.max);
    return r;
}

/** @brief Box expanded by e on every side. */
ReAabb re_aabb_expand(ReAabb b, float e)
{
    ReAabb r;

    r.min = re_vec3_make(b.min.x - e, b.min.y - e, b.min.z - e);
    r.max = re_vec3_make(b.max.x + e, b.max.y + e, b.max.z + e);
    return r;
}

/** @brief Center point. */
ReVec3 re_aabb_center(ReAabb b)
{
    return re_vec3_scale(re_vec3_add(b.min, b.max), 0.5f);
}

/** @brief Half extents ((max - min) / 2). */
ReVec3 re_aabb_extents(ReAabb b)
{
    return re_vec3_scale(re_vec3_sub(b.max, b.min), 0.5f);
}

/** @brief Surface area (0 for empty boxes). */
float re_aabb_surface_area(ReAabb b)
{
    ReVec3 d;

    if (re_aabb_is_empty(b)) {
        return 0.0f;
    }
    d = re_vec3_sub(b.max, b.min);
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

/** @brief True when p is inside (inclusive) the box. */
bool re_aabb_contains_point(ReAabb b, ReVec3 p)
{
    return p.x >= b.min.x && p.x <= b.max.x
        && p.y >= b.min.y && p.y <= b.max.y
        && p.z >= b.min.z && p.z <= b.max.z;
}

/** @brief True when inner is fully inside outer (inclusive). */
bool re_aabb_contains_aabb(ReAabb outer, ReAabb inner)
{
    return inner.min.x >= outer.min.x && inner.max.x <= outer.max.x
        && inner.min.y >= outer.min.y && inner.max.y <= outer.max.y
        && inner.min.z >= outer.min.z && inner.max.z <= outer.max.z;
}

/** @brief True when the boxes overlap or touch. */
bool re_aabb_intersects(ReAabb a, ReAabb b)
{
    return a.min.x <= b.max.x && a.max.x >= b.min.x
        && a.min.y <= b.max.y && a.max.y >= b.min.y
        && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

/**
 * @brief Slab test of a ray against the box.
 *
 * A zero direction component requires the origin to lie inside that
 * slab. t_out receives max(tmin, 0), so a ray starting inside reports an
 * entry distance of 0.
 */
bool re_aabb_ray(ReAabb b, ReRay ray, float t_max, float *t_out)
{
    float tmin = 0.0f;
    float tmax = t_max;
    float o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    float d[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
    float lo[3] = {b.min.x, b.min.y, b.min.z};
    float hi[3] = {b.max.x, b.max.y, b.max.z};

    if (!(t_max >= 0.0f)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (d[i] == 0.0f) {
            if (o[i] < lo[i] || o[i] > hi[i]) {
                return false;
            }
        } else {
            float inv = 1.0f / d[i];
            float t1 = (lo[i] - o[i]) * inv;
            float t2 = (hi[i] - o[i]) * inv;
            float near = fminf(t1, t2);
            float far = fmaxf(t1, t2);

            tmin = fmaxf(tmin, near);
            tmax = fminf(tmax, far);
            if (tmin > tmax) {
                return false;
            }
        }
    }
    if (t_out != NULL) {
        *t_out = tmin;
    }
    return true;
}

/** @brief Transformed box (Arvo's conservative method). */
ReAabb re_aabb_transform(ReAabb b, ReMat4 m)
{
    ReVec3 center;
    ReVec3 e;
    ReVec3 ext;
    ReAabb r;

    if (re_aabb_is_empty(b)) {
        return b;
    }
    center = re_mat4_transform_point(m, re_aabb_center(b));
    e = re_vec3_abs(re_aabb_extents(b));
    ext = re_vec3_make(
        fabsf(m.col[0].x) * e.x + fabsf(m.col[1].x) * e.y + fabsf(m.col[2].x) * e.z,
        fabsf(m.col[0].y) * e.x + fabsf(m.col[1].y) * e.y + fabsf(m.col[2].y) * e.z,
        fabsf(m.col[0].z) * e.x + fabsf(m.col[1].z) * e.y + fabsf(m.col[2].z) * e.z);
    r.min = re_vec3_sub(center, ext);
    r.max = re_vec3_add(center, ext);
    return r;
}

/* ------------------------------ ReRay ------------------------------- */

/** @brief Point at distance t along the ray. */
ReVec3 re_ray_point_at(ReRay ray, float t)
{
    return re_vec3_add(ray.origin, re_vec3_scale(ray.dir, t));
}

/* ----------------------------- ReFrustum ---------------------------- */

/* Scales a plane so that (a, b, c) has unit length. */
static ReVec4 plane_normalize(ReVec4 p)
{
    float len = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);

    if (!(len > 0.0f) || !isfinite(len)) {
        return p;
    }
    return re_vec4_scale(p, 1.0f / len);
}

/* Builds a plane from a linear combination of two matrix rows. */
static ReVec4 plane_from_rows(ReVec4 row_a, ReVec4 row_b, bool subtract)
{
    ReVec4 p = subtract ? vec4_sub(row_a, row_b) : vec4_add(row_a, row_b);

    return plane_normalize(p);
}

/**
 * @brief Extracts the six frustum planes (Gribb-Hartmann).
 *
 * The matrix has a Vulkan-style depth range [0, w] and reverse Z, so
 * near is the w - z >= 0 halfspace and far is z >= 0. Clip-space rows
 * are row3 +/- row0, row3 +/- row1, row3 - row2 and row2.
 */
ReFrustum re_frustum_from_mat4(ReMat4 vp)
{
    ReFrustum f;
    ReVec4 r0 = re_vec4_make(vp.col[0].x, vp.col[1].x, vp.col[2].x, vp.col[3].x);
    ReVec4 r1 = re_vec4_make(vp.col[0].y, vp.col[1].y, vp.col[2].y, vp.col[3].y);
    ReVec4 r2 = re_vec4_make(vp.col[0].z, vp.col[1].z, vp.col[2].z, vp.col[3].z);
    ReVec4 r3 = re_vec4_make(vp.col[0].w, vp.col[1].w, vp.col[2].w, vp.col[3].w);

    f.planes[0] = plane_from_rows(r3, r0, false);
    f.planes[1] = plane_from_rows(r3, r0, true);
    f.planes[2] = plane_from_rows(r3, r1, false);
    f.planes[3] = plane_from_rows(r3, r1, true);
    f.planes[4] = plane_from_rows(r3, r2, true);
    f.planes[5] = plane_normalize(r2);
    return f;
}

/** @brief True when the point is inside or on the frustum. */
bool re_frustum_contains_point(ReFrustum f, ReVec3 p)
{
    for (int i = 0; i < 6; ++i) {
        ReVec4 pl = f.planes[i];
        float d = pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w;

        if (d < 0.0f) {
            return false;
        }
    }
    return true;
}

/** @brief True when the sphere intersects the frustum. */
bool re_frustum_contains_sphere(ReFrustum f, ReSphere s)
{
    for (int i = 0; i < 6; ++i) {
        ReVec4 pl = f.planes[i];
        float d = pl.x * s.center.x + pl.y * s.center.y
                + pl.z * s.center.z + pl.w;

        if (d < -s.radius) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Conservative AABB-vs-frustum test.
 *
 * Uses the positive vertex (p-vertex) for the outside test and the
 * negative vertex (n-vertex) to prove full containment.
 */
int re_frustum_test_aabb(ReFrustum f, ReAabb b)
{
    bool intersects = false;

    for (int i = 0; i < 6; ++i) {
        ReVec4 pl = f.planes[i];
        float px = (pl.x >= 0.0f) ? b.max.x : b.min.x;
        float py = (pl.y >= 0.0f) ? b.max.y : b.min.y;
        float pz = (pl.z >= 0.0f) ? b.max.z : b.min.z;
        float nx = (pl.x >= 0.0f) ? b.min.x : b.max.x;
        float ny = (pl.y >= 0.0f) ? b.min.y : b.max.y;
        float nz = (pl.z >= 0.0f) ? b.min.z : b.max.z;

        if (pl.x * px + pl.y * py + pl.z * pz + pl.w < 0.0f) {
            return RE_FRUSTUM_OUTSIDE;
        }
        if (pl.x * nx + pl.y * ny + pl.z * nz + pl.w < 0.0f) {
            intersects = true;
        }
    }
    return intersects ? RE_FRUSTUM_INTERSECT : RE_FRUSTUM_INSIDE;
}

/* --------------------------- Fixed conversion ----------------------- */

/**
 * @brief Converts a Q32.32 fixed value to float (exact rounding).
 *
 * The integer is rounded to float before the exact 2^-32 scaling; using a
 * double intermediate could double-round near a float midpoint.
 */
float re_fixed_to_f32(int64_t v)
{
    return (float)v / 4294967296.0f;
}

/** @brief Converts a float to Q32.32 with saturation. */
int64_t re_f32_to_fixed(float v)
{
    double scaled = (double)v * 4294967296.0;

    if (!isfinite(v)) {
        return 0;
    }
    if (scaled >= 9223372036854775808.0) {
        return INT64_MAX;
    }
    if (scaled < -9223372036854775808.0) {
        return INT64_MIN;
    }
    return (int64_t)llround(scaled);
}
