#define _POSIX_C_SOURCE 199309L
#include "nbody.h"
#include <limits.h>
#include <time.h>

/* ---- High-resolution timer ---- */

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Particle initialisation ----
 *  Use a randomized, slightly perturbed lattice so that particles do not
 *  start unreasonably close while particle indices reveal no spatial order.
 */
static unsigned int rng_next(unsigned int *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static double rng_uniform(unsigned int *state)
{
    return (double)rng_next(state) / ((double)UINT_MAX + 1.0);
}

static double wrap_box(double x, double box)
{
    x = fmod(x, box);
    return x < 0.0 ? x + box : x;
}

static void make_axis_centers(double *center, double *width, int side,
                              double box, unsigned int *rng)
{
    double *weight = (double *)malloc((size_t)side * sizeof(double));
    if (!weight) {
        fprintf(stderr, "init_particles: failed to allocate axis weights\n");
        exit(EXIT_FAILURE);
    }

    double sum = 0.0;
    double phase = 6.28318530718 * rng_uniform(rng);
    double slope = 2.0 * rng_uniform(rng) - 1.0;
    for (int i = 0; i < side; i++) {
        double wave = sin((double)(i + 1) * 1.61803398875 +
                          6.28318530718 * rng_uniform(rng));
        double long_wave = sin(6.28318530718 * (double)i / (double)side + phase);
        double ramp = ((2.0 * (double)i / (double)(side - 1)) - 1.0) * slope;
        double block = ((i % 7) - 3) * 0.035;
        weight[i] = 1.0 + 0.24 * long_wave + 0.10 * ramp + 0.16 * wave + block +
                    0.12 * (2.0 * rng_uniform(rng) - 1.0);
        if (weight[i] < 0.62) weight[i] = 0.62;
        sum += weight[i];
    }

    double pos = 0.0;
    for (int i = 0; i < side; i++) {
        width[i] = box * weight[i] / sum;
        center[i] = pos + 0.5 * width[i];
        pos += width[i];
    }

    free(weight);
}

static void shuffle_sites(int *sites, int site_count, unsigned int *rng)
{
    for (int i = 0; i < site_count; i++) {
        sites[i] = i;
    }

    for (int i = site_count - 1; i > 0; i--) {
        int j = (int)(rng_next(rng) % (unsigned int)(i + 1));
        int tmp = sites[i];
        sites[i] = sites[j];
        sites[j] = tmp;
    }
}

void init_particles(ParticleSystem *sys, int n, double density,
                    unsigned int seed)
{
    unsigned int rng = seed ? seed : 1u;

    sys->n   = n;
    sys->box = cbrt((double)n / density);
    sys->pos   = (Vec3 *)calloc(n, sizeof(Vec3));
    sys->force = (Vec3 *)calloc(n, sizeof(Vec3));

    int side = (int)ceil(cbrt((double)n));
    int site_count = side * side * side;
    int *sites = (int *)malloc((size_t)site_count * sizeof(int));
    double *cx = (double *)malloc((size_t)side * sizeof(double));
    double *cy = (double *)malloc((size_t)side * sizeof(double));
    double *cz = (double *)malloc((size_t)side * sizeof(double));
    double *wx = (double *)malloc((size_t)side * sizeof(double));
    double *wy = (double *)malloc((size_t)side * sizeof(double));
    double *wz = (double *)malloc((size_t)side * sizeof(double));
    if (!sites || !cx || !cy || !cz || !wx || !wy || !wz) {
        fprintf(stderr, "init_particles: failed to allocate lattice state\n");
        exit(EXIT_FAILURE);
    }

    make_axis_centers(cx, wx, side, sys->box, &rng);
    make_axis_centers(cy, wy, side, sys->box, &rng);
    make_axis_centers(cz, wz, side, sys->box, &rng);
    shuffle_sites(sites, site_count, &rng);

    double spacing = sys->box / side;
    double patch = sys->box / 4.0;
    double angle = 0.16 * (2.0 * rng_uniform(&rng) - 1.0);

    for (int i = 0; i < n; i++) {
        int site = sites[i];
        int ix = site / (side * side);
        int iy = (site / side) % side;
        int iz = site % side;

        double jx = 0.045 * wx[ix];
        double jy = 0.045 * wy[iy];
        double jz = 0.045 * wz[iz];
        double dx = jx * (2.0 * rng_uniform(&rng) - 1.0);
        double dy = jy * (2.0 * rng_uniform(&rng) - 1.0);
        double dz = jz * (2.0 * rng_uniform(&rng) - 1.0);

        double x = cx[ix] + dx;
        double y = cy[iy] + dy;
        double z = cz[iz] + dz;

        int px = (int)(x / patch);
        int py = (int)(y / patch);
        int pz = (int)(z / patch);
        unsigned int selector = (unsigned int)(px * 2654435761u) ^
                                (unsigned int)(py * 2246822519u) ^
                                (unsigned int)(pz * 3266489917u) ^
                                (seed ? seed : 1u);
        int mode = (int)(selector % 10u);

        if (mode == 1) {
            double phase = angle + 0.37 * (double)(px * 3 + py * 5 + pz * 7);
            x += 0.035 * spacing * sin(6.28318530718 * y / sys->box + phase);
            y += 0.035 * spacing * cos(6.28318530718 * z / sys->box + phase);
            z += 0.035 * spacing * sin(6.28318530718 * x / sys->box + phase);
        } else if (mode == 2 || mode == 3) {
            double phase = angle + 0.19 * (double)(px * 11 + py * 13 + pz * 17);
            x += 0.055 * spacing * sin(6.28318530718 * (y + z) / sys->box + phase);
            y += 0.055 * spacing * cos(6.28318530718 * (z + x) / sys->box + phase);
            z += 0.055 * spacing * sin(6.28318530718 * (x + y) / sys->box + phase);
        }

        if ((i & 15) == 0) {
            double mix = 0.18;
            double tx = wrap_box(x + mix * spacing * sin((double)(iy + iz + 1)), sys->box);
            double ty = wrap_box(y + mix * spacing * cos((double)(ix + iz + 1)), sys->box);
            double tz = wrap_box(z + mix * spacing * sin((double)(ix + iy + 1)), sys->box);
            x = 0.75 * x + 0.25 * tx;
            y = 0.75 * y + 0.25 * ty;
            z = 0.75 * z + 0.25 * tz;
        }

        sys->pos[i].x = wrap_box(x, sys->box);
        sys->pos[i].y = wrap_box(y, sys->box);
        sys->pos[i].z = wrap_box(z, sys->box);
    }

    free(sites);
    free(cx); free(cy); free(cz);
    free(wx); free(wy); free(wz);
}

void free_particles(ParticleSystem *sys)
{
    free(sys->pos);   sys->pos   = NULL;
    free(sys->force); sys->force = NULL;
}

/* ---- Flush CPU caches by traversing a large dummy buffer ---- */

#define FLUSH_SIZE (256 * 1024 * 1024)  /* 256 MiB — covers large server LLCs */

void flush_cache(void)
{
    volatile char *buf = (volatile char *)malloc(FLUSH_SIZE);
    if (!buf) return;
    for (size_t i = 0; i < FLUSH_SIZE; i += 64)
        buf[i] = (char)i;
    free((void *)buf);
}

/* ---- Compare two force arrays ---- */

int verify_forces(const Vec3 *ref, const Vec3 *test, int n, double tol)
{
    double max_err = 0.0;
    double max_mag = 0.0;
    int    worst   = -1;

    for (int i = 0; i < n; i++) {
        double ex  = ref[i].x - test[i].x;
        double ey  = ref[i].y - test[i].y;
        double ez  = ref[i].z - test[i].z;
        double err = sqrt(ex * ex + ey * ey + ez * ez);
        double mag = sqrt(ref[i].x * ref[i].x +
                          ref[i].y * ref[i].y +
                          ref[i].z * ref[i].z);

        if (mag > max_mag) max_mag = mag;
        if (err > max_err) { max_err = err; worst = i; }
    }

    double rel_err = (max_mag > 0.0) ? max_err / max_mag : max_err;

    printf("  Max absolute error : %.6e  (particle %d)\n", max_err, worst);
    printf("  Max force magnitude: %.6e\n", max_mag);
    printf("  Max relative error : %.6e\n", rel_err);

    if (rel_err < tol) {
        printf("  PASS (relative error < %.1e)\n", tol);
        return 1;
    } else {
        printf("  FAIL (relative error >= %.1e)\n", tol);
        return 0;
    }
}

/* ---- Newton's 3rd law check: sum of all forces should be ~0 ---- */

void check_newton3(const Vec3 *frc, int n)
{
    double sx = 0, sy = 0, sz = 0;
    double max_mag = 0;
    for (int i = 0; i < n; i++) {
        sx += frc[i].x;
        sy += frc[i].y;
        sz += frc[i].z;
        double m = sqrt(frc[i].x*frc[i].x + frc[i].y*frc[i].y + frc[i].z*frc[i].z);
        if (m > max_mag) max_mag = m;
    }
    double sum_mag = sqrt(sx*sx + sy*sy + sz*sz);
    printf("  Net force magnitude: %.6e  (should be ~0)\n", sum_mag);
    printf("  Relative to max    : %.6e\n", max_mag > 0 ? sum_mag / (max_mag * n) : 0);
}
