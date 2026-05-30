#pragma GCC optimize("O3,unroll-loops,fast-math")
#include "nbody.h"
#include <immintrin.h>
#include <omp.h>

/* ================================================================
 *  Optimised force computation.
 *
 *  The baseline linked-list cell layout is intentionally unfriendly to cache
 *  and cannot be parallelised directly without write conflicts.  This version
 *  stores each cell as a contiguous range, evaluates unique cell pairs with
 *  Newton's 3rd law, and uses phase scheduling so parallel threads never
 *  update the same cell in the same phase.
 * ================================================================ */

typedef struct {
    int ncx, ncy, ncz;
    int ncells;
    double cell_size;
    int *start;      /* start[c]..start[c + 1] is the packed range for c */
    int *order;      /* original particle index for each packed entry */
    double *x, *y, *z;
} PackedCells;

#ifndef FORCE_CHUNK
#define FORCE_CHUNK 128
#endif

#ifndef SELF_CHUNK
#define SELF_CHUNK 128
#endif

#ifndef PAIR_CHUNK
#define PAIR_CHUNK 512
#endif

#ifndef RECIP_NR_STEPS
#define RECIP_NR_STEPS 2
#endif

#ifndef PRIVATE_COUNT_MAX_THREADS
#define PRIVATE_COUNT_MAX_THREADS 32
#endif

static void die_oom(void)
{
    fprintf(stderr, "compute_forces_optimized: out of memory\n");
    exit(EXIT_FAILURE);
}

static void pc_init(PackedCells *pc, double box, double cutoff)
{
    pc->ncx = (int)(box / cutoff);
    pc->ncy = pc->ncx;
    pc->ncz = pc->ncx;
    if (pc->ncx < 3) pc->ncx = pc->ncy = pc->ncz = 3;

    pc->ncells = pc->ncx * pc->ncy * pc->ncz;
    pc->cell_size = box / pc->ncx;
    pc->start = NULL;
    pc->order = NULL;
    pc->x = pc->y = pc->z = NULL;
}

static void pc_free(PackedCells *pc)
{
    free(pc->start);
    free(pc->order);
    free(pc->x);
    free(pc->y);
    free(pc->z);
    pc->start = NULL;
    pc->order = NULL;
    pc->x = pc->y = pc->z = NULL;
}

static void pc_build(PackedCells *pc, const ParticleSystem *sys)
{
    const int n = sys->n;
    const int ncx = pc->ncx, ncy = pc->ncy, ncz = pc->ncz;
    const double inv_cs = 1.0 / pc->cell_size;
    const Vec3 *pos = sys->pos;

    int *cell_id = (int *)malloc((size_t)n * sizeof(int));
    int nthreads = omp_get_max_threads();
    const int use_private_count = (nthreads <= PRIVATE_COUNT_MAX_THREADS);
    size_t count_len = use_private_count
        ? (size_t)nthreads * (size_t)pc->ncells
        : (size_t)pc->ncells;
    int *count = (int *)calloc(count_len, sizeof(int));
    pc->start = (int *)malloc((size_t)(pc->ncells + 1) * sizeof(int));
    pc->order = (int *)malloc((size_t)n * sizeof(int));
    pc->x = (double *)malloc((size_t)n * sizeof(double));
    pc->y = (double *)malloc((size_t)n * sizeof(double));
    pc->z = (double *)malloc((size_t)n * sizeof(double));
    if (!cell_id || !count || !pc->start || !pc->order ||
        !pc->x || !pc->y || !pc->z) {
        free(cell_id);
        free(count);
        pc_free(pc);
        die_oom();
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        int cx = (int)(pos[i].x * inv_cs);
        int cy = (int)(pos[i].y * inv_cs);
        int cz = (int)(pos[i].z * inv_cs);
        if (cx >= ncx) cx = ncx - 1;
        if (cy >= ncy) cy = ncy - 1;
        if (cz >= ncz) cz = ncz - 1;

        int c = cell_flat(cx, cy, cz, ncy, ncz);
        cell_id[i] = c;
        if (use_private_count) {
            count[(size_t)tid * (size_t)pc->ncells + (size_t)c]++;
        } else {
#pragma omp atomic update
            count[c]++;
        }
    }

    if (use_private_count) {
        int running = 0;
        for (int c = 0; c < pc->ncells; c++) {
            pc->start[c] = running;
            for (int t = 0; t < nthreads; t++) {
                int cnt = count[(size_t)t * (size_t)pc->ncells + (size_t)c];
                count[(size_t)t * (size_t)pc->ncells + (size_t)c] = running;
                running += cnt;
            }
        }
        pc->start[pc->ncells] = running;
    } else {
        pc->start[0] = 0;
        for (int c = 0; c < pc->ncells; c++) {
            pc->start[c + 1] = pc->start[c] + count[c];
            count[c] = pc->start[c];
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        int c = cell_id[i];
        int dst;
        if (use_private_count) {
            dst = count[(size_t)tid * (size_t)pc->ncells + (size_t)c]++;
        } else {
#pragma omp atomic capture
            dst = count[c]++;
        }
        pc->order[dst] = i;
        pc->x[dst] = pos[i].x;
        pc->y[dst] = pos[i].y;
        pc->z[dst] = pos[i].z;
    }

    free(cell_id);
    free(count);
}

static inline double hsum4(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum);
}

static inline __m256d recip4_pd(__m256d x)
{
    __m128 xf = _mm256_cvtpd_ps(x);
    __m256d y = _mm256_cvtps_pd(_mm_rcp_ps(xf));
#if RECIP_NR_STEPS >= 1
    const __m256d two = _mm256_set1_pd(2.0);
    y = _mm256_mul_pd(y, _mm256_sub_pd(two, _mm256_mul_pd(x, y)));
#endif
#if RECIP_NR_STEPS >= 2
    y = _mm256_mul_pd(y, _mm256_sub_pd(two, _mm256_mul_pd(x, y)));
#endif
    return y;
}

typedef struct {
    __m256d cut, one, two, c24, zero;
} SimdConsts;

static inline void add_cell_pair_vec(const PackedCells *pc,
                                     double *restrict fx,
                                     double *restrict fy,
                                     double *restrict fz,
                                     int c0, int c1,
                                     double sx, double sy, double sz,
                                     const SimdConsts *vc)
{
    const double *restrict px = pc->x;
    const double *restrict py = pc->y;
    const double *restrict pz = pc->z;
    const int a0 = pc->start[c0];
    const int a1 = pc->start[c0 + 1];
    const int b0 = pc->start[c1];
    const int b1 = pc->start[c1 + 1];
    const __m256d vsx = _mm256_set1_pd(sx);
    const __m256d vsy = _mm256_set1_pd(sy);
    const __m256d vsz = _mm256_set1_pd(sz);

    for (int a = a0; a < a1; a++) {
        const double xi = px[a];
        const double yi = py[a];
        const double zi = pz[a];
        const __m256d vxi = _mm256_set1_pd(xi);
        const __m256d vyi = _mm256_set1_pd(yi);
        const __m256d vzi = _mm256_set1_pd(zi);
        __m256d vfxa = vc->zero;
        __m256d vfya = vc->zero;
        __m256d vfza = vc->zero;
        double sfx = 0.0, sfy = 0.0, sfz = 0.0;
        int b = (c0 == c1) ? a + 1 : b0;

        for (; b + 3 < b1; b += 4) {
            __m256d rx = _mm256_add_pd(_mm256_sub_pd(vxi, _mm256_loadu_pd(px + b)), vsx);
            __m256d ry = _mm256_add_pd(_mm256_sub_pd(vyi, _mm256_loadu_pd(py + b)), vsy);
            __m256d rz = _mm256_add_pd(_mm256_sub_pd(vzi, _mm256_loadu_pd(pz + b)), vsz);

            __m256d r2 = _mm256_fmadd_pd(rx, rx,
                          _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
            __m256d mask = _mm256_cmp_pd(r2, vc->cut, _CMP_LT_OQ);
            int maskbits = _mm256_movemask_pd(mask);
            if (maskbits == 0) continue;
            r2 = _mm256_blendv_pd(vc->one, r2, mask);
            __m256d r2inv = recip4_pd(r2);
            __m256d r6inv = _mm256_mul_pd(r2inv, _mm256_mul_pd(r2inv, r2inv));
            __m256d term = _mm256_fmsub_pd(vc->two,
                                           _mm256_mul_pd(r6inv, r6inv),
                                           r6inv);
            __m256d fscal = _mm256_mul_pd(_mm256_mul_pd(vc->c24, r2inv),
                                          term);
            fscal = _mm256_and_pd(fscal, mask);

            __m256d tx = _mm256_mul_pd(fscal, rx);
            __m256d ty = _mm256_mul_pd(fscal, ry);
            __m256d tz = _mm256_mul_pd(fscal, rz);
            vfxa = _mm256_add_pd(vfxa, tx);
            vfya = _mm256_add_pd(vfya, ty);
            vfza = _mm256_add_pd(vfza, tz);
            _mm256_storeu_pd(fx + b, _mm256_sub_pd(_mm256_loadu_pd(fx + b), tx));
            _mm256_storeu_pd(fy + b, _mm256_sub_pd(_mm256_loadu_pd(fy + b), ty));
            _mm256_storeu_pd(fz + b, _mm256_sub_pd(_mm256_loadu_pd(fz + b), tz));
        }

        for (; b < b1; b++) {
            double rx = xi - px[b] + sx;
            double ry = yi - py[b] + sy;
            double rz = zi - pz[b] + sz;
            double r2 = rx * rx + ry * ry + rz * rz;

            if (r2 < LJ_CUTOFF_SQ) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double fscal = 24.0 * r2inv *
                               (2.0 * r6inv * r6inv - r6inv);
                double tx = fscal * rx;
                double ty = fscal * ry;
                double tz = fscal * rz;
                sfx += tx;
                sfy += ty;
                sfz += tz;
                fx[b] -= tx;
                fy[b] -= ty;
                fz[b] -= tz;
            }
        }

        fx[a] += hsum4(vfxa) + sfx;
        fy[a] += hsum4(vfya) + sfy;
        fz[a] += hsum4(vfza) + sfz;
    }
}

static inline void add_offset_cell(const PackedCells *pc,
                                   double *restrict fx,
                                   double *restrict fy,
                                   double *restrict fz,
                                   int cx, int cy, int cz,
                                   int dx, int dy, int dz,
                                   int ncx, int ncy, int ncz,
                                   double box,
                                   const SimdConsts *vc)
{
    int nx = cx + dx;
    int ny = cy + dy;
    int nz = cz + dz;
    const double sx = (nx < 0) ? box : ((nx >= ncx) ? -box : 0.0);
    const double sy = (ny < 0) ? box : ((ny >= ncy) ? -box : 0.0);
    const double sz = (nz < 0) ? box : ((nz >= ncz) ? -box : 0.0);
    if (nx < 0) nx += ncx;
    else if (nx >= ncx) nx -= ncx;
    if (ny < 0) ny += ncy;
    else if (ny >= ncy) ny -= ncy;
    if (nz < 0) nz += ncz;
    else if (nz >= ncz) nz -= ncz;

    add_cell_pair_vec(pc, fx, fy, fz,
                      (cx * ncy + cy) * ncz + cz,
                      (nx * ncy + ny) * ncz + nz,
                      sx, sy, sz, vc);
}

static void compute_packed_forces(ParticleSystem *sys, const PackedCells *pc)
{
    const int ncx = pc->ncx, ncy = pc->ncy, ncz = pc->ncz;
    const double box = sys->box;
    const int n = sys->n;
    const int *restrict order = pc->order;
    SimdConsts vc = {
        _mm256_set1_pd(LJ_CUTOFF_SQ),
        _mm256_set1_pd(1.0),
        _mm256_set1_pd(2.0),
        _mm256_set1_pd(24.0),
        _mm256_setzero_pd()
    };
    static const int off[13][3] = {
        {0, 0, 1},
        {0, 1, -1}, {0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0}, {1, -1, 1},
        {1, 0, -1},  {1, 0, 0},  {1, 0, 1},
        {1, 1, -1},  {1, 1, 0},  {1, 1, 1}
    };

    double *fx = (double *)malloc((size_t)n * sizeof(double));
    double *fy = (double *)malloc((size_t)n * sizeof(double));
    double *fz = (double *)malloc((size_t)n * sizeof(double));
    if (!fx || !fy || !fz) {
        free(fx);
        free(fy);
        free(fz);
        die_oom();
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

#pragma omp parallel for schedule(static)
    for (int c0 = 0; c0 < pc->ncells; c0++) {
        add_cell_pair_vec(pc, fx, fy, fz, c0, c0, 0.0, 0.0, 0.0, &vc);
    }

    for (int oi = 0; oi < 13; oi++) {
        const int dx = off[oi][0];
        const int dy = off[oi][1];
        const int dz = off[oi][2];
        const int axis = dx ? 0 : (dy ? 1 : 2);
        const int limit = axis == 0 ? ncx : (axis == 1 ? ncy : ncz);

        for (int phase = 0; phase < 3; phase++) {
            const int start = (phase < 2) ? phase : limit - 1;
            const int stop = (phase < 2) ? limit - 1 : limit;
            const int step = (phase < 2) ? 2 : 1;

            if (axis == 0) {
#pragma omp parallel for collapse(3) schedule(static)
                for (int cx = start; cx < stop; cx += step) {
                    for (int cy = 0; cy < ncy; cy++) {
                        for (int cz = 0; cz < ncz; cz++) {
                            add_offset_cell(pc, fx, fy, fz, cx, cy, cz,
                                            dx, dy, dz, ncx, ncy, ncz, box, &vc);
                        }
                    }
                }
            } else if (axis == 1) {
#pragma omp parallel for collapse(3) schedule(static)
                for (int cx = 0; cx < ncx; cx++) {
                    for (int cy = start; cy < stop; cy += step) {
                        for (int cz = 0; cz < ncz; cz++) {
                            add_offset_cell(pc, fx, fy, fz, cx, cy, cz,
                                            dx, dy, dz, ncx, ncy, ncz, box, &vc);
                        }
                    }
                }
            } else {
#pragma omp parallel for collapse(3) schedule(static)
                for (int cx = 0; cx < ncx; cx++) {
                    for (int cy = 0; cy < ncy; cy++) {
                        for (int cz = start; cz < stop; cz += step) {
                            add_offset_cell(pc, fx, fy, fz, cx, cy, cz,
                                            dx, dy, dz, ncx, ncy, ncz, box, &vc);
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int a = 0; a < n; a++) {
        const int i = order[a];
        sys->force[i].x = fx[a];
        sys->force[i].y = fy[a];
        sys->force[i].z = fz[a];
    }

    free(fx);
    free(fy);
    free(fz);
}

void compute_forces_optimized(ParticleSystem *sys)
{
    PackedCells pc;
    pc_init(&pc, sys->box, LJ_CUTOFF);
    pc_build(&pc, sys);
    compute_packed_forces(sys, &pc);
    pc_free(&pc);
}
