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
    int *count = (int *)calloc((size_t)pc->ncells, sizeof(int));
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
        int cx = (int)(pos[i].x * inv_cs);
        int cy = (int)(pos[i].y * inv_cs);
        int cz = (int)(pos[i].z * inv_cs);
        if (cx >= ncx) cx = ncx - 1;
        if (cy >= ncy) cy = ncy - 1;
        if (cz >= ncz) cz = ncz - 1;

        int c = cell_flat(cx, cy, cz, ncy, ncz);
        cell_id[i] = c;
#pragma omp atomic update
        count[c]++;
    }

    pc->start[0] = 0;
    for (int c = 0; c < pc->ncells; c++) {
        pc->start[c + 1] = pc->start[c] + count[c];
        count[c] = pc->start[c];
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int dst;
#pragma omp atomic capture
        dst = count[cell_id[i]]++;
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

static inline double min_image_d(double d, double box, double half_box)
{
    if (d > half_box) return d - box;
    if (d < -half_box) return d + box;
    return d;
}

typedef struct {
    __m256d box, half, neghalf, cut, one, two, c24, zero;
} SimdConsts;

static inline void add_cell_pair_vec(const PackedCells *pc,
                                     double *restrict fx,
                                     double *restrict fy,
                                     double *restrict fz,
                                     int c0, int c1,
                                     double box, double half_box,
                                     const SimdConsts *vc)
{
    const double *restrict px = pc->x;
    const double *restrict py = pc->y;
    const double *restrict pz = pc->z;
    const int a0 = pc->start[c0];
    const int a1 = pc->start[c0 + 1];
    const int b0 = pc->start[c1];
    const int b1 = pc->start[c1 + 1];

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
            __m256d rx = _mm256_sub_pd(vxi, _mm256_loadu_pd(px + b));
            __m256d ry = _mm256_sub_pd(vyi, _mm256_loadu_pd(py + b));
            __m256d rz = _mm256_sub_pd(vzi, _mm256_loadu_pd(pz + b));

            __m256d gt = _mm256_cmp_pd(rx, vc->half, _CMP_GT_OQ);
            __m256d lt = _mm256_cmp_pd(rx, vc->neghalf, _CMP_LT_OQ);
            rx = _mm256_sub_pd(rx, _mm256_and_pd(gt, vc->box));
            rx = _mm256_add_pd(rx, _mm256_and_pd(lt, vc->box));
            gt = _mm256_cmp_pd(ry, vc->half, _CMP_GT_OQ);
            lt = _mm256_cmp_pd(ry, vc->neghalf, _CMP_LT_OQ);
            ry = _mm256_sub_pd(ry, _mm256_and_pd(gt, vc->box));
            ry = _mm256_add_pd(ry, _mm256_and_pd(lt, vc->box));
            gt = _mm256_cmp_pd(rz, vc->half, _CMP_GT_OQ);
            lt = _mm256_cmp_pd(rz, vc->neghalf, _CMP_LT_OQ);
            rz = _mm256_sub_pd(rz, _mm256_and_pd(gt, vc->box));
            rz = _mm256_add_pd(rz, _mm256_and_pd(lt, vc->box));

            __m256d r2 = _mm256_fmadd_pd(rx, rx,
                          _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
            __m256d mask = _mm256_cmp_pd(r2, vc->cut, _CMP_LT_OQ);
            r2 = _mm256_blendv_pd(vc->one, r2, mask);

            __m256d r2inv = _mm256_div_pd(vc->one, r2);
            __m256d r6inv = _mm256_mul_pd(r2inv, _mm256_mul_pd(r2inv, r2inv));
            __m256d fscal = _mm256_mul_pd(
                vc->c24,
                _mm256_mul_pd(
                    r2inv,
                    _mm256_sub_pd(_mm256_mul_pd(vc->two,
                                                _mm256_mul_pd(r6inv, r6inv)),
                                  r6inv)));
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
            double rx = min_image_d(xi - px[b], box, half_box);
            double ry = min_image_d(yi - py[b], box, half_box);
            double rz = min_image_d(zi - pz[b], box, half_box);
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

static void compute_packed_forces(ParticleSystem *sys, const PackedCells *pc)
{
    const int ncx = pc->ncx, ncy = pc->ncy, ncz = pc->ncz;
    const int ncyz = ncy * ncz;
    const double box = sys->box;
    const double half_box = 0.5 * box;
    const int n = sys->n;
    const int *restrict order = pc->order;
    SimdConsts vc = {
        _mm256_set1_pd(box),
        _mm256_set1_pd(half_box),
        _mm256_set1_pd(-half_box),
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
        add_cell_pair_vec(pc, fx, fy, fz, c0, c0, box, half_box, &vc);
    }

    for (int oi = 0; oi < 13; oi++) {
        const int dx = off[oi][0];
        const int dy = off[oi][1];
        const int dz = off[oi][2];
        const int axis = dx ? 0 : (dy ? 1 : 2);
        const int limit = axis == 0 ? ncx : (axis == 1 ? ncy : ncz);

        for (int phase = 0; phase < 3; phase++) {
#pragma omp parallel for schedule(static)
            for (int c0 = 0; c0 < pc->ncells; c0++) {
                const int cx = c0 / ncyz;
                const int rem = c0 - cx * ncyz;
                const int cy = rem / ncz;
                const int cz = rem - cy * ncz;
                const int coord = axis == 0 ? cx : (axis == 1 ? cy : cz);

                if (phase < 2) {
                    if (coord >= limit - 1 || (coord & 1) != phase) continue;
                } else {
                    if (coord != limit - 1) continue;
                }

                int nx = cx + dx;
                int ny = cy + dy;
                int nz = cz + dz;
                if (nx < 0) nx += ncx;
                else if (nx >= ncx) nx -= ncx;
                if (ny < 0) ny += ncy;
                else if (ny >= ncy) ny -= ncy;
                if (nz < 0) nz += ncz;
                else if (nz >= ncz) nz -= ncz;

                add_cell_pair_vec(pc, fx, fy, fz, c0,
                                  (nx * ncy + ny) * ncz + nz,
                                  box, half_box, &vc);
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
