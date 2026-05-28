#include "nbody.h"

/* ================================================================
 *  Cell-list based Lennard-Jones force computation — NAIVE BASELINE
 *
 *  This code is CORRECT but deliberately unoptimised:
 *    - AoS data layout (cache-unfriendly for SIMD)
 *    - Linked-list cell list (pointer chasing, poor locality)
 *    - No Newton's 3rd law (each pair counted twice)
 *    - No SIMD, no OpenMP, no NUMA awareness
 *
 *  Students: copy this file to optimized.c and optimise it.
 * ================================================================ */

/* ---- Cell list management ---- */

void cell_list_init(CellList *cl, double box, double cutoff)
{
    cl->ncx = (int)(box / cutoff);
    cl->ncy = cl->ncx;
    cl->ncz = cl->ncx;
    if (cl->ncx < 3) cl->ncx = cl->ncy = cl->ncz = 3;  /* need >=3 for PBC */
    cl->ncells    = cl->ncx * cl->ncy * cl->ncz;
    cl->cell_size = box / cl->ncx;
    cl->head = (int *)malloc(cl->ncells * sizeof(int));
    cl->next = NULL;   /* allocated in build */
}

void cell_list_build(CellList *cl, const ParticleSystem *sys)
{
    int ncx = cl->ncx, ncy = cl->ncy, ncz = cl->ncz;
    double inv_cs = 1.0 / cl->cell_size;

    for (int c = 0; c < cl->ncells; c++)
        cl->head[c] = -1;

    cl->next = (int *)realloc(cl->next, sys->n * sizeof(int));

    for (int i = 0; i < sys->n; i++) {
        int cx = (int)(sys->pos[i].x * inv_cs);
        int cy = (int)(sys->pos[i].y * inv_cs);
        int cz = (int)(sys->pos[i].z * inv_cs);
        if (cx >= ncx) cx = ncx - 1;
        if (cy >= ncy) cy = ncy - 1;
        if (cz >= ncz) cz = ncz - 1;
        int c = cell_flat(cx, cy, cz, ncy, ncz);
        cl->next[i]  = cl->head[c];
        cl->head[c]  = i;
    }
}

void cell_list_free(CellList *cl)
{
    free(cl->head); cl->head = NULL;
    free(cl->next); cl->next = NULL;
}

/* ---- Force computation (cell list, no Newton's 3rd law) ---- */

void compute_forces_baseline(ParticleSystem *sys, CellList *cl)
{
    const int    n   = sys->n;
    const double box = sys->box;
    const Vec3  *pos = sys->pos;
    Vec3        *frc = sys->force;
    const int ncx = cl->ncx, ncy = cl->ncy, ncz = cl->ncz;

    memset(frc, 0, n * sizeof(Vec3));

    for (int cx = 0; cx < ncx; cx++) {
        for (int cy = 0; cy < ncy; cy++) {
            for (int cz = 0; cz < ncz; cz++) {
                int c0 = cell_flat(cx, cy, cz, ncy, ncz);

                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dz = -1; dz <= 1; dz++) {
                            int nx = (cx + dx + ncx) % ncx;
                            int ny = (cy + dy + ncy) % ncy;
                            int nz = (cz + dz + ncz) % ncz;
                            int c1 = cell_flat(nx, ny, nz, ncy, ncz);

                            for (int i = cl->head[c0]; i != -1; i = cl->next[i]) {
                                for (int j = cl->head[c1]; j != -1; j = cl->next[j]) {
                                    if (i == j) continue;

                                    double rx = pbc_dist(pos[i].x - pos[j].x, box);
                                    double ry = pbc_dist(pos[i].y - pos[j].y, box);
                                    double rz = pbc_dist(pos[i].z - pos[j].z, box);
                                    double r2 = rx * rx + ry * ry + rz * rz;

                                    if (r2 < LJ_CUTOFF_SQ) {
                                        double r2inv  = 1.0 / r2;
                                        double r6inv  = r2inv * r2inv * r2inv;
                                        double fscal  = 24.0 * r2inv * (2.0 * r6inv * r6inv - r6inv);
                                        frc[i].x += fscal * rx;
                                        frc[i].y += fscal * ry;
                                        frc[i].z += fscal * rz;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
