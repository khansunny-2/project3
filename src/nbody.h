#ifndef NBODY_H
#define NBODY_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* LJ parameters in reduced units (epsilon=1, sigma=1) */
#define LJ_CUTOFF    2.5
#define LJ_CUTOFF_SQ (LJ_CUTOFF * LJ_CUTOFF)

typedef struct {
    double x, y, z;
} Vec3;

/* Particle system */
typedef struct {
    Vec3   *pos;       /* positions */
    Vec3   *force;     /* forces    */
    int     n;         /* number of particles */
    double  box;       /* cubic box side length */
} ParticleSystem;

/* Cell list (linked-list based — deliberately cache-unfriendly) */
typedef struct {
    int    *head;       /* head[cell] = first particle, or -1 */
    int    *next;       /* next[i]    = next particle in same cell, or -1 */
    int     ncx, ncy, ncz;   /* cells per dimension */
    int     ncells;           /* total number of cells */
    double  cell_size;        /* actual cell side length (>= cutoff) */
} CellList;

/* ---- inline helpers ---- */

static inline double pbc_dist(double d, double box)
{
    return d - box * round(d / box);
}

static inline int cell_flat(int cx, int cy, int cz, int ncy, int ncz)
{
    return (cx * ncy + cy) * ncz + cz;
}

/* ---- common.c ---- */
double wtime(void);
void   init_particles(ParticleSystem *sys, int n, double density, unsigned int seed);
void   free_particles(ParticleSystem *sys);
int    verify_forces(const Vec3 *ref, const Vec3 *test, int n, double tol);
void   check_newton3(const Vec3 *frc, int n);
void   flush_cache(void);

/* ---- baseline.c ---- */
void cell_list_init (CellList *cl, double box, double cutoff);
void cell_list_build(CellList *cl, const ParticleSystem *sys);
void cell_list_free (CellList *cl);
void compute_forces_baseline(ParticleSystem *sys, CellList *cl);

/* ---- optimized.c (student implementation) ---- */
void compute_forces_optimized(ParticleSystem *sys);

/* ---- bruteforce.c ---- */
void compute_forces_bruteforce(ParticleSystem *sys);

#endif /* NBODY_H */
