#include "nbody.h"

/* ================================================================
 *  O(N^2) brute-force LJ force computation.
 *  Used ONLY for small-N correctness verification:
 *      make correctness
 * ================================================================ */

void compute_forces_bruteforce(ParticleSystem *sys)
{
    const int    n   = sys->n;
    const double box = sys->box;
    const Vec3  *pos = sys->pos;
    Vec3        *frc = sys->force;

    memset(frc, 0, n * sizeof(Vec3));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double rx = pbc_dist(pos[i].x - pos[j].x, box);
            double ry = pbc_dist(pos[i].y - pos[j].y, box);
            double rz = pbc_dist(pos[i].z - pos[j].z, box);
            double r2 = rx * rx + ry * ry + rz * rz;

            if (r2 < LJ_CUTOFF_SQ) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double fscal = 24.0 * r2inv * (2.0 * r6inv * r6inv - r6inv);
                frc[i].x += fscal * rx;
                frc[i].y += fscal * ry;
                frc[i].z += fscal * rz;
            }
        }
    }
}

/* ---- main: brute-force vs baseline on small N ---- */

int main(int argc, char **argv)
{
    int    n       = 512;    /* small default for brute-force */
    double density = 0.5;
    unsigned int seed = 42;

    if (argc > 1) n       = atoi(argv[1]);
    if (argc > 2) density = atof(argv[2]);
    if (argc > 3) seed    = (unsigned int)atoi(argv[3]);

    printf("=== Correctness Check (brute-force vs baseline) ===\n");
    printf("Particles : %d\n", n);
    printf("Density   : %.2f\n", density);
    printf("Cutoff    : %.2f\n", LJ_CUTOFF);
    printf("Seed      : %u\n\n", seed);

    ParticleSystem sys;
    init_particles(&sys, n, density, seed);
    printf("Box size  : %.4f\n\n", sys.box);

    /* ---- Brute-force ---- */
    printf("[1] Brute-force (O(N^2)) ...\n");
    double t0 = wtime();
    compute_forces_bruteforce(&sys);
    double t1 = wtime();
    printf("  Time: %.4f s\n", t1 - t0);

    Vec3 *ref_force = (Vec3 *)malloc(n * sizeof(Vec3));
    memcpy(ref_force, sys.force, n * sizeof(Vec3));

    printf("  Newton's 3rd law check:\n");
    check_newton3(ref_force, n);

    /* ---- Baseline (cell list) ---- */
    printf("\n[2] Baseline (cell list) ...\n");
    CellList cl;
    cell_list_init(&cl, sys.box, LJ_CUTOFF);
    printf("  Cells: %d x %d x %d = %d  (cell size %.4f)\n",
           cl.ncx, cl.ncy, cl.ncz, cl.ncells, cl.cell_size);
    cell_list_build(&cl, &sys);

    t0 = wtime();
    compute_forces_baseline(&sys, &cl);
    t1 = wtime();
    printf("  Time: %.4f s\n", t1 - t0);

    printf("  Newton's 3rd law check:\n");
    check_newton3(sys.force, n);

    /* ---- Verify ---- */
    printf("\n[3] Verification (baseline vs brute-force):\n");
    int ok = verify_forces(ref_force, sys.force, n, 1e-10);

    cell_list_free(&cl);
    free(ref_force);
    free_particles(&sys);

    return ok ? 0 : 1;
}
