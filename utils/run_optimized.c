#include "nbody.h"

/* Run student optimized, write forces to binary file.
 * Prints compute time (seconds) to stderr for benchmarking.
 * Usage: ./nbody_optimized [N] [density] [seed] [output_file] */

int main(int argc, char **argv)
{
    int    n       = 4096;
    double density = 0.5;
    unsigned int seed = 42;
    const char *outfile = "forces_optimized.bin";

    if (argc > 1) n       = atoi(argv[1]);
    if (argc > 2) density = atof(argv[2]);
    if (argc > 3) seed    = (unsigned int)atoi(argv[3]);
    if (argc > 4) outfile = argv[4];

    ParticleSystem sys;
    init_particles(&sys, n, density, seed);

    flush_cache();

    double t0 = wtime();
    compute_forces_optimized(&sys);
    double t1 = wtime();

    fprintf(stderr, "%.6f\n", t1 - t0);

    /* write forces */
    FILE *fp = fopen(outfile, "wb");
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(sys.force, sizeof(Vec3), n, fp);
    fclose(fp);

    free_particles(&sys);
    return 0;
}
