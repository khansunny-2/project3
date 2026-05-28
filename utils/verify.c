#include "nbody.h"

/* Compare two binary force files.
 * Usage: ./nbody_verify <file_a> <file_b> [tolerance] */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <file_a> <file_b> [tolerance]\n", argv[0]);
        return 2;
    }

    double tol = 1e-6;
    if (argc > 3) tol = atof(argv[3]);

    /* read file A */
    FILE *fa = fopen(argv[1], "rb");
    if (!fa) { perror(argv[1]); return 2; }
    int na;
    fread(&na, sizeof(int), 1, fa);
    Vec3 *forces_a = (Vec3 *)malloc(na * sizeof(Vec3));
    fread(forces_a, sizeof(Vec3), na, fa);
    fclose(fa);

    /* read file B */
    FILE *fb = fopen(argv[2], "rb");
    if (!fb) { perror(argv[2]); return 2; }
    int nb;
    fread(&nb, sizeof(int), 1, fb);
    Vec3 *forces_b = (Vec3 *)malloc(nb * sizeof(Vec3));
    fread(forces_b, sizeof(Vec3), nb, fb);
    fclose(fb);

    if (na != nb) {
        printf("FAIL: particle count mismatch (%d vs %d)\n", na, nb);
        free(forces_a); free(forces_b);
        return 1;
    }

    printf("Comparing %s vs %s  (N=%d, tol=%.1e)\n", argv[1], argv[2], na, tol);
    int ok = verify_forces(forces_a, forces_b, na, tol);

    printf("\nNewton's 3rd law check (file B):\n");
    check_newton3(forces_b, na);

    free(forces_a);
    free(forces_b);

    return ok ? 0 : 1;
}
