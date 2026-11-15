package io.github.zzuegg.brainiac.backend.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD versions of the hottest loops, written with the incubating Vector API. Only
 * loaded when the {@code jdk.incubator.vector} module is present; see
 * {@link CpuBackend}.
 *
 * <p>Every value is computed lane by lane with fused multiply-add, in the same order for
 * every vector width, so results do not depend on the CPU's SIMD width either.
 */
final class VectorKernels {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = SPECIES.length();
    private static final int ROWS = 4;
    private static final int DEPTH = 256;

    private VectorKernels() {
    }

    /** @return the number of float lanes per vector */
    static int lanes() {
        return LANES;
    }

    /**
     * Rows {@code [from, to)} of {@code z = x * y}; see {@link Kernels#matmulRows}. Works on
     * register tiles of 4 rows by 2 vectors, so each loaded value of {@code x} and
     * {@code y} is used several times before it leaves the registers.
     */
    static void matmulRows(float[] x, float[] y, float[] z, int k, int n, int from, int to) {
        for (int p0 = 0; p0 < k; p0 += DEPTH) {
            int p1 = Math.min(k, p0 + DEPTH);
            int i = from;
            for (; i + ROWS <= to; i += ROWS) {
                fourRows(x, y, z, k, n, i, p0, p1);
            }
            for (; i < to; i++) {
                oneRow(x, y, z, k, n, i, p0, p1);
            }
        }
    }

    private static void fourRows(float[] x, float[] y, float[] z, int k, int n, int i, int p0, int p1) {
        int z0 = i * n;
        int z1 = z0 + n;
        int z2 = z1 + n;
        int z3 = z2 + n;
        int x0 = i * k;
        int x1 = x0 + k;
        int x2 = x1 + k;
        int x3 = x2 + k;
        int j = 0;
        for (; j + 2 * LANES <= n; j += 2 * LANES) {
            int jb = j + LANES;
            FloatVector c00 = FloatVector.fromArray(SPECIES, z, z0 + j);
            FloatVector c01 = FloatVector.fromArray(SPECIES, z, z0 + jb);
            FloatVector c10 = FloatVector.fromArray(SPECIES, z, z1 + j);
            FloatVector c11 = FloatVector.fromArray(SPECIES, z, z1 + jb);
            FloatVector c20 = FloatVector.fromArray(SPECIES, z, z2 + j);
            FloatVector c21 = FloatVector.fromArray(SPECIES, z, z2 + jb);
            FloatVector c30 = FloatVector.fromArray(SPECIES, z, z3 + j);
            FloatVector c31 = FloatVector.fromArray(SPECIES, z, z3 + jb);
            for (int p = p0; p < p1; p++) {
                int yRow = p * n;
                FloatVector b0 = FloatVector.fromArray(SPECIES, y, yRow + j);
                FloatVector b1 = FloatVector.fromArray(SPECIES, y, yRow + jb);
                FloatVector a = FloatVector.broadcast(SPECIES, x[x0 + p]);
                c00 = a.fma(b0, c00);
                c01 = a.fma(b1, c01);
                a = FloatVector.broadcast(SPECIES, x[x1 + p]);
                c10 = a.fma(b0, c10);
                c11 = a.fma(b1, c11);
                a = FloatVector.broadcast(SPECIES, x[x2 + p]);
                c20 = a.fma(b0, c20);
                c21 = a.fma(b1, c21);
                a = FloatVector.broadcast(SPECIES, x[x3 + p]);
                c30 = a.fma(b0, c30);
                c31 = a.fma(b1, c31);
            }
            c00.intoArray(z, z0 + j);
            c01.intoArray(z, z0 + jb);
            c10.intoArray(z, z1 + j);
            c11.intoArray(z, z1 + jb);
            c20.intoArray(z, z2 + j);
            c21.intoArray(z, z2 + jb);
            c30.intoArray(z, z3 + j);
            c31.intoArray(z, z3 + jb);
        }
        for (; j < n; j += LANES) {
            VectorMask<Float> m = SPECIES.indexInRange(j, n);
            FloatVector c0 = FloatVector.fromArray(SPECIES, z, z0 + j, m);
            FloatVector c1 = FloatVector.fromArray(SPECIES, z, z1 + j, m);
            FloatVector c2 = FloatVector.fromArray(SPECIES, z, z2 + j, m);
            FloatVector c3 = FloatVector.fromArray(SPECIES, z, z3 + j, m);
            for (int p = p0; p < p1; p++) {
                FloatVector b = FloatVector.fromArray(SPECIES, y, p * n + j, m);
                c0 = FloatVector.broadcast(SPECIES, x[x0 + p]).fma(b, c0);
                c1 = FloatVector.broadcast(SPECIES, x[x1 + p]).fma(b, c1);
                c2 = FloatVector.broadcast(SPECIES, x[x2 + p]).fma(b, c2);
                c3 = FloatVector.broadcast(SPECIES, x[x3 + p]).fma(b, c3);
            }
            c0.intoArray(z, z0 + j, m);
            c1.intoArray(z, z1 + j, m);
            c2.intoArray(z, z2 + j, m);
            c3.intoArray(z, z3 + j, m);
        }
    }

    private static void oneRow(float[] x, float[] y, float[] z, int k, int n, int i, int p0, int p1) {
        int zRow = i * n;
        int xRow = i * k;
        for (int j = 0; j < n; j += LANES) {
            VectorMask<Float> m = SPECIES.indexInRange(j, n);
            FloatVector c = FloatVector.fromArray(SPECIES, z, zRow + j, m);
            for (int p = p0; p < p1; p++) {
                FloatVector b = FloatVector.fromArray(SPECIES, y, p * n + j, m);
                c = FloatVector.broadcast(SPECIES, x[xRow + p]).fma(b, c);
            }
            c.intoArray(z, zRow + j, m);
        }
    }
}
