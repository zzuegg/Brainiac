package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class ActivationKernelsTest {

    private static float[] inputs(int n) {
        SplittableRandom r = new SplittableRandom(7);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) r.nextDouble(-8, 8);
        }
        // edge cases, including the kinks
        float[] special = {0f, -0f, 1f, -1f, 3f, -3f, 6f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            1e-30f, -1e-30f, 80f, -80f};
        System.arraycopy(special, 0, x, 0, special.length);
        return x;
    }

    /** f'(x) by central differences in double, from the kernel's own f. */
    private static double numericDerivative(ActivationOp op, float p, double x) {
        double h = 1e-3;
        float[] in = {(float) (x + h), (float) (x - h)};
        float[] out = new float[2];
        ActivationKernels.forward(op, p, in, out, 0, 2);
        return (out[0] - out[1]) / ((double) in[0] - in[1]);
    }

    @ParameterizedTest
    @EnumSource(ActivationOp.class)
    void derivativeMatchesTheFunction(ActivationOp op) {
        float p = 0.3f;
        float[] x = inputs(200);
        float[] ones = new float[x.length];
        java.util.Arrays.fill(ones, 1f);
        float[] d = new float[x.length];
        ActivationKernels.backward(op, p, x, ones, d, 0, x.length);
        for (int i = 14; i < x.length; i++) {
            double xi = x[i];
            // skip points next to a kink
            if (Math.abs(xi) < 0.01 || Math.abs(Math.abs(xi) - 1) < 0.01 || Math.abs(Math.abs(xi) - 3) < 0.01
                    || Math.abs(xi - 6) < 0.01) {
                continue;
            }
            double numeric = numericDerivative(op, p, xi);
            assertEquals(numeric, d[i], 2e-3 * Math.max(1, Math.abs(numeric)), op + " at " + xi);
        }
    }

    @ParameterizedTest
    @EnumSource(ActivationOp.class)
    void simdGivesIdenticalBits(ActivationOp op) {
        if (!CpuBackend.simd() || !VectorActivationKernels.supports(op)) {
            return;
        }
        float[] x = inputs(1037);
        float[] g = inputs(1037);
        float[] scalar = new float[x.length];
        float[] vector = new float[x.length];
        ActivationKernels.forward(op, 0.2f, x, scalar, 0, x.length);
        VectorActivationKernels.forward(op, 0.2f, x, vector, 3, x.length);
        ActivationKernels.forward(op, 0.2f, x, vector, 0, 3);
        assertArrayEquals(scalar, vector, op + " forward");
        ActivationKernels.backward(op, 0.2f, x, g, scalar, 0, x.length);
        VectorActivationKernels.backward(op, 0.2f, x, g, vector, 0, x.length);
        assertArrayEquals(scalar, vector, op + " backward");
    }

    @Test
    void simdArithmeticGivesIdenticalBits() {
        if (!CpuBackend.simd()) {
            return;
        }
        float[] x = inputs(515);
        float[] y = inputs(515);
        for (int i = 0; i < y.length; i++) {
            y[i] = y[(i * 7) % y.length] + 0.5f;
        }
        for (BinaryOp op : new BinaryOp[] {BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV, BinaryOp.MAX,
            BinaryOp.MIN}) {
            float[] scalar = new float[x.length];
            float[] vector = new float[x.length];
            Kernels.binary(op, x, y, scalar, 0, x.length);
            VectorActivationKernels.binary(op, x, y, vector, 0, x.length);
            assertArrayEquals(scalar, vector, op.toString());
            Kernels.binaryRows(op, x, y, scalar, 103, 0, 5);
            VectorActivationKernels.binaryRows(op, x, y, vector, 103, 0, 5);
            assertArrayEquals(scalar, vector, op + " rows");
            Kernels.binaryScalarRight(op, x, 1.7f, scalar, 0, x.length);
            VectorActivationKernels.binaryScalarRight(op, x, 1.7f, vector, 0, x.length);
            assertArrayEquals(scalar, vector, op + " scalar");
        }
    }

    @Test
    void tensorApiAndKnownValues() {
        Tensor x = Tensor.of(-1f, 0.5f, 2f);
        assertArrayEquals(new float[] {-0.1f, 0.5f, 2f}, x.activation(ActivationOp.LEAKY_RELU, 0.1f).toArray(), 1e-7f);
        assertArrayEquals(new float[] {-0.15865526f, 0.34573123f, 1.9544997f},
                x.activation(ActivationOp.GELU, 0).toArray(), 1e-6f);
        assertArrayEquals(new float[] {(float) (0.5 * Math.expm1(-1)), 0.5f, 2f},
                x.activation(ActivationOp.ELU, 0.5f).toArray(), 1e-7f);
        Tensor g = Tensor.of(2f, 2f, 2f);
        assertArrayEquals(new float[] {0.2f, 2f, 2f},
                x.activationBackward(ActivationOp.LEAKY_RELU, 0.1f, g).toArray(), 1e-7f);
        for (ActivationOp op : ActivationOp.values()) {
            for (float v : Tensor.of(-500f, 500f).activation(op, 1f).toArray()) {
                assertTrue(!Float.isNaN(v), op + " gave NaN");
            }
        }
    }
}
