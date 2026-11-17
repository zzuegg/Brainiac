package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EquivariantGraphLayerTest {
    private static final int N = 5;
    private static final int F = 3;

    /** Rotation about an axis by Rodrigues' formula, then a reflection of x, then a translation. */
    private static float[] transform(float[] in, double[][] r, double[] shift) {
        float[] out = in.clone();
        for (int i = 0; i < N; i++) {
            int o = i * (3 + F);
            for (int a = 0; a < 3; a++) {
                double v = shift[a];
                for (int c = 0; c < 3; c++) {
                    v += r[a][c] * in[o + c];
                }
                out[o + a] = (float) v;
            }
        }
        return out;
    }

    private static double[][] orthogonal() {
        double t = 0.9;
        double[] k = {1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)};
        double c = Math.cos(t);
        double s = Math.sin(t);
        double[][] r = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] = (i == j ? c : 0) + (1 - c) * k[i] * k[j];
            }
        }
        r[0][1] -= s * k[2];
        r[0][2] += s * k[1];
        r[1][0] += s * k[2];
        r[1][2] -= s * k[0];
        r[2][0] -= s * k[1];
        r[2][1] += s * k[0];
        for (int j = 0; j < 3; j++) {
            r[0][j] = -r[0][j];   // add a reflection: E(3), not just SE(3)
        }
        return r;
    }

    @Test
    void stackedLayersAreRotationReflectionAndTranslationEquivariant() {
        EquivariantGraphLayer a = EquivariantGraphLayer.of(3, F, 8, 1);
        EquivariantGraphLayer b = EquivariantGraphLayer.of(3, F, 8, 2);
        // larger coordinate weights than the default so the coordinate update is not negligible
        a.parameters().get(6).assign(Tensor.normal(Shape.of(8, 1), 0, 1, new SplittableRandom(3)));
        float[] in = Tensor.normal(Shape.of(1, N, 3 + F), 0, 1, new SplittableRandom(4)).toArray();
        double[][] r = orthogonal();
        double[] shift = {3, -1, 2};
        float[] out = b.forward(a.forward(Var.of(Tensor.of(Shape.of(1, N, 3 + F), in)))).value().toArray();
        float[] outOfMoved = b.forward(a.forward(Var.of(Tensor.of(Shape.of(1, N, 3 + F), transform(in, r, shift)))))
                .value().toArray();
        float[] movedOut = transform(out, r, shift);
        double moved = 0;
        for (int i = 0; i < in.length; i++) {
            assertEquals(movedOut[i], outOfMoved[i], 1e-4, "element " + i);
            moved += Math.abs(out[i] - in[i]);
        }
        assertTrue(moved > 0.1, "layers should change coordinates and features");
    }

    @Test
    void gradientsAreCorrect() {
        EquivariantGraphLayer a = EquivariantGraphLayer.of(2, 2, 4, 5);
        Tensor x = Tensor.normal(Shape.of(2, 3, 4), 0, 1, new SplittableRandom(6));
        GradCheck.Result g = GradCheck.check(v -> a.forward(v.get(0)).square().sum(), List.of(x), 1e-3f, 2e-2f);
        assertTrue(g.passed(), g.toString());
    }
}
