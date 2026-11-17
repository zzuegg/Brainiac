package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** SLERP, Fisher-weighted averaging and RegMean on the two-skill merging task of ModelMergingTest. */
class MergingMethodsTest {

    @Test
    void slerpKeepsTheNormBetweenOrthogonalVectors() {
        List<Tensor> a = List.of(Tensor.of(1f, 0f));
        List<Tensor> b = List.of(Tensor.of(0f, 1f));
        float[] mid = ModelMerging.slerp(a, b, 0.5).get(0).toArray();
        assertArrayEquals(new float[] {(float) Math.sqrt(0.5), (float) Math.sqrt(0.5)}, mid, 1e-6f);
        assertArrayEquals(new float[] {1, 0}, ModelMerging.slerp(a, b, 0).get(0).toArray(), 1e-6f);
    }

    @Test
    void fisherAndRegMeanHaveTheirClosedForms() {
        List<Tensor> m = ModelMerging.fisher(List.of(List.of(Tensor.of(1f, 1f)), List.of(Tensor.of(3f, 3f))),
                List.of(List.of(Tensor.of(1f, 0f)), List.of(Tensor.of(1f, 2f))), 0);
        assertArrayEquals(new float[] {2, 3}, m.get(0).toArray(), 1e-6f);
        // one-dimensional regression: grams 1 and 3 -> (1 * 2 + 3 * 6) / 4 = 5
        Tensor w = ModelMerging.regMean(List.of(Tensor.of(Shape.of(1, 1), 2f), Tensor.of(Shape.of(1, 1), 6f)),
                List.of(Tensor.of(Shape.of(1, 1), 1f), Tensor.of(Shape.of(1, 1), 3f)), 1);
        assertEquals(5, w.toArray()[0], 1e-4);
    }

    private static double loss(float[] w, double[][] x, double[] y) {
        double l = 0;
        for (int i = 0; i < x.length; i++) {
            double p = 0;
            for (int j = 0; j < w.length; j++) {
                p += w[j] * x[i][j];
            }
            l += (p - y[i]) * (p - y[i]);
        }
        return l / x.length;
    }

    private static float[] fit(double[][] x, double[] y, SplittableRandom r) {
        double[] w = new double[x[0].length];
        for (int it = 0; it < 400; it++) {
            double[] g = new double[w.length];
            for (int i = 0; i < x.length; i++) {
                double e = -y[i];
                for (int j = 0; j < w.length; j++) {
                    e += w[j] * x[i][j];
                }
                for (int j = 0; j < w.length; j++) {
                    g[j] += 2 * e * x[i][j] / x.length;
                }
            }
            for (int j = 0; j < w.length; j++) {
                w[j] -= 0.05 * g[j] + 0.002 * r.nextGaussian();
            }
        }
        float[] f = new float[w.length];
        for (int j = 0; j < w.length; j++) {
            f[j] = (float) w[j];
        }
        return f;
    }

    private static Tensor matrix(double[][] x) {
        float[] f = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                f[i * x[0].length + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, x[0].length), f);
    }

    /** The output Fisher of a linear model with unit Gaussian noise: per-example gradients {@code x}. */
    private static List<Tensor> fisher(double[][] x) {
        List<List<Tensor>> grads = new ArrayList<>();
        for (double[] row : x) {
            float[] g = new float[row.length];
            for (int j = 0; j < row.length; j++) {
                g[j] = (float) row[j];
            }
            grads.add(List.of(Tensor.of(g)));
        }
        return ModelMerging.diagonalFisher(grads);
    }

    @Test
    void mergersOnTwoSkills() {
        SplittableRandom r = new SplittableRandom(1);
        int d = 20;
        int n = 200;
        // two skills over shared inputs with different strengths: task A mostly uses features
        // 0-9, task B mostly 10-19, and both see a little of everything
        double[][] xa = new double[n][d];
        double[][] xb = new double[n][d];
        double[] ya = new double[n];
        double[] yb = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                xa[i][j] = r.nextGaussian() * (j < 10 ? 1 : 0.1);
                xb[i][j] = r.nextGaussian() * (j < 10 ? 0.1 : 1);
            }
            for (int j = 0; j < 5; j++) {
                ya[i] += (j + 1) * xa[i][j];
                yb[i] += (j + 1) * xb[i][10 + j];
            }
        }
        float[] wa = fit(xa, ya, r);
        float[] wb = fit(xb, yb, r);
        List<Tensor> base = List.of(Tensor.of(new float[d]));
        List<Tensor> a = List.of(Tensor.of(wa));
        List<Tensor> b = List.of(Tensor.of(wb));

        float[] soup = ModelMerging.uniformSoup(List.of(a, b)).get(0).toArray();
        float[] ties = ModelMerging.ties(base, List.of(a, b), 0.25, 1).get(0).toArray();
        List<Tensor> va = ModelMerging.taskVector(base, a);
        List<Tensor> vb = ModelMerging.taskVector(base, b);
        float[] dare = ModelMerging.add(base, List.of(ModelMerging.dare(va, 0.5, 3), ModelMerging.dare(vb, 0.5, 4)), 1)
                .get(0).toArray();
        float[] slerp = ModelMerging.add(base, List.of(ModelMerging.slerp(va, vb, 0.5)), 1).get(0).toArray();
        float[] fisher = ModelMerging.fisher(List.of(a, b), List.of(fisher(xa), fisher(xb)), 1e-8).get(0).toArray();
        Tensor regMean = ModelMerging.regMean(
                List.of(Tensor.of(Shape.of(d, 1), wa), Tensor.of(Shape.of(d, 1), wb)),
                List.of(ModelMerging.gram(matrix(xa)), ModelMerging.gram(matrix(xb))), 0.9);
        float[] reg = regMean.toArray();

        double soupLoss = loss(soup, xa, ya) + loss(soup, xb, yb);
        double tiesLoss = loss(ties, xa, ya) + loss(ties, xb, yb);
        double dareLoss = loss(dare, xa, ya) + loss(dare, xb, yb);
        double slerpLoss = loss(slerp, xa, ya) + loss(slerp, xb, yb);
        double fisherLoss = loss(fisher, xa, ya) + loss(fisher, xb, yb);
        double regLoss = loss(reg, xa, ya) + loss(reg, xb, yb);
        System.out.printf("merge loss on both skills (lower is better): RegMean %.3f, Fisher %.3f, SLERP %.3f"
                + " vs uniform soup %.3f, TIES %.3f, DARE %.3f%n", regLoss, fisherLoss, slerpLoss, soupLoss, tiesLoss,
                dareLoss);
        assertTrue(regLoss < 0.3 * soupLoss, "RegMean " + regLoss + " vs soup " + soupLoss);
        assertTrue(fisherLoss < 0.5 * soupLoss, "Fisher " + fisherLoss + " vs soup " + soupLoss);
        assertTrue(slerpLoss < soupLoss, "SLERP " + slerpLoss + " vs soup " + soupLoss);
    }
}
