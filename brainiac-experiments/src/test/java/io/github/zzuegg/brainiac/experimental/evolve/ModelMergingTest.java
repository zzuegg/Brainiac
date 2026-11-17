package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ModelMergingTest {

    @Test
    void tiesTrimsElectsAndAveragesDisjointly() {
        double[] base = {0, 0, 0, 0};
        double[] a = {1.0, -0.5, 0.01, 0};
        double[] b = {1.0, 0.3, 0, 2};
        // keep half of each task vector: a -> {1, -0.5, 0, 0}, b -> {1, 0, 0, 2}
        double[] m = ModelMerging.ties(base, List.of(a, b), 0.5, 1);
        assertArrayEquals(new double[] {1.0, -0.5, 0, 2}, m, 1e-12);
        assertArrayEquals(new double[] {1, -0.1, 0.005, 1}, ModelMerging.average(List.of(a, b)), 1e-12);
    }

    private static double loss(double[] w, double[][] x, double[] y) {
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

    /** Ridge-free least squares by gradient descent from zero. */
    private static double[] fit(double[][] x, double[] y, SplittableRandom r) {
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
                w[j] -= 0.05 * g[j] + 0.002 * r.nextGaussian();   // noisy fine-tuning
            }
        }
        return w;
    }

    @Test
    void tiesKeepsBothSkillsWhereAveragingHalvesThem() {
        SplittableRandom r = new SplittableRandom(1);
        int d = 20;
        // two input domains: task A sees features 0-9, task B features 10-19
        double[][] xa = new double[200][d];
        double[][] xb = new double[200][d];
        double[] ya = new double[200];
        double[] yb = new double[200];
        for (int i = 0; i < 200; i++) {
            for (int j = 0; j < 10; j++) {
                xa[i][j] = r.nextGaussian();
                xb[i][10 + j] = r.nextGaussian();
            }
            for (int j = 0; j < 5; j++) {
                ya[i] += (j + 1) * xa[i][j];
                yb[i] += (j + 1) * xb[i][10 + j];
            }
        }
        double[] wa = fit(xa, ya, r);
        double[] wb = fit(xb, yb, r);
        double[] base = new double[d];
        double[] avg = ModelMerging.average(List.of(wa, wb));
        double[] ties = ModelMerging.ties(base, List.of(wa, wb), 0.25, 1);
        double avgLoss = loss(avg, xa, ya) + loss(avg, xb, yb);
        double tiesLoss = loss(ties, xa, ya) + loss(ties, xb, yb);
        assertTrue(tiesLoss < 0.3 * avgLoss, "ties " + tiesLoss + " vs average " + avgLoss);
    }

    @Test
    void mergesBrainiacModels() {
        Model base = Model.sequential().input(3).dense(4, Activation.TANH).dense(2).seed(1).build();
        Model a = base.copy(base.device());
        Model b = base.copy(base.device());
        double[] pa = ParameterVector.flatten(a.parameters());
        pa[0] += 1;
        ParameterVector.assign(a.parameters(), pa);
        double[] pb = ParameterVector.flatten(b.parameters());
        pb[5] -= 2;
        ParameterVector.assign(b.parameters(), pb);
        Model merged = ModelMerging.ties(base, List.of(a, b), 0.1, 1);
        double[] m = ParameterVector.flatten(merged.parameters());
        double[] p0 = ParameterVector.flatten(base.parameters());
        assertTrue(Math.abs(m[0] - p0[0] - 1) < 1e-6 && Math.abs(m[5] - p0[5] + 2) < 1e-6);
    }
}
