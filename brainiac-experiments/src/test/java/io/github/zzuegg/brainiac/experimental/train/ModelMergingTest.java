package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ModelMergingTest {

    @Test
    void tiesTrimsElectsAndAveragesDisjointly() {
        List<Tensor> base = List.of(Tensor.of(0, 0, 0, 0));
        List<Tensor> a = List.of(Tensor.of(1.0f, -0.5f, 0.01f, 0));
        List<Tensor> b = List.of(Tensor.of(1.0f, 0.3f, 0, 2));
        // keep half of each task vector: a -> {1, -0.5, 0, 0}, b -> {1, 0, 0, 2}
        List<Tensor> m = ModelMerging.ties(base, List.of(a, b), 0.5, 1);
        assertArrayEquals(new float[] {1.0f, -0.5f, 0, 2}, m.get(0).toArray(), 1e-6f);
        assertArrayEquals(m.get(0).toArray(), ModelMerging.crossover(base, a, b, 0.5).get(0).toArray());
        assertArrayEquals(new float[] {1, -0.1f, 0.005f, 1}, ModelMerging.uniformSoup(List.of(a, b)).get(0).toArray(),
                1e-6f);
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

    /** Ridge-free least squares by gradient descent from zero. */
    private static List<Tensor> fit(double[][] x, double[] y, SplittableRandom r) {
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
        float[] f = new float[w.length];
        for (int j = 0; j < w.length; j++) {
            f[j] = (float) w[j];
        }
        return List.of(Tensor.of(f));
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
        List<Tensor> wa = fit(xa, ya, r);
        List<Tensor> wb = fit(xb, yb, r);
        List<Tensor> base = List.of(Tensor.of(new float[d]));
        float[] avg = ModelMerging.uniformSoup(List.of(wa, wb)).get(0).toArray();
        float[] ties = ModelMerging.ties(base, List.of(wa, wb), 0.25, 1).get(0).toArray();
        double avgLoss = loss(avg, xa, ya) + loss(avg, xb, yb);
        double tiesLoss = loss(ties, xa, ya) + loss(ties, xb, yb);
        assertTrue(tiesLoss < 0.3 * avgLoss, "ties " + tiesLoss + " vs average " + avgLoss);
    }

    private static void nudge(Model model, int parameter, int index, float delta) {
        Var p = model.parameters().get(parameter);
        float[] v = p.value().toArray();
        v[index] += delta;
        p.assign(Tensor.of(p.shape(), v));
    }

    @Test
    void mergesBrainiacModels() {
        Model base = Model.sequential().input(3).dense(4, Activation.TANH).dense(2).seed(1).build();
        Model a = base.copy(base.device());
        Model b = base.copy(base.device());
        nudge(a, 0, 0, 1);
        nudge(b, 0, 5, -2);
        Model merged = ModelMerging.ties(base, List.of(a, b), 0.1, 1);
        float[] m = merged.parameters().get(0).value().toArray();
        float[] p0 = base.parameters().get(0).value().toArray();
        assertTrue(Math.abs(m[0] - p0[0] - 1) < 1e-6 && Math.abs(m[5] - p0[5] + 2) < 1e-6);
        // the base is untouched
        assertArrayEquals(p0, base.parameters().get(0).value().toArray());
    }
}
