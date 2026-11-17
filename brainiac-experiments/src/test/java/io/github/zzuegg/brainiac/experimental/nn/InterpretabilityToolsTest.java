package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class InterpretabilityToolsTest {

    /** Three Gaussian clusters; cluster c is shifted along coordinate {@code c % dim}. */
    static float[][] clusters(int n, int dim, int[] y, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][dim];
        for (int i = 0; i < n; i++) {
            y[i] = i % 3;
            for (int d = 0; d < dim; d++) {
                x[i][d] = (float) r.nextGaussian() * 0.4f;
            }
            x[i][y[i] % dim] += 2;
        }
        return x;
    }

    static Model train(Model m, float[][] x, int[] y, int epochs) {
        float[][] t = new float[x.length][3];
        for (int i = 0; i < x.length; i++) {
            t[i][y[i]] = 1;
        }
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(epochs)
                .fit(Dataset.of(x, t));
        return m;
    }

    @Test
    void tunedLensReadsEarlyLayersBetterThanTheLogitLens() {
        int[] y = new int[150];
        float[][] x = clusters(150, 4, y, 1);
        Model m = train(Model.sequential().input(4).dense(12, Activation.TANH).dense(12, Activation.TANH)
                .dense(12, Activation.TANH).dense(3, Activation.SOFTMAX).seed(2).build(), x, y, 40);
        LogitLens lens = LogitLens.of(m, 3);
        Tensor in = Tensor.of(x);
        // the last state decoded by the readout is the model's own output
        assertArrayEquals(m.predict(in).toArray(), lens.decode(in).get(2).toArray(), 1e-5f);
        double[] plain = lens.divergences(in, false);
        assertEquals(0, plain[2], 1e-4);
        double[] tuned = lens.tune(in, 200, 0.02f);
        assertTrue(tuned[0] < plain[0] * 0.5, "tuned " + tuned[0] + " vs plain " + plain[0]);
        assertEquals(3, lens.targetProbability(in, y).length);
        assertEquals(3, lens.decodeTuned(in).size());
    }

    @Test
    void probeIsSelectiveForALinearPropertyAndFailsOnXor() {
        SplittableRandom r = new SplittableRandom(4);
        int types = 40;
        float[][] emb = new float[types][3];
        int[] prop = new int[types];
        for (int t = 0; t < types; t++) {
            for (int d = 0; d < 3; d++) {
                emb[t][d] = (float) r.nextGaussian();
            }
            prop[t] = emb[t][0] > 0 ? 1 : 0;
        }
        int n = 400;
        float[][] x = new float[n][3];
        int[] y = new int[n];
        int[] type = new int[n];
        int[] xor = new int[n];
        for (int i = 0; i < n; i++) {
            type[i] = r.nextInt(types);
            for (int d = 0; d < 3; d++) {
                x[i][d] = emb[type[i]][d] + (float) r.nextGaussian() * 0.05f;
            }
            y[i] = prop[type[i]];
            xor[i] = (x[i][0] > 0) ^ (x[i][1] > 0) ? 1 : 0;
        }
        float[][] tr = Arrays.copyOfRange(x, 0, 300);
        float[][] te = Arrays.copyOfRange(x, 300, n);
        LinearProbe lin = LinearProbe.fit(tr, Arrays.copyOf(y, 300), 2, 300, 0.05, 1e-3);
        assertTrue(lin.accuracy(te, Arrays.copyOfRange(y, 300, n)) > 0.9);
        LinearProbe x2 = LinearProbe.fit(tr, Arrays.copyOf(xor, 300), 2, 300, 0.05, 1e-3);
        assertTrue(x2.accuracy(te, Arrays.copyOfRange(xor, 300, n)) < 0.75);
        double[] s = LinearProbe.selectivity(tr, Arrays.copyOf(y, 300), Arrays.copyOf(type, 300), te,
                Arrays.copyOfRange(y, 300, n), Arrays.copyOfRange(type, 300, n), 2, 9);
        assertTrue(s[2] > 0.2, "task " + s[0] + " control " + s[1]);
    }

    @Test
    void integratedGradientsAreCompleteAndExactForLinearModels() {
        int[] y = new int[150];
        float[][] x = clusters(150, 4, y, 3);
        Model m = train(Model.sequential().input(4).dense(10, Activation.TANH).dense(3, Activation.SOFTMAX).seed(5)
                .build(), x, y, 30);
        float[] input = x[1];
        float[] base = new float[4];
        float[] ig = Attribution.integratedGradients(m, input, base, 1, 64);
        double sum = 0;
        for (float v : ig) {
            sum += v;
        }
        double delta = m.predict(input)[1] - m.predict(base)[1];
        assertEquals(delta, sum, 0.01 + 0.02 * Math.abs(delta));
        // the cluster's own coordinate carries the most attribution
        int top = 0;
        for (int i = 1; i < 4; i++) {
            if (Math.abs(ig[i]) > Math.abs(ig[top])) {
                top = i;
            }
        }
        assertEquals(1, top);

        Model lin = Model.sequential().input(3).dense(1).seed(7).build();
        float[] p = {1, -2, 0.5f};
        float[] w = Attribution.saliency(lin, new float[3], 0);
        float[] linIg = Attribution.integratedGradients(lin, p, new float[3], 0, 4);
        float[] gxi = Attribution.gradientTimesInput(lin, p, 0);
        float[] sg = Attribution.smoothGrad(lin, p, 0, 8, 0.5f, 1);
        for (int i = 0; i < 3; i++) {
            assertEquals(w[i] * p[i], linIg[i], 1e-5);
            assertEquals(w[i] * p[i], gxi[i], 1e-5);
            assertEquals(w[i], sg[i], 1e-5);
        }
    }

    @Test
    void laplaceIsLessConfidentFarFromTheData() {
        int[] y = new int[300];
        float[][] x = clusters(300, 2, y, 6);
        for (int i = 0; i < x.length; i++) {
            if (y[i] == 2) {   // give the third class its own corner
                x[i][0] -= 2;
                x[i][1] -= 2;
            }
        }
        Model m = train(Model.sequential().input(2).dense(16, Activation.RELU).dense(16, Activation.RELU)
                .dense(3, Activation.SOFTMAX).seed(8).build(), x, y, 40);
        LastLayerLaplace la = LastLayerLaplace.fit(m, Tensor.of(x), 1);
        assertTrue(la.fitPriorPrecision(Tensor.of(x), y) > 0);
        double[][] mapIn = la.predictMap(Tensor.of(x));
        double[][] lapIn = la.predict(Tensor.of(x));
        int okMap = 0;
        int okLap = 0;
        for (int i = 0; i < x.length; i++) {
            okMap += argmax(mapIn[i]) == y[i] ? 1 : 0;
            okLap += argmax(lapIn[i]) == y[i] ? 1 : 0;
        }
        assertTrue(okLap >= okMap - 3, okLap + " vs " + okMap);
        float[][] far = new float[50][];
        SplittableRandom r = new SplittableRandom(11);
        for (int i = 0; i < far.length; i++) {
            double a = r.nextDouble() * 2 * Math.PI;
            far[i] = new float[] {(float) (40 * Math.cos(a)), (float) (40 * Math.sin(a))};
        }
        la.priorPrecision(1);
        double[][] mapFar = la.predictMap(Tensor.of(far));
        double[][] lapFar = la.predict(Tensor.of(far));
        double cm = 0;
        double cl = 0;
        for (int i = 0; i < far.length; i++) {
            cm += max(mapFar[i]) / far.length;
            cl += max(lapFar[i]) / far.length;
        }
        assertTrue(cl < cm - 0.1, "laplace " + cl + " map " + cm);
    }

    static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }

    static double max(double[] p) {
        return p[argmax(p)];
    }
}
