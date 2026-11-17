package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Hebbian plasticity and fast weight programmers: short-term memory without a KV cache. */
class FastWeightsTest {

    private static final int K = 4;

    /** Trains {@code layer + head} on recall-after-facts episodes; returns test accuracy. */
    static double recallAccuracy(Layer layer, int width, int steps, float lr) {
        SplittableRandom random = new SplittableRandom(5);
        Dense head = new Dense(width, K, Activation.IDENTITY, Init.XAVIER, random.split());
        List<Var> params = new ArrayList<>(layer.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom data = new SplittableRandom(3);
        int batch = 8;
        int test = 10;
        double correct = 0;
        for (int step = 0; step < steps + test; step++) {
            float[] x = new float[batch * 4 * K * K];
            float[] y = new float[batch * K];
            for (int b = 0; b < batch; b++) {
                int[] perm = InfiniAttentionTest.shuffled(data, K);
                int[] order = InfiniAttentionTest.shuffled(data, K);
                System.arraycopy(InfiniAttentionTest.episode(perm, order, K), 0, x, b * 4 * K * K, 4 * K * K);
                for (int i = 0; i < K; i++) {
                    y[b * K + i] = perm[order[i]];
                }
            }
            Var h = layer.forward(Var.of(Tensor.of(Shape.of(batch, 2 * K, 2 * K), x))).slice(1, K, 2 * K);
            Var out = head.forward(h.reshape(batch * K, width));
            if (step < steps) {
                Functional.crossEntropy(out, Tensor.of(y)).backward();
                opt.step(params);
            } else {
                float[] p = out.value().argmax(1).toArray();
                for (int i = 0; i < p.length; i++) {
                    correct += p[i] == y[i] ? 1 : 0;
                }
            }
        }
        return correct / (test * batch * K);
    }

    @Test
    void plasticDenseBindsKeysToValuesWithinASequence() {
        double plastic = recallAccuracy(new PlasticDense(2 * K, 16, Activation.TANH, new SplittableRandom(1)), 16, 300,
                0.03f);
        double frozen = recallAccuracy(new PlasticDense(2 * K, 16, Activation.TANH, new SplittableRandom(1))
                .plastic(false), 16, 300, 0.03f);
        assertTrue(plastic > 0.85, "plastic " + plastic);
        assertTrue(frozen < 0.45, "frozen " + frozen);
    }

    @Test
    void plasticRecurrentRemembersPatternsAPlainRnnCannot() {
        double plastic = completion(true);
        double plain = completion(false);
        assertTrue(plastic < 0.6 * plain, "plastic error " + plastic + " vs plain " + plain);
    }

    /** Miconi's pattern completion: two +-1 patterns shown, then half of one; error on the hidden half. */
    private static double completion(boolean plastic) {
        int d = 8;
        PlasticRecurrent rnn = new PlasticRecurrent(d, d, PlasticRecurrent.Rule.HEBB, new SplittableRandom(2))
                .plastic(plastic);
        Optimizer opt = Optimizer.adam(0.03f);
        SplittableRandom data = new SplittableRandom(4);
        int batch = 8;
        int steps = 6;
        double error = 0;
        int train = 2500;
        for (int it = 0; it < train + 10; it++) {
            float[] x = new float[batch * steps * d];
            float[] target = new float[batch * d];
            float[] hidden = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                float[][] pats = new float[2][d];
                for (float[] p : pats) {
                    for (int i = 0; i < d; i++) {
                        p[i] = data.nextBoolean() ? 1 : -1;
                    }
                }
                for (int s = 0; s < 4; s++) {
                    System.arraycopy(pats[s / 2], 0, x, (b * steps + s) * d, d);
                }
                int pick = data.nextInt(2);
                for (int i = 0; i < d; i++) {
                    if (i < d / 2) {
                        x[(b * steps + 4) * d + i] = pats[pick][i];
                    } else {
                        hidden[b * d + i] = 1;
                    }
                    target[b * d + i] = pats[pick][i];
                }
            }
            Var out = rnn.forward(Var.of(Tensor.of(Shape.of(batch, steps, d), x))).slice(1, steps - 1, steps)
                    .reshape(batch, d);
            Var loss = out.sub(Tensor.of(Shape.of(batch, d), target)).square().mul(Tensor.of(Shape.of(batch, d), hidden))
                    .sum().mul(1f / (batch * d / 2f));
            if (it < train) {
                loss.backward();
                opt.step(rnn.parameters());
            } else {
                error += loss.value().item() / 10;
            }
        }
        return error;
    }

    @Test
    void sumRuleEqualsCausalLinearAttention() {
        FastWeightProgrammer fwp = new FastWeightProgrammer(6, 2, 3, FastWeightProgrammer.Rule.SUM,
                new SplittableRandom(1)).normalizeKeys(false);
        Tensor x = Tensor.normal(Shape.of(2, 5, 6), 0, 1, new SplittableRandom(2));
        float[] got = fwp.forward(Var.of(x)).value().toArray();
        List<Var> p = fwp.parameters();
        Var xv = Var.of(x);
        Var k = InfiniAttention.eluPlusOne(xv.matmul(p.get(0)).reshape(2, 5, 2, 3).permute(0, 2, 1, 3));
        Var v = xv.matmul(p.get(1)).reshape(2, 5, 2, 3).permute(0, 2, 1, 3);
        Var q = InfiniAttention.eluPlusOne(xv.matmul(p.get(2)).reshape(2, 5, 2, 3).permute(0, 2, 1, 3));
        float[] lower = new float[25];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j <= i; j++) {
                lower[i * 5 + j] = 1;
            }
        }
        Var att = q.matmul(k.transpose()).mul(Tensor.of(Shape.of(5, 5), lower)).matmul(v);
        Var expected = att.permute(0, 2, 1, 3).reshape(2, 5, 6).matmul(p.get(4));
        assertArrayEquals(expected.value().toArray(), got, 1e-4f);
    }

    @Test
    void fastWeightProgrammersLearnInContextRecall() {
        double delta = recallAccuracy(new FastWeightProgrammer(2 * K, 1, 2 * K, FastWeightProgrammer.Rule.DELTA,
                new SplittableRandom(1)), 2 * K, 700, 0.02f);
        assertTrue(delta > 0.85, "delta rule " + delta);
    }

    @Test
    void brainProjectionBecomesPlastic() {
        Brain brain = Brain.builder().region("r", 6).initialNeurons(6).activation(Activation.TANH)
                .connect("input", "r", 0.5).output("r").build().create(Shape.of(4), new SplittableRandom(3));
        PlasticDense p = PlasticDense.fromBrain(brain, 0, Activation.TANH, new SplittableRandom(4)).plastic(false);
        Tensor x = Tensor.normal(Shape.of(3, 4), 0, 1, new SplittableRandom(5));
        float[] expected = brain.forward(Var.of(x)).value().toArray();
        float[] got = p.forward(Var.of(x.reshape(3, 1, 4))).value().toArray();
        assertArrayEquals(expected, got, 1e-5f);
        // plastic: the second presentation of an input differs from the first
        p.plastic(true);
        Tensor twice = Tensor.concat(1, x.reshape(3, 1, 4), x.reshape(3, 1, 4));
        p.parameters().get(1).assign(Tensor.full(Shape.of(4, 6), 1f));
        p.parameters().get(2).assign(Tensor.full(Shape.of(1), 2f));
        float[] y = p.forward(Var.of(twice)).value().toArray();
        double diff = 0;
        for (int i = 0; i < 6; i++) {
            diff += Math.abs(y[i] - y[6 + i]);
        }
        assertTrue(diff > 1e-2, "trace changed nothing");
    }
}
