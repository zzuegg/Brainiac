package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

/** Recurrent layers, attention and transformer blocks. */
class SequenceTest {

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    private static void gradCheck(String name, Function<List<Var>, Var> f, Tensor... inputs) {
        Tensor out = f.apply(java.util.Arrays.stream(inputs).map(Var::of).toList()).value();
        Tensor weights = Tensor.uniform(out.shape(), 0.5f, 1.5f, new SplittableRandom(99));
        GradCheck.Result r = GradCheck.check(v -> f.apply(v).mul(weights).sum(), List.of(inputs));
        assertTrue(r.passed(), name + ": " + r);
    }

    private static Var recurrent(Recurrent.Cell cell, boolean resetAfter, ActivationOp act, List<Var> v) {
        RecurrentOp op = new RecurrentOp(cell, act, ActivationOp.SIGMOID, resetAfter);
        return Var.apply(op, v.toArray(Var[]::new));
    }

    @Test
    void backpropagationThroughTimeMatchesFiniteDifferences() {
        int n = 2;
        int t = 4;
        int h = 3;
        gradCheck("rnn", v -> recurrent(Recurrent.Cell.RNN, false, ActivationOp.TANH, v),
                random(1, n, t, h), random(2, h, h), random(3, n, h));
        gradCheck("lstm", v -> recurrent(Recurrent.Cell.LSTM, false, ActivationOp.TANH, v),
                random(4, n, t, 4 * h), random(5, h, 4 * h), random(6, n, h), random(7, n, h));
        gradCheck("gru reset after", v -> recurrent(Recurrent.Cell.GRU, true, ActivationOp.TANH, v),
                random(8, n, t, 3 * h), random(9, h, 3 * h), random(10, n, h), random(11, h));
        gradCheck("gru reset before", v -> recurrent(Recurrent.Cell.GRU, false, ActivationOp.TANH, v),
                random(12, n, t, 3 * h), random(13, h, 3 * h), random(14, n, h));
        gradCheck("lstm softsign", v -> recurrent(Recurrent.Cell.LSTM, false, ActivationOp.SOFTSIGN, v),
                random(15, n, t, 4 * h), random(16, h, 4 * h), random(17, n, h), random(18, n, h));
    }

    @Test
    void lstmMatchesTheTextbookEquations() {
        int h = 2;
        Tensor xp = random(20, 1, 3, 4 * h);
        Tensor wh = random(21, h, 4 * h);
        Tensor out = recurrent(Recurrent.Cell.LSTM, false, ActivationOp.TANH,
                List.of(Var.of(xp), Var.of(wh), Var.of(Tensor.zeros(Shape.of(1, h))),
                        Var.of(Tensor.zeros(Shape.of(1, h))))).value();
        float[] x = xp.toArray();
        float[] w = wh.toArray();
        double[] hs = new double[h];
        double[] cs = new double[h];
        float[] expected = new float[3 * h];
        for (int t = 0; t < 3; t++) {
            double[] z = new double[4 * h];
            for (int j = 0; j < 4 * h; j++) {
                z[j] = x[t * 4 * h + j];
                for (int k = 0; k < h; k++) {
                    z[j] += hs[k] * w[k * 4 * h + j];
                }
            }
            for (int k = 0; k < h; k++) {
                double i = 1 / (1 + Math.exp(-z[k]));
                double f = 1 / (1 + Math.exp(-z[h + k]));
                double g = Math.tanh(z[2 * h + k]);
                double o = 1 / (1 + Math.exp(-z[3 * h + k]));
                cs[k] = f * cs[k] + i * g;
                hs[k] = o * Math.tanh(cs[k]);
                expected[t * h + k] = (float) hs[k];
            }
        }
        assertArrayEquals(expected, out.toArray(), 1e-5f);
    }

    @Test
    void recurrentLayerOptionsAndShapes() {
        SplittableRandom random = new SplittableRandom(1);
        Shape seq = Shape.of(5, 3);
        Tensor x = random(30, 2, 5, 3);
        for (Recurrent.Spec spec : List.of(
                Recurrent.lstm(4),
                Recurrent.lstm(4).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT),
                Recurrent.gru(4).resetAfter(false).reverse(true).returnSequences(true),
                Recurrent.gru(4).bidirectional(Recurrent.Merge.AVERAGE).learnInitialState(true),
                Recurrent.rnn(4).activation(ActivationOp.RELU).bias(false),
                Recurrent.lstm(4).recurrentActivation(ActivationOp.HARD_SIGMOID).learnInitialState(true)
                        .bidirectional(Recurrent.Merge.MULTIPLY).returnSequences(true))) {
            Layer layer = spec.create(seq, random);
            Var y = layer.forward(Var.of(x));
            int[] dims = y.shape().toArray();
            assertEquals(spec.outputShape(seq), Shape.of(java.util.Arrays.copyOfRange(dims, 1, dims.length)));
            y.sum().backward();
            for (Var p : layer.parameters()) {
                assertTrue(p.grad() != null, layer + " parameter " + p.shape() + " has no gradient");
            }
        }
    }

    @Test
    void reversedLayerReadsTheSequenceBackwards() {
        Layer fwd = Recurrent.gru(3).returnSequences(true).create(Shape.of(4, 2), new SplittableRandom(5));
        Layer bwd = Recurrent.gru(3).returnSequences(true).reverse(true).create(Shape.of(4, 2),
                new SplittableRandom(5));
        Tensor x = random(31, 1, 4, 2);
        float[] flipped = new float[8];
        float[] xs = x.toArray();
        for (int t = 0; t < 4; t++) {
            System.arraycopy(xs, (3 - t) * 2, flipped, t * 2, 2);
        }
        float[] a = fwd.forward(Var.of(Tensor.of(Shape.of(1, 4, 2), flipped))).value().toArray();
        float[] b = bwd.forward(Var.of(x)).value().toArray();
        for (int t = 0; t < 4; t++) {
            for (int k = 0; k < 3; k++) {
                assertEquals(a[t * 3 + k], b[(3 - t) * 3 + k], 1e-6f);
            }
        }
    }

    @Test
    void attentionGradients() {
        Tensor bias = Functional.causalMask(3, 4, io.github.zzuegg.brainiac.tensor.Device.cpu());
        gradCheck("attention gqa causal", v -> Functional.attention(v.get(0), v.get(1), v.get(2), bias, 0.5f, 0, null),
                random(40, 2, 4, 3, 2), random(41, 2, 2, 4, 2), random(42, 2, 2, 4, 3));
        gradCheck("rotary", v -> Functional.rotary(v.get(0), 10000, 1), random(43, 2, 3, 4));
        gradCheck("attention layer", v -> {
            MultiHeadAttention a = new MultiHeadAttention(new MultiHeadAttention.Settings(4, 4, 2, 1, 2, 2, 4, true,
                    true, 100, true, 0, 0, false), Init.XAVIER, new SplittableRandom(3));
            return a.forward(List.of(v.get(0)), false);
        }, random(44, 2, 3, 4));
    }

    @Test
    void customAttentionBiasMasksBlocks() {
        TransformerBlock causal = (TransformerBlock) TransformerBlock.of(2).causal(true)
                .create(Shape.of(6, 4), new SplittableRandom(8));
        TransformerBlock open = (TransformerBlock) TransformerBlock.of(2)
                .create(Shape.of(6, 4), new SplittableRandom(8));
        Tensor x = random(51, 2, 6, 4);
        // a causal mask given as a bias equals the built-in one
        assertArrayEquals(causal.forward(Var.of(x)).value().toArray(), open.forward(List.of(Var.of(x)),
                Functional.causalMask(6, 6, x.device()), false).value().toArray(), 1e-5f);
        // block-causal: steps see their own block of 3 and earlier blocks
        float[] m = new float[36];
        for (int q = 0; q < 6; q++) {
            for (int k = 0; k < 6; k++) {
                m[q * 6 + k] = k / 3 <= q / 3 ? 0 : -1e9f;
            }
        }
        Tensor blocks = Tensor.of(Shape.of(6, 6), m);
        float[] changed = x.toArray();
        changed[4 * 4] += 3;   // step 4, in the second block
        float[] a = open.forward(List.of(Var.of(x)), blocks, false).value().toArray();
        float[] b = open.forward(List.of(Var.of(Tensor.of(Shape.of(2, 6, 4), changed))), blocks, false)
                .value().toArray();
        for (int i = 0; i < 12; i++) {
            assertEquals(a[i], b[i], 1e-5f, "the first block saw step 4");
        }
        assertTrue(Math.abs(a[12] - b[12]) > 1e-4, "step 3 shares a block with step 4");
    }

    @Test
    void causalAttentionIgnoresTheFuture() {
        Layer block = TransformerBlock.of(2).causal(true).rotary(10000).gated(true).rmsNorm(true)
                .create(Shape.of(5, 4), new SplittableRandom(7));
        Tensor x = random(50, 1, 5, 4);
        float[] changed = x.toArray();
        changed[4 * 4] += 3;   // only the last step changes
        float[] a = block.forward(Var.of(x)).value().toArray();
        float[] b = block.forward(Var.of(Tensor.of(Shape.of(1, 5, 4), changed))).value().toArray();
        for (int i = 0; i < 16; i++) {
            assertEquals(a[i], b[i], 1e-5f, "step " + i / 4 + " changed");
        }
        boolean lastChanged = false;
        for (int i = 16; i < 20; i++) {
            lastChanged |= Math.abs(a[i] - b[i]) > 1e-4;
        }
        assertTrue(lastChanged);
    }

    @Test
    void keyMaskHidesPaddedSteps() {
        MultiHeadAttention a = (MultiHeadAttention) MultiHeadAttention.of(2).keyMask(true)
                .create(List.of(Shape.of(3, 4), Shape.of(3)), new SplittableRandom(1));
        Tensor x = random(60, 1, 3, 4);
        Tensor mask = Tensor.of(Shape.of(1, 3), 1, 1, 0);
        float[] y1 = a.forward(List.of(Var.of(x), Var.of(mask)), false).value().toArray();
        float[] padded = x.toArray();
        for (int i = 8; i < 12; i++) {
            padded[i] = 100;
        }
        float[] y2 = a.forward(List.of(Var.of(Tensor.of(Shape.of(1, 3, 4), padded)), Var.of(mask)), false)
                .value().toArray();
        // queries 0 and 1 do not see the padded key
        for (int i = 0; i < 8; i++) {
            assertEquals(y1[i], y2[i], 1e-4f);
        }
    }

    @Test
    void encoderDecoderGraphSavesAndLoads(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws Exception {
        Node src = Node.input(6, 5);
        Node encoded = src.dense(8).positionalEncoding().transformer(2, false)
                .add(TransformerBlock.of(4).kvHeads(2).preNorm(false).activation(Activation.RELU).dropout(0.1f));
        Node decoded = encoded.add(Recurrent.lstm(8).returnSequences(true))
                .merge(TransformerBlock.of(2).crossAttention(true).alibi(true), encoded)
                .merge(MultiHeadAttention.of(2).headSize(3).valueSize(5).outputSize(6), encoded)
                .add(Recurrent.gru(4).bidirectional(Recurrent.Merge.SUM).returnSequences(true))
                .attention(2, true)
                .layer(PositionalEncoding.learned(6, 4, new SplittableRandom(1)))
                .sequencePool(SequencePool.Kind.MAX)
                .dense(3, Activation.SOFTMAX);
        Model model = Model.graph(src, decoded).seed(4).build();
        Tensor x = random(70, 2, 30);
        Var out = model.forward(Var.of(x), true);
        assertEquals(Shape.of(2, 3), out.shape());
        out.sum().backward();
        java.nio.file.Path file = dir.resolve("seq.bnm");
        model.save(file);
        Model loaded = Model.load(file);
        assertEquals(model.summary(), loaded.summary());
        assertArrayEquals(model.predict(x).toArray(), loaded.predict(x).toArray(), 1e-5f);
    }

    @Test
    void stackedRecurrentModelLearnsToSumASequence() {
        // target: the running sum's sign at every step
        SplittableRandom random = new SplittableRandom(3);
        int n = 256;
        int t = 8;
        float[][] xs = new float[n][t];
        float[][] ys = new float[n][t];
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int s = 0; s < t; s++) {
                xs[i][s] = (float) random.nextDouble(-1, 1);
                sum += xs[i][s];
                ys[i][s] = sum > 0 ? 1 : 0;
            }
        }
        Model model = Model.sequential().input(t, 1)
                .gru(16, true)
                .lstm(16, true)
                .dense(1, Activation.SIGMOID)
                .seed(2)
                .build();
        assertEquals(Shape.of(t, 1), model.outputShape());
        Tensor x = Tensor.of(xs).reshape(n, t, 1);
        Tensor y = Tensor.of(ys).reshape(n, t, 1);
        float first = 0;
        float last = 0;
        java.util.List<Var> params = model.parameters();
        float lr = 0.01f;
        // a few hand-rolled Adam-free SGD steps keep this test independent of brainiac-train
        for (int step = 0; step < 150; step++) {
            for (Var p : params) {
                p.zeroGrad();
            }
            Var loss = Loss.BINARY_CROSS_ENTROPY.compute(model.forward(Var.of(x), true), y);
            loss.backward();
            for (Var p : params) {
                p.assign(p.value().sub(p.grad().mul(lr * 20)));
            }
            if (step == 0) {
                first = loss.value().item();
            }
            last = loss.value().item();
        }
        assertTrue(last < first * 0.6f, "loss went from " + first + " to " + last);
    }
}
