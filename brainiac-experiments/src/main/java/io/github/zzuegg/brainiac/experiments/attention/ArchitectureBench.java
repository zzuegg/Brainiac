package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The shared harness of the architecture-refinement experiments: a toy language for
 * validation loss, retrieval and exact-lag tasks at several lengths for length
 * generalization, and a tiny model around any body {@link LayerSpec} (token embedding, the
 * body, RMS norm, readout), trained with Adam on equal budgets.
 *
 * <p>The toy language has two parts per sequence: a third drawn from a noisy Markov chain
 * (each token has two likely successors, otherwise a uniform draw), which a model can learn
 * only partly from local statistics, then a periodic continuation that keeps repeating an
 * earlier span (period 2 to 7), which only in-context retrieval (an induction head) can
 * predict. The validation loss therefore mostly measures how fast attention learns to
 * retrieve; the model gets a previous-token embedding so a single layer can do it.
 */
public final class ArchitectureBench {

    /** The toy language's vocabulary. */
    public static final int VOCAB = 32;
    /** The toy language's sequence length. */
    public static final int LENGTH = 24;

    private static final int[][] SUCCESSORS = successors();

    private ArchitectureBench() {
    }

    /**
     * One trained language model.
     *
     * @param name the configuration's name
     * @param validationLoss the mean next-token cross-entropy on 128 fresh sequences
     * @param parameters the body's parameter count
     * @param millis the training time
     * @param model the trained model, for inspection
     * @param validation the validation tokens
     */
    public record LmResult(String name, double validationLoss, long parameters, long millis, TinyModel model,
            int[][] validation) {

        /** @return the trained body */
        public Layer body() {
            return model.body();
        }
    }

    /**
     * A tiny model: token embedding (plus an optional previous-token embedding, a bigram
     * feature that lets one attention layer do induction), body, RMS norm and readout.
     *
     * @param embed the token embedding {@code [vocab, width]}
     * @param previous the previous-token embedding {@code [vocab, width]}, or {@code null}
     * @param body the body
     * @param readout the readout {@code [width, vocab]}
     */
    public record TinyModel(Var embed, Var previous, Layer body, Var readout) {

        /**
         * The hidden states after the body (before the final norm).
         *
         * @param tokens {@code [batch][steps]}
         * @param training whether the body runs in training mode
         * @return {@code [batch, steps, width]}
         */
        public Var hidden(int[][] tokens, boolean training) {
            int b = tokens.length;
            int t = tokens[0].length;
            int width = embed.shape().dim(1);
            Var x = embed.gather(0, flat(tokens)).reshape(b, t, width);
            if (previous != null) {
                int[][] prev = new int[b][t];
                for (int i = 0; i < b; i++) {
                    for (int j = 0; j < t; j++) {
                        prev[i][j] = tokens[i][Math.max(0, j - 1)];
                    }
                }
                x = x.add(previous.gather(0, flat(prev)).reshape(b, t, width));
            }
            return body.forward(x, training);
        }

        /**
         * The logits at every step.
         *
         * @param tokens {@code [batch][steps]}
         * @param training whether the body runs in training mode
         * @return {@code [batch, steps, vocab]}
         */
        public Var logits(int[][] tokens, boolean training) {
            return readout(hidden(tokens, training));
        }

        /**
         * RMS-normalizes hidden states and reads out logits.
         *
         * @param h {@code [..., width]}
         * @return {@code [..., vocab]}
         */
        public Var readout(Var h) {
            return h.div(h.square().mean(-1, true).add(1e-6f).sqrt()).matmul(readout);
        }

        /** @return every parameter: embeddings, body and readout */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, readout));
            if (previous != null) {
                p.add(previous);
            }
            p.addAll(body.parameters());
            return p;
        }
    }

    /**
     * The core transformer block configured like {@link ResidualMixerBlock} around softmax
     * RoPE attention: pre-RMSNorm, SwiGLU with twice the width, no biases; the shared
     * baseline.
     *
     * @param heads the attention heads
     * @param width the model width
     * @return the spec
     */
    public static TransformerBlock.Spec coreBlock(int heads, int width) {
        return TransformerBlock.of(heads).causal(true).rmsNorm(true).gated(true).activation(Activation.SILU)
                .rotary(10000f).bias(false).ffnSize(2 * width);
    }

    /**
     * {@code depth} pre-norm blocks around one attention spec.
     *
     * @param attention the mixer of every block
     * @param depth the number of blocks
     * @return the spec
     */
    public static LayerSpec stack(LayerSpec attention, int depth) {
        return LayerChain.repeat(ResidualMixerBlock.of(attention), depth);
    }

    /**
     * Draws toy-language sequences.
     *
     * @param random the random source
     * @param batch the number of sequences
     * @return {@code [batch][LENGTH]} tokens
     */
    public static int[][] corpus(SplittableRandom random, int batch) {
        int markov = LENGTH / 3;
        int copy = LENGTH - markov;
        int[][] out = new int[batch][LENGTH];
        for (int b = 0; b < batch; b++) {
            out[b][0] = random.nextInt(VOCAB);
            for (int i = 1; i < markov; i++) {
                double u = random.nextDouble();
                int[] next = SUCCESSORS[out[b][i - 1]];
                out[b][i] = u < 0.3 ? next[0] : u < 0.4 ? next[1] : random.nextInt(VOCAB);
            }
            int start = random.nextInt(markov - 1);
            for (int i = 0; i < copy; i++) {
                out[b][markov + i] = out[b][start + i];
            }
        }
        return out;
    }

    /**
     * Trains a language model with the given body on the toy language and measures its
     * validation loss.
     *
     * @param name the configuration's name
     * @param body the body, built for {@code [LENGTH, width]}
     * @param width the model width
     * @param steps Adam steps on batches of 16
     * @param seed the seed for weights and data
     * @return the result
     */
    public static LmResult languageModel(String name, LayerSpec body, int width, int steps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        TinyModel m = model(body, VOCAB, LENGTH, width, true, random);
        List<Var> params = m.parameters();
        Optimizer opt = Optimizer.adam(1e-2f);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int[][] batch = corpus(random, 16);
            Var loss = nextTokenLoss(m, batch, true);
            Var aux = m.body().auxiliaryLoss();
            (aux == null ? loss : loss.add(aux)).backward();
            opt.step(params);
        }
        long millis = (System.nanoTime() - start) / 1_000_000;
        int[][] validation = corpus(new SplittableRandom(seed + 999), 128);
        double loss = nextTokenLoss(m, validation, false).value().item();
        return new LmResult(name, loss, count(m.body()), millis, m, validation);
    }

    /**
     * The mean next-token cross-entropy of a model on sequences.
     *
     * @param m the model
     * @param tokens {@code [batch][steps]}
     * @param training whether the body runs in training mode
     * @return the scalar loss
     */
    public static Var nextTokenLoss(TinyModel m, int[][] tokens, boolean training) {
        int b = tokens.length;
        int t = tokens[0].length;
        int vocab = m.readout().shape().dim(1);
        Var logits = m.logits(tokens, training).slice(1, 0, t - 1).reshape(b * (t - 1), vocab);
        float[] target = new float[b * (t - 1)];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t - 1; j++) {
                target[i * (t - 1) + j] = tokens[i][j + 1];
            }
        }
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(target.length), target));
    }

    /**
     * Trains on a {@link MixerShootout} retrieval task (answer read at the last step) and
     * tests on the same and other tasks, for example longer versions of it.
     *
     * @param body the body
     * @param width the model width
     * @param train the training task
     * @param tests the test tasks; same vocabulary
     * @param steps Adam steps on batches of 32
     * @param lr the learning rate
     * @param seed the seed
     * @param previousToken whether to add a previous-token embedding (which hands the model
     *     a bigram feature, so one attention layer suffices for recall)
     * @return the accuracy on 256 fresh samples of every test task
     */
    public static double[] retrieval(LayerSpec body, int width, MixerShootout.Task train,
            List<MixerShootout.Task> tests, int steps, float lr, long seed, boolean previousToken) {
        SplittableRandom random = new SplittableRandom(seed);
        int vocab = train.vocabulary();
        TinyModel m = model(body, vocab, train.length(), width, previousToken, random);
        List<Var> params = m.parameters();
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            int[][][] batch = train.batch(random, 32);
            Functional.crossEntropy(lastLogits(m, batch[0], true), ids(batch[1][0])).backward();
            opt.step(params);
        }
        double[] acc = new double[tests.size()];
        for (int k = 0; k < acc.length; k++) {
            int[][][] test = tests.get(k).batch(new SplittableRandom(seed + 1000 + k), 256);
            Tensor pred = lastLogits(m, test[0], false).value().argmax(1);
            acc[k] = pred.equalTo(ids(test[1][0])).mean().item();
        }
        return acc;
    }

    /**
     * Trains on an exact-lag copy task (at every step output the token {@code lag} steps
     * back; content alone cannot solve it) at one length and tests at others.
     *
     * @param body the body
     * @param width the model width
     * @param lag the lag
     * @param trainLength the training length
     * @param testLengths the test lengths
     * @param steps Adam steps on batches of 32
     * @param seed the seed
     * @return the accuracy over the steps with a token {@code lag} back, per test length
     */
    public static double[] lag(LayerSpec body, int width, int lag, int trainLength, int[] testLengths, int steps,
            long seed) {
        int vocab = 6;
        SplittableRandom random = new SplittableRandom(seed);
        TinyModel m = model(body, vocab, trainLength, width, false, random);
        List<Var> params = m.parameters();
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int[][] x = randomTokens(random, 32, trainLength, vocab);
            lagLoss(m, x, lag, true).backward();
            opt.step(params);
        }
        double[] acc = new double[testLengths.length];
        for (int k = 0; k < acc.length; k++) {
            int len = testLengths[k];
            int[][] x = randomTokens(new SplittableRandom(seed + 77 + k), 128, len, vocab);
            float[] l = m.logits(x, false).value().argmax(2).toArray();
            int correct = 0;
            for (int b = 0; b < x.length; b++) {
                for (int t = lag; t < len; t++) {
                    correct += (int) l[b * len + t] == x[b][t - lag] ? 1 : 0;
                }
            }
            acc[k] = correct / (double) (x.length * (len - lag));
        }
        return acc;
    }

    /**
     * Finds every {@link RefinedAttention} inside a layer (through chains and blocks).
     *
     * @param layer the layer
     * @return the attention layers, in order
     */
    public static List<RefinedAttention> attentions(Layer layer) {
        List<RefinedAttention> out = new ArrayList<>();
        if (layer instanceof RefinedAttention a) {
            out.add(a);
        } else if (layer instanceof LayerChain c) {
            c.layers().forEach(l -> out.addAll(attentions(l)));
        } else if (layer instanceof ResidualMixerBlock b) {
            out.addAll(attentions(b.mixer()));
        }
        return out;
    }

    /**
     * Counts a layer's parameters.
     *
     * @param layer the layer
     * @return the number of trainable numbers
     */
    public static long count(Layer layer) {
        long n = 0;
        for (Var p : layer.parameters()) {
            n += p.value().size();
        }
        return n;
    }

    /**
     * Builds an untrained model.
     *
     * @param body the body spec
     * @param vocab the vocabulary
     * @param length the sequence length the body is built for
     * @param width the width
     * @param previousToken whether to add a previous-token embedding
     * @param random the random source
     * @return the model
     */
    public static TinyModel model(LayerSpec body, int vocab, int length, int width, boolean previousToken,
            SplittableRandom random) {
        Layer layer = body.create(Shape.of(length, width), random);
        Var embed = Var.param(Tensor.normal(Shape.of(vocab, width), 0, 1, random));
        Var previous = previousToken ? Var.param(Tensor.normal(Shape.of(vocab, width), 0, 1, random)) : null;
        Var readout = Var.param(Tensor.uniform(Shape.of(width, vocab), -1, 1, random)
                .mul((float) Math.sqrt(3.0 / width)));
        return new TinyModel(embed, previous, layer, readout);
    }

    private static Var lastLogits(TinyModel m, int[][] tokens, boolean training) {
        int b = tokens.length;
        int t = tokens[0].length;
        int width = m.embed().shape().dim(1);
        return m.readout(m.hidden(tokens, training).slice(1, t - 1, t).reshape(b, width));
    }

    private static Var lagLoss(TinyModel m, int[][] x, int lag, boolean training) {
        int b = x.length;
        int t = x[0].length;
        int vocab = m.readout().shape().dim(1);
        Var logits = m.logits(x, training).slice(1, lag, t).reshape(b * (t - lag), vocab);
        float[] target = new float[b * (t - lag)];
        for (int i = 0; i < b; i++) {
            for (int j = lag; j < t; j++) {
                target[i * (t - lag) + j - lag] = x[i][j - lag];
            }
        }
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(target.length), target));
    }

    private static int[][] randomTokens(SplittableRandom random, int batch, int length, int vocab) {
        int[][] x = new int[batch][length];
        for (int[] row : x) {
            for (int i = 0; i < length; i++) {
                row[i] = random.nextInt(vocab);
            }
        }
        return x;
    }

    private static Tensor flat(int[][] tokens) {
        int t = tokens[0].length;
        float[] f = new float[tokens.length * t];
        for (int i = 0; i < tokens.length; i++) {
            for (int j = 0; j < t; j++) {
                f[i * t + j] = tokens[i][j];
            }
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    private static Tensor ids(int[] values) {
        float[] f = new float[values.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = values[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    private static int[][] successors() {
        SplittableRandom r = new SplittableRandom(7);
        int[][] s = new int[VOCAB][3];
        for (int v = 0; v < VOCAB; v++) {
            for (int k = 0; k < 3; k++) {
                s[v][k] = r.nextInt(VOCAB);
            }
        }
        return s;
    }
}
