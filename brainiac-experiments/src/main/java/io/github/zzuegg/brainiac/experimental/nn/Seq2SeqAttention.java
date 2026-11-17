package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A GRU encoder-decoder with attention, trained with teacher forcing: a bidirectional GRU
 * {@link Encoder} reads the source, and a GRU decoder emits the target one token at a time,
 * at each step attending over all encoder states with a {@link Scorer}:
 *
 * <ul>
 *   <li>{@link Attention#BAHDANAU}: additive scores {@code v . tanh(W s_{t-1} + U h_j)} from the
 *       previous decoder state; the context is fed into the GRU and the output layer;</li>
 *   <li>{@link Attention#LUONG}: multiplicative ("general") scores {@code h_j . (W s_t)} from the
 *       current state, an attentional vector {@code tanh(W_c [s_t; c_t])} for the output, fed
 *       back as the next input ("input feeding");</li>
 *   <li>{@link Attention#NONE}: the classic fixed-context seq2seq, where the encoder's final
 *       state is the only summary the decoder sees.</li>
 * </ul>
 *
 * <p>Sources are token ids or, with {@link #continuous}, feature vectors. Target id
 * {@code targetVocab} is the begin-of-sequence token. The encoder and scorer are public so
 * pointer networks ({@link PointerNetwork}) reuse them.
 *
 * <p>Papers: Bahdanau, Cho and Bengio, "Neural Machine Translation by Jointly Learning to Align
 * and Translate" (2014), <a href="https://arxiv.org/abs/1409.0473">arXiv:1409.0473</a>; Luong,
 * Pham and Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015),
 * <a href="https://arxiv.org/abs/1508.04025">arXiv:1508.04025</a>.
 */
public final class Seq2SeqAttention {

    /** The attention mechanism. */
    public enum Attention {
        /** no attention: a fixed context vector */
        NONE,
        /** additive attention from the previous state */
        BAHDANAU,
        /** multiplicative attention from the current state, with input feeding */
        LUONG
    }

    /** A bidirectional GRU encoder; the states are the sums of both directions. */
    public static final class Encoder {
        private final GruCell forward;
        private final GruCell backward;

        /**
         * Creates the encoder.
         *
         * @param inputs the input features per step
         * @param units the hidden units of each direction
         * @param random the source of initial weights
         */
        public Encoder(int inputs, int units, RandomGenerator random) {
            this.forward = new GruCell(inputs, units, random);
            this.backward = new GruCell(inputs, units, random);
        }

        /**
         * Encodes a sequence.
         *
         * @param x {@code [batch, steps, inputs]}
         * @return {@code {states [batch, steps, units], summary [batch, units]}}; the summary is
         *     the sum of the forward final state and the backward state at step 0
         */
        public Var[] encode(Var x) {
            int b = x.shape().dim(0);
            int t = x.shape().dim(1);
            Var pf = forward.project(x);
            Var pb = backward.project(x);
            int g = 3 * forward.units();
            Var[] fs = new Var[t];
            Var[] bs = new Var[t];
            Var h = forward.zeroState(b);
            for (int i = 0; i < t; i++) {
                h = forward.stepProjected(pf.slice(1, i, i + 1).reshape(b, g), h);
                fs[i] = h;
            }
            h = backward.zeroState(b);
            for (int i = t - 1; i >= 0; i--) {
                h = backward.stepProjected(pb.slice(1, i, i + 1).reshape(b, g), h);
                bs[i] = h;
            }
            int u = forward.units();
            Var[] steps = new Var[t];
            for (int i = 0; i < t; i++) {
                steps[i] = fs[i].add(bs[i]).reshape(b, 1, u);
            }
            return new Var[] {Var.concat(1, steps), fs[t - 1].add(bs[0])};
        }

        /** @return the hidden units */
        public int units() {
            return forward.units();
        }

        /** @return the trainable values */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(forward.parameters());
            p.addAll(backward.parameters());
            return p;
        }
    }

    /** Attention scores of one query against a sequence of keys. */
    public static final class Scorer {
        private final Attention kind;
        private final Var queryWeight;
        private final Var keyWeight;
        private final Var vector;

        /**
         * Creates a scorer.
         *
         * @param kind {@link Attention#BAHDANAU} or {@link Attention#LUONG}
         * @param queryUnits the query width
         * @param keyUnits the key width
         * @param units the additive attention's hidden width (ignored by Luong)
         * @param random the source of initial weights
         */
        public Scorer(Attention kind, int queryUnits, int keyUnits, int units, RandomGenerator random) {
            if (kind == Attention.NONE) {
                throw new IllegalArgumentException("a scorer needs an attention kind");
            }
            this.kind = kind;
            if (kind == Attention.BAHDANAU) {
                queryWeight = ClassicOps.matrix(random, queryUnits, units);
                keyWeight = ClassicOps.matrix(random, keyUnits, units);
                vector = ClassicOps.matrix(random, units, 1);
            } else {
                queryWeight = ClassicOps.matrix(random, queryUnits, keyUnits);
                keyWeight = null;
                vector = null;
            }
        }

        /**
         * Scores keys against a query.
         *
         * @param query {@code [batch, queryUnits]}
         * @param keys {@code [batch, steps, keyUnits]}
         * @return {@code [batch, steps]} scores before the softmax
         */
        public Var scores(Var query, Var keys) {
            int b = keys.shape().dim(0);
            int t = keys.shape().dim(1);
            if (kind == Attention.BAHDANAU) {
                int a = vector.shape().dim(0);
                Var q = query.matmul(queryWeight).reshape(b, 1, a);
                return keys.matmul(keyWeight).add(q).tanh().matmul(vector).reshape(b, t);
            }
            int k = keys.shape().dim(2);
            return keys.matmul(query.matmul(queryWeight).reshape(b, k, 1)).reshape(b, t);
        }

        /**
         * Softmax-weighted context.
         *
         * @param weights {@code [batch, steps]} attention weights
         * @param values {@code [batch, steps, units]}
         * @return {@code [batch, units]}
         */
        public static Var context(Var weights, Var values) {
            int b = values.shape().dim(0);
            int t = values.shape().dim(1);
            return weights.reshape(b, 1, t).matmul(values).reshape(b, values.shape().dim(2));
        }

        /** @return the trainable values */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(queryWeight));
            if (keyWeight != null) {
                p.add(keyWeight);
                p.add(vector);
            }
            return p;
        }
    }

    private final Attention attention;
    private final int sourceFeatures;
    private final int targetVocab;
    private final int units;
    private final Var sourceEmbedding;
    private final Var targetEmbedding;
    private final Encoder encoder;
    private final GruCell decoder;
    private final Scorer scorer;
    private final Var combine;
    private final Var output;
    private final Var outputBias;
    private float[][] lastAttention;

    private Seq2SeqAttention(int sourceVocab, int sourceFeatures, int targetVocab, int embed, int units,
            Attention attention, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.attention = attention;
        this.sourceFeatures = sourceFeatures;
        this.targetVocab = targetVocab;
        this.units = units;
        this.sourceEmbedding = sourceFeatures > 0 ? ClassicOps.matrix(r, sourceFeatures, embed)
                : Var.param(Tensor.normal(Shape.of(sourceVocab, embed), 0, 0.3f, r));
        this.targetEmbedding = Var.param(Tensor.normal(Shape.of(targetVocab + 1, embed), 0, 0.3f, r));
        this.encoder = new Encoder(embed, units, r);
        this.decoder = new GruCell(embed + units, units, r);
        this.scorer = attention == Attention.NONE ? null : new Scorer(attention, units, units, units, r);
        this.combine = attention == Attention.LUONG ? ClassicOps.matrix(r, 2 * units, units) : null;
        this.output = ClassicOps.matrix(r, attention == Attention.LUONG ? units : 2 * units, targetVocab);
        this.outputBias = ClassicOps.zeros(targetVocab);
    }

    /**
     * Creates a model over token sources.
     *
     * @param sourceVocab the source vocabulary
     * @param targetVocab the target vocabulary (one more id is used for begin-of-sequence)
     * @param embed the embedding width
     * @param units the GRU width
     * @param attention the attention mechanism
     * @param seed the initialization seed
     * @return the model
     */
    public static Seq2SeqAttention of(int sourceVocab, int targetVocab, int embed, int units, Attention attention,
            long seed) {
        return new Seq2SeqAttention(sourceVocab, 0, targetVocab, embed, units, attention, seed);
    }

    /**
     * Creates a model whose sources are feature vectors {@code [batch, steps, features]}.
     *
     * @param features the source features per step
     * @param targetVocab the target vocabulary
     * @param embed the embedding width
     * @param units the GRU width
     * @param attention the attention mechanism
     * @param seed the initialization seed
     * @return the model
     */
    public static Seq2SeqAttention continuous(int features, int targetVocab, int embed, int units,
            Attention attention, long seed) {
        return new Seq2SeqAttention(0, features, targetVocab, embed, units, attention, seed);
    }

    /**
     * Embeds token sources.
     *
     * @param source {@code [batch][steps]} ids
     * @return {@code [batch, steps, embed]} for {@link #loss(Var, int[][])}
     */
    public Var embedSource(int[][] source) {
        if (sourceFeatures > 0) {
            throw new IllegalStateException("this model takes feature sources");
        }
        return sourceEmbedding.gather(0, ClassicOps.ids(source));
    }

    private Var sourceInput(Var source) {
        return sourceFeatures > 0 ? source.matmul(sourceEmbedding) : source;
    }

    /** The decoder's running state for one batch. */
    private final class Decoding {
        private final Var states;
        private Var h;
        private Var feed;
        private final boolean record;
        private final List<float[]> weights = new ArrayList<>();

        Decoding(Var source, boolean record) {
            Var[] enc = encoder.encode(sourceInput(source));
            this.states = enc[0];
            int b = states.shape().dim(0);
            this.h = enc[1].tanh();
            this.feed = attention == Attention.NONE ? enc[1] : Var.of(Tensor.zeros(Shape.of(b, units)));
            this.record = record;
        }

        /** Feeds one token per example and returns the next-token logits. */
        Var step(int[] tokens) {
            Var emb = targetEmbedding.gather(0, ClassicOps.ids(tokens));
            Var out;
            switch (attention) {
                case NONE -> {
                    h = decoder.step(Var.concat(1, emb, feed), h);
                    out = Var.concat(1, h, feed);
                }
                case BAHDANAU -> {
                    Var w = Activation.SOFTMAX.forward(scorer.scores(h, states));
                    keep(w);
                    Var ctx = Scorer.context(w, states);
                    h = decoder.step(Var.concat(1, emb, ctx), h);
                    out = Var.concat(1, h, ctx);
                }
                default -> {
                    h = decoder.step(Var.concat(1, emb, feed), h);
                    Var w = Activation.SOFTMAX.forward(scorer.scores(h, states));
                    keep(w);
                    feed = Var.concat(1, h, Scorer.context(w, states)).matmul(combine).tanh();
                    out = feed;
                }
            }
            return out.matmul(output).add(outputBias);
        }

        private void keep(Var w) {
            if (record) {
                weights.add(w.value().slice(0, 0, 1).toArray());
            }
        }
    }

    /**
     * The teacher-forced cross-entropy of target sequences.
     *
     * @param source {@code [batch, steps, embed or features]}, from {@link #embedSource} or features
     * @param target {@code [batch][length]} target ids; negative ids are not scored
     * @return the mean loss
     */
    public Var loss(Var source, int[][] target) {
        int b = target.length;
        int len = target[0].length;
        int[][] in = new int[b][len];
        float[] ids = new float[b * len];
        for (int i = 0; i < b; i++) {
            in[i][0] = targetVocab;
            for (int t = 0; t < len; t++) {
                if (t > 0) {
                    in[i][t] = Math.max(0, target[i][t - 1]);
                }
                ids[t * b + i] = target[i][t];
            }
        }
        Decoding d = new Decoding(source, false);
        Var[] logits = new Var[len];
        int[] col = new int[b];
        for (int t = 0; t < len; t++) {
            for (int i = 0; i < b; i++) {
                col[i] = in[i][t];
            }
            logits[t] = d.step(col);
        }
        return Functional.crossEntropy(Var.concat(0, logits), Tensor.of(Shape.of(b * len), ids));
    }

    /**
     * Greedy decoding.
     *
     * @param source the source, as for {@link #loss}
     * @param length the tokens to emit
     * @return {@code [batch][length]} ids
     */
    public int[][] greedy(Var source, int length) {
        int b = source.shape().dim(0);
        int[][] out = new int[b][length];
        int[] col = new int[b];
        java.util.Arrays.fill(col, targetVocab);
        Decoding d = new Decoding(source, true);
        for (int t = 0; t < length; t++) {
            col = ClassicOps.argmax(d.step(col).value());
            for (int i = 0; i < b; i++) {
                out[i][t] = col[i];
            }
        }
        lastAttention = attention == Attention.NONE ? null : d.weights.toArray(float[][]::new);
        return out;
    }

    /**
     * The attention weights of the first example from the last full greedy decode.
     *
     * @return {@code [step][sourceStep]}, or {@code null} without attention
     */
    public float[][] lastAttention() {
        return lastAttention;
    }

    /** @return the attention mechanism */
    public Attention attention() {
        return attention;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(sourceEmbedding, targetEmbedding));
        p.addAll(encoder.parameters());
        p.addAll(decoder.parameters());
        if (scorer != null) {
            p.addAll(scorer.parameters());
        }
        if (combine != null) {
            p.add(combine);
        }
        p.add(output);
        p.add(outputBias);
        return p;
    }

    @Override
    public String toString() {
        return "Seq2SeqAttention(" + attention + ", " + units + " units)";
    }
}
