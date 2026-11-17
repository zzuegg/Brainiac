package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A pointer network: a seq2seq model whose output "vocabulary" is the positions of its own
 * input. It reuses {@link Seq2SeqAttention}'s bidirectional {@link Seq2SeqAttention.Encoder}
 * and additive {@link Seq2SeqAttention.Scorer}, but instead of blending the encoder states
 * into a context, the attention distribution itself is the output: at each step the decoder
 * points at one input element, and the embedding of that element is its next input. The
 * output space therefore grows with the input, so a model trained on 5 to 8 elements can
 * sort, or find the convex hull of, 10. An optional learned stop key lets it emit
 * variable-length answers.
 *
 * <p>{@link Generator} is the pointer-generator of See et al.: a seq2seq summariser whose
 * output distribution mixes generating from a small vocabulary with copying source tokens
 * through the attention, {@code P(w) = p_gen P_vocab(w) + (1 - p_gen) sum_{i: x_i = w} a_i},
 * so it can reproduce tokens it could never generate.
 *
 * <p>Papers: Vinyals, Fortunato and Jaitly, "Pointer Networks" (NeurIPS 2015),
 * <a href="https://arxiv.org/abs/1506.03134">arXiv:1506.03134</a>; See, Liu and Manning, "Get To
 * The Point: Summarization with Pointer-Generator Networks" (ACL 2017),
 * <a href="https://arxiv.org/abs/1704.04368">arXiv:1704.04368</a>.
 */
public final class PointerNetwork {

    private final int features;
    private final int units;
    private final boolean stop;
    private final Var inputProjection;
    private final Seq2SeqAttention.Encoder encoder;
    private final GruCell decoder;
    private final Seq2SeqAttention.Scorer scorer;
    private final Var start;
    private final Var stopKey;

    private PointerNetwork(int features, int units, boolean stop, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.features = features;
        this.units = units;
        this.stop = stop;
        this.inputProjection = ClassicOps.matrix(r, features, units);
        this.encoder = new Seq2SeqAttention.Encoder(units, units, r);
        this.decoder = new GruCell(units, units, r);
        this.scorer = new Seq2SeqAttention.Scorer(Seq2SeqAttention.Attention.BAHDANAU, units, units, units, r);
        this.start = Var.param(Tensor.normal(Shape.of(1, units), 0, 0.3f, r));
        this.stopKey = stop ? Var.param(Tensor.normal(Shape.of(1, 1, units), 0, 0.3f, r)) : null;
    }

    /**
     * Creates a pointer network.
     *
     * @param features the features of each input element
     * @param units the hidden width
     * @param stopToken whether position {@code steps} (one past the last input) means "stop"
     * @param seed the initialization seed
     * @return the network
     */
    public static PointerNetwork of(int features, int units, boolean stopToken, long seed) {
        return new PointerNetwork(features, units, stopToken, seed);
    }

    /** Encodes; returns {@code {keys, element embeddings, summary}}, both with the stop slot. */
    private Var[] encode(Var inputs) {
        if (inputs.shape().rank() != 3 || inputs.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got "
                    + inputs.shape());
        }
        Var x = inputs.matmul(inputProjection);
        Var[] enc = encoder.encode(x);
        Var keys = enc[0];
        if (stop) {
            int b = x.shape().dim(0);
            Var s = stopKey.expand(Shape.of(b, 1, units));
            keys = Var.concat(1, keys, s);
            x = Var.concat(1, x, s);
        }
        return new Var[] {keys, x, enc[1]};
    }

    private static Var pick(Var elements, int[] index) {
        int b = elements.shape().dim(0);
        int n = elements.shape().dim(1);
        int u = elements.shape().dim(2);
        float[] flat = new float[b];
        for (int i = 0; i < b; i++) {
            flat[i] = (float) i * n + index[i];
        }
        return elements.reshape(b * n, u).gather(0, Tensor.of(Shape.of(b), flat));
    }

    /**
     * Teacher-forced cross-entropy of pointer targets.
     *
     * @param inputs {@code [batch, steps, features]}
     * @param targets {@code [batch][length]} positions in {@code [0, steps)}, or {@code steps} for
     *     stop; negative entries are not scored
     * @return the mean loss
     */
    public Var loss(Var inputs, int[][] targets) {
        Var[] e = encode(inputs);
        int b = targets.length;
        int len = targets[0].length;
        Var h = e[2].tanh();
        Var in = start.expand(Shape.of(b, units));
        Var[] logits = new Var[len];
        float[] ids = new float[b * len];
        int[] prev = new int[b];
        for (int t = 0; t < len; t++) {
            h = decoder.step(in, h);
            logits[t] = scorer.scores(h, e[0]);
            for (int i = 0; i < b; i++) {
                ids[t * b + i] = targets[i][t];
                prev[i] = Math.max(0, targets[i][t]);
            }
            in = pick(e[1], prev);
        }
        return Functional.crossEntropy(Var.concat(0, logits), Tensor.of(Shape.of(b * len), ids));
    }

    /**
     * Greedy pointing.
     *
     * @param inputs {@code [batch, steps, features]}
     * @param length the most outputs
     * @param noRepeat whether a position may be chosen only once (for permutations)
     * @return {@code [batch][length]} positions; after a stop, the rest is {@code -1}
     */
    public int[][] decode(Var inputs, int length, boolean noRepeat) {
        Var[] e = encode(inputs);
        int b = inputs.shape().dim(0);
        int n = e[0].shape().dim(1);
        int real = inputs.shape().dim(1);
        Var h = e[2].tanh();
        Var in = start.expand(Shape.of(b, units));
        int[][] out = new int[b][length];
        boolean[][] used = new boolean[b][n];
        boolean[] done = new boolean[b];
        for (int t = 0; t < length; t++) {
            h = decoder.step(in, h);
            float[] s = scorer.scores(h, e[0]).value().toArray();
            int[] pick = new int[b];
            for (int i = 0; i < b; i++) {
                int best = -1;
                for (int j = 0; j < n; j++) {
                    if (noRepeat && used[i][j] && j < real) {
                        continue;
                    }
                    if (best < 0 || s[i * n + j] > s[i * n + best]) {
                        best = j;
                    }
                }
                used[i][best] = true;
                pick[i] = best;
                out[i][t] = done[i] ? -1 : best;
                done[i] |= stop && best == real;
            }
            in = pick(e[1], pick);
        }
        return out;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inputProjection, start));
        p.addAll(encoder.parameters());
        p.addAll(decoder.parameters());
        p.addAll(scorer.parameters());
        if (stopKey != null) {
            p.add(stopKey);
        }
        return p;
    }

    @Override
    public String toString() {
        return "PointerNetwork(" + features + " features, " + units + " units" + (stop ? ", stop" : "") + ")";
    }

    /**
     * The pointer-generator: an attentional GRU seq2seq over a shared id space of
     * {@code vocab} tokens whose generator only knows the first {@code generatorVocab} (the
     * rest are unknown to it, embedded as one UNK token), mixed with copying through the
     * attention.
     */
    public static final class Generator {
        private final int vocab;
        private final int generatorVocab;
        private final int units;
        private final Var embedding;
        private final Seq2SeqAttention.Encoder encoder;
        private final GruCell decoder;
        private final Seq2SeqAttention.Scorer scorer;
        private final Var output;
        private final Var outputBias;
        private final Var gate;
        private final Var gateBias;
        private final List<int[]> lastGreedy = new ArrayList<>();
        private float lastGenerate = Float.NaN;

        /**
         * Creates the model.
         *
         * @param vocab the full id space (ids at or above {@code generatorVocab} can only be copied)
         * @param generatorVocab the ids the generator can produce
         * @param embed the embedding width
         * @param units the GRU width
         * @param seed the initialization seed
         */
        public Generator(int vocab, int generatorVocab, int embed, int units, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.vocab = vocab;
            this.generatorVocab = generatorVocab;
            this.units = units;
            this.embedding = Var.param(Tensor.normal(Shape.of(generatorVocab + 2, embed), 0, 0.3f, r));
            this.encoder = new Seq2SeqAttention.Encoder(embed, units, r);
            this.decoder = new GruCell(embed + units, units, r);
            this.scorer = new Seq2SeqAttention.Scorer(Seq2SeqAttention.Attention.BAHDANAU, units, units, units, r);
            this.output = ClassicOps.matrix(r, 2 * units, generatorVocab);
            this.outputBias = ClassicOps.zeros(generatorVocab);
            this.gate = ClassicOps.matrix(r, 2 * units + embed, 1);
            this.gateBias = ClassicOps.zeros(1);
        }

        private int[][] mapUnknown(int[][] ids) {
            int[][] m = new int[ids.length][];
            for (int i = 0; i < ids.length; i++) {
                m[i] = new int[ids[i].length];
                for (int j = 0; j < ids[i].length; j++) {
                    m[i][j] = ids[i][j] < generatorVocab ? ids[i][j] : generatorVocab;
                }
            }
            return m;
        }

        /** The copy distribution's scatter indices: {@code i * vocab + source[i][j]}. */
        private Tensor scatterIndex(int[][] source) {
            int b = source.length;
            int s = source[0].length;
            float[] f = new float[b * s];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < s; j++) {
                    f[i * s + j] = (float) i * vocab + source[i][j];
                }
            }
            return Tensor.of(Shape.of(b * s), f);
        }

        /** Runs teacher-forced or greedy decoding; returns the per-step probabilities over {@code vocab}. */
        private List<Var> run(int[][] source, int[][] target, int length) {
            int b = source.length;
            int s = source[0].length;
            Var[] enc = encoder.encode(embedding.gather(0, ClassicOps.ids(mapUnknown(source))));
            Var states = enc[0];
            Var h = enc[1].tanh();
            Tensor scatter = scatterIndex(source);
            Var pad = vocab > generatorVocab ? Var.of(Tensor.zeros(Shape.of(b, vocab - generatorVocab))) : null;
            int[] prev = new int[b];
            java.util.Arrays.fill(prev, generatorVocab + 1);
            List<Var> probs = new ArrayList<>();
            double gen = 0;
            for (int t = 0; t < length; t++) {
                Var emb = embedding.gather(0, ClassicOps.ids(prev));
                Var w = Activation.SOFTMAX.forward(scorer.scores(h, states));
                Var ctx = Seq2SeqAttention.Scorer.context(w, states);
                h = decoder.step(Var.concat(1, emb, ctx), h);
                Var hc = Var.concat(1, h, ctx);
                Var pVocab = Activation.SOFTMAX.forward(hc.matmul(output).add(outputBias));
                if (pad != null) {
                    pVocab = Var.concat(1, pVocab, pad);
                }
                Var pGen = Var.concat(1, hc, emb).matmul(gate).add(gateBias).sigmoid();
                Var copy = w.reshape(b * s).scatterAdd(0, scatter, b * vocab).reshape(b, vocab);
                Var p = pVocab.mul(pGen).add(copy.mul(pGen.neg().add(1f)));
                probs.add(p);
                gen += pGen.value().mean().item();
                int[] chosen = target == null ? ClassicOps.argmax(p.value()) : null;
                for (int i = 0; i < b; i++) {
                    int id = target != null ? Math.max(0, target[i][t]) : chosen[i];
                    prev[i] = id < generatorVocab ? id : generatorVocab;
                }
                if (chosen != null) {
                    lastGreedy.add(chosen);
                }
            }
            lastGenerate = (float) (gen / length);
            return probs;
        }

        /**
         * Teacher-forced negative log-likelihood of target sequences.
         *
         * @param source {@code [batch][steps]} ids in {@code [0, vocab)}
         * @param target {@code [batch][length]} ids; negative entries are not scored
         * @return the mean loss
         */
        public Var loss(int[][] source, int[][] target) {
            int b = target.length;
            int len = target[0].length;
            List<Var> probs = run(source, target, len);
            Var total = null;
            int counted = 0;
            for (int t = 0; t < len; t++) {
                float[] idx = new float[b];
                float[] keep = new float[b];
                for (int i = 0; i < b; i++) {
                    idx[i] = Math.max(0, target[i][t]);
                    keep[i] = target[i][t] >= 0 ? 1 : 0;
                    counted += (int) keep[i];
                }
                Var picked = probs.get(t).takeAlong(Tensor.of(Shape.of(b, 1), idx)).reshape(b);
                Var nll = picked.add(1e-9f).log().neg().mul(Tensor.of(Shape.of(b), keep)).sum();
                total = total == null ? nll : total.add(nll);
            }
            return total.div(Math.max(1, counted));
        }

        /**
         * Greedy decoding.
         *
         * @param source {@code [batch][steps]} ids
         * @param length the tokens to emit
         * @return {@code [batch][length]} ids
         */
        public int[][] greedy(int[][] source, int length) {
            lastGreedy.clear();
            run(source, null, length);
            int b = source.length;
            int[][] out = new int[b][length];
            for (int t = 0; t < length; t++) {
                for (int i = 0; i < b; i++) {
                    out[i][t] = lastGreedy.get(t)[i];
                }
            }
            return out;
        }

        /** @return the mean generation probability {@code p_gen} of the last run */
        public float lastGenerationProbability() {
            return lastGenerate;
        }

        /** @return the hidden width */
        public int units() {
            return units;
        }

        /** @return the trainable values */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embedding));
            p.addAll(encoder.parameters());
            p.addAll(decoder.parameters());
            p.addAll(scorer.parameters());
            p.addAll(List.of(output, outputBias, gate, gateBias));
            return p;
        }
    }
}
