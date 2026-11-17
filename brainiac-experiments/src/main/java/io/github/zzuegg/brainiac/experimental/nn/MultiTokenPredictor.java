package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A causal language model with multi-token prediction: a shared trunk and {@code k}
 * independent output heads, where head {@code i} predicts the token {@code i + 1} steps
 * ahead, all through one shared unembedding. Training on all heads gives a denser signal
 * per sequence (better sample efficiency); at inference the extra heads draft tokens that
 * the next-token head verifies in one forward pass, a self-speculative decoding that
 * produces exactly the greedy output with fewer forward passes.
 *
 * <p>Each head is a residual SiLU MLP on the trunk output. The trunk is any causal
 * {@code [b, t, d] -> [b, t, d]} layer (applied with a residual connection).
 *
 * <p>Papers: Gloeckle et al., "Better and Faster Large Language Models via Multi-token
 * Prediction", <a href="https://arxiv.org/abs/2404.19737">arXiv:2404.19737</a>; Stern et al.,
 * "Blockwise Parallel Decoding", <a href="https://arxiv.org/abs/1811.03115">arXiv:1811.03115</a>.
 */
public final class MultiTokenPredictor {

    /**
     * The result of speculative decoding.
     *
     * @param tokens the generated tokens (without the prompt)
     * @param forwardPasses how many model forward passes it took
     */
    public record Decoded(int[] tokens, int forwardPasses) {
        /** @return the mean number of tokens accepted per forward pass */
        public double tokensPerPass() {
            return tokens.length / (double) forwardPasses;
        }
    }

    private final int vocabulary;
    private final int features;
    private final int horizon;
    private final Layer trunk;
    private final Var embedding;
    private final Var unembedding;
    private final List<Var[]> heads = new ArrayList<>();

    /**
     * Creates the model.
     *
     * @param vocabulary the number of token ids
     * @param features the model width
     * @param horizon {@code k}, the number of heads (tokens predicted per position)
     * @param trunk a causal sequence layer {@code [b, t, features] -> [b, t, features]}
     * @param random the random source
     */
    public MultiTokenPredictor(int vocabulary, int features, int horizon, Layer trunk, RandomGenerator random) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be at least 1");
        }
        this.vocabulary = vocabulary;
        this.features = features;
        this.horizon = horizon;
        this.trunk = trunk;
        this.embedding = Var.param(Tensor.normal(Shape.of(vocabulary, features), 0, 1, random));
        this.unembedding = MixerOps.weight(random, features, vocabulary);
        for (int i = 0; i < horizon; i++) {
            heads.add(new Var[] {MixerOps.weight(random, features, 2 * features),
                MixerOps.weight(random, 2 * features, features)});
        }
    }

    /** @return the number of heads */
    public int horizon() {
        return horizon;
    }

    /**
     * Computes every head's logits.
     *
     * @param tokens {@code [batch][steps]} token ids
     * @return {@code horizon} logits of shape {@code [batch, steps, vocabulary]}; head {@code i}
     *     at step {@code t} predicts token {@code t + 1 + i}
     */
    public Var[] logits(int[][] tokens) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
            }
        }
        Var x = embedding.gather(0, Tensor.of(Shape.of(b * t), ids)).reshape(b, t, features);
        Var h = x.add(trunk.forward(x));
        Var[] out = new Var[horizon];
        for (int i = 0; i < horizon; i++) {
            Var[] w = heads.get(i);
            out[i] = h.add(MixerOps.mlp(h, w[0], w[1])).matmul(unembedding);
        }
        return out;
    }

    /**
     * The training loss: the mean over heads of each head's cross-entropy against the
     * token {@code i + 1} steps ahead.
     *
     * @param tokens {@code [batch][steps]} training sequences
     * @return a scalar
     */
    public Var loss(int[][] tokens) {
        Var[] logits = logits(tokens);
        int b = tokens.length;
        int t = tokens[0].length;
        Var total = null;
        for (int i = 0; i < horizon; i++) {
            float[] target = new float[b * t];
            for (int n = 0; n < b; n++) {
                for (int s = 0; s < t; s++) {
                    target[n * t + s] = s + 1 + i < t ? tokens[n][s + 1 + i] : -1;
                }
            }
            Var l = Functional.crossEntropy(logits[i].reshape(b * t, vocabulary), Tensor.of(Shape.of(b * t), target));
            total = total == null ? l : total.add(l);
        }
        return total.div(horizon);
    }

    /**
     * Greedy decoding with the next-token head only: one forward pass per token.
     *
     * @param prompt the prompt, at least one token
     * @param count how many tokens to generate
     * @return the generated tokens
     */
    public int[] generateGreedy(int[] prompt, int count) {
        int[] seq = prompt.clone();
        for (int i = 0; i < count; i++) {
            Var[] l = logits(new int[][] {seq});
            int next = argmax(l[0], seq.length - 1);
            seq = Arrays.copyOf(seq, seq.length + 1);
            seq[seq.length - 1] = next;
        }
        return Arrays.copyOfRange(seq, prompt.length, seq.length);
    }

    /**
     * Self-speculative greedy decoding: the heads draft {@code horizon} tokens, one forward
     * pass over the draft checks them with the next-token head, and the longest agreeing
     * prefix is kept. The output equals {@link #generateGreedy} exactly.
     *
     * @param prompt the prompt, at least one token
     * @param count how many tokens to generate
     * @return the tokens and the number of forward passes used
     */
    public Decoded generateSpeculative(int[] prompt, int count) {
        int[] seq = prompt.clone();
        Var[] l = logits(new int[][] {seq});
        int passes = 1;
        int[] draft = drafts(l, seq.length - 1);
        while (seq.length - prompt.length < count) {
            int[] candidate = Arrays.copyOf(seq, seq.length + draft.length);
            System.arraycopy(draft, 0, candidate, seq.length, draft.length);
            l = logits(new int[][] {candidate});
            passes++;
            int accept = 1;   // draft[0] came from the next-token head, so it is exact
            while (accept < draft.length && argmax(l[0], seq.length + accept - 1) == draft[accept]) {
                accept++;
            }
            accept = Math.min(accept, count - (seq.length - prompt.length));
            int[] next = Arrays.copyOf(seq, seq.length + accept);
            System.arraycopy(draft, 0, next, seq.length, accept);
            seq = next;
            draft = drafts(l, seq.length - 1);
        }
        return new Decoded(Arrays.copyOfRange(seq, prompt.length, seq.length), passes);
    }

    private int[] drafts(Var[] logits, int position) {
        int[] d = new int[horizon];
        for (int i = 0; i < horizon; i++) {
            d[i] = argmax(logits[i], position);
        }
        return d;
    }

    private int argmax(Var logits, int position) {
        float[] row = logits.value().slice(1, position, position + 1).toArray();
        int best = 0;
        for (int i = 1; i < row.length; i++) {
            if (row[i] > row[best]) {
                best = i;
            }
        }
        return best;
    }

    /** @return all trainable parameters, trunk included */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, unembedding));
        heads.forEach(h -> p.addAll(List.of(h)));
        p.addAll(trunk.parameters());
        return p;
    }
}
