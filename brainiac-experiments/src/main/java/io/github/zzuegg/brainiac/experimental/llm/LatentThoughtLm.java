package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A recurrent language model that can reason in latent space, as in Coconut ("Chain of
 * Continuous Thought", Hao et al., <a href="https://arxiv.org/abs/2412.06769">2412.06769</a>):
 * instead of writing a reasoning step as a token and reading it back, the last hidden
 * state is fed back directly as the next input embedding. A continuous thought can hold
 * several candidate steps at once (a superposition, enabling breadth-first reasoning) and
 * never loses information to sampling.
 *
 * <p>Coconut is trained with a curriculum: at stage {@code k} the first {@code k} language
 * reasoning steps are replaced by {@code k} continuous thoughts and the loss covers only the
 * remaining tokens, so gradients flow back through the chain of thoughts. The cell is a
 * two-layer MLP {@code h' = tanh(silu([h, e] W1 + b1) W2 + b2)}, the output {@code h' Wo + bo}.
 */
public final class LatentThoughtLm {

    private final int vocab;
    private final int dim;
    private final Var embedding;
    private final Var w;
    private final Var b;
    private final Var w2;
    private final Var b2;
    private final Var wo;
    private final Var bo;
    private final Var h0;

    private LatentThoughtLm(int vocab, int dim, SplittableRandom r) {
        this.vocab = vocab;
        this.dim = dim;
        this.embedding = gauss(r, 0.5f, vocab, dim);
        this.w = gauss(r, (float) Math.sqrt(1.0 / (2 * dim)), 2 * dim, 2 * dim);
        this.b = Var.param(Tensor.zeros(Shape.of(1, 2 * dim)));
        this.w2 = gauss(r, (float) Math.sqrt(1.0 / (2 * dim)), 2 * dim, dim);
        this.b2 = Var.param(Tensor.zeros(Shape.of(1, dim)));
        this.wo = gauss(r, (float) Math.sqrt(1.0 / dim), dim, vocab);
        this.bo = Var.param(Tensor.zeros(Shape.of(1, vocab)));
        this.h0 = Var.param(Tensor.zeros(Shape.of(1, dim)));
    }

    private static Var gauss(SplittableRandom r, float scale, int... dims) {
        Shape s = Shape.of(dims);
        float[] v = new float[Math.toIntExact(s.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(s, v));
    }

    /**
     * @param vocab the vocabulary size
     * @param dim the hidden and embedding width (equal, so a hidden state can be an input)
     * @param seed the initialization seed
     * @return a fresh model
     */
    public static LatentThoughtLm of(int vocab, int dim, long seed) {
        return new LatentThoughtLm(vocab, dim, new SplittableRandom(seed));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.of(embedding, w, b, w2, b2, wo, bo, h0);
    }

    /** @return the vocabulary size */
    public int vocabSize() {
        return vocab;
    }

    private Var embed(int[] ids) {
        float[] oneHot = new float[ids.length * vocab];
        for (int i = 0; i < ids.length; i++) {
            oneHot[i * vocab + ids[i]] = 1;
        }
        return Var.of(Tensor.of(Shape.of(ids.length, vocab), oneHot)).matmul(embedding);
    }

    private Var cell(Var h, Var e) {
        Var u = Var.concat(1, h, e).matmul(w).add(b);
        return u.mul(u.sigmoid()).matmul(w2).add(b2).tanh();
    }

    private static int[] column(int[][] rows, int t) {
        int[] c = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            c[i] = rows[i][t];
        }
        return c;
    }

    /**
     * Reads a prefix, thinks {@code thoughts} continuous steps, then predicts the targets
     * with teacher forcing.
     *
     * @param prefix {@code [batch][length]} tokens, the same length for every row
     * @param thoughts how many hidden states to feed back as inputs
     * @param targets {@code [batch][length]} tokens to predict after the thoughts
     * @return logits {@code [batch, vocab]} for each target position
     */
    public List<Var> run(int[][] prefix, int thoughts, int[][] targets) {
        int batch = prefix.length;
        Var h = Var.of(Tensor.ones(Shape.of(batch, 1))).matmul(h0);
        for (int t = 0; t < prefix[0].length; t++) {
            h = cell(h, embed(column(prefix, t)));
        }
        for (int k = 0; k < thoughts; k++) {
            h = cell(h, h);   // the continuous thought: the hidden state is the next input
        }
        List<Var> logits = new ArrayList<>();
        int steps = targets.length == 0 ? 0 : targets[0].length;
        for (int t = 0; t < steps; t++) {
            logits.add(h.matmul(wo).add(bo));
            h = cell(h, embed(column(targets, t)));
        }
        return logits;
    }

    /**
     * @param prefix the prefixes
     * @param thoughts continuous thoughts
     * @param targets the tokens to predict
     * @return the mean cross-entropy over target positions
     */
    public Var loss(int[][] prefix, int thoughts, int[][] targets) {
        List<Var> logits = run(prefix, thoughts, targets);
        Var total = null;
        for (int t = 0; t < logits.size(); t++) {
            float[] ids = new float[targets.length];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = targets[i][t];
            }
            Var l = Functional.crossEntropy(logits.get(t), Tensor.of(Shape.of(ids.length), ids));
            total = total == null ? l : total.add(l);
        }
        return total.div(logits.size());
    }

    /**
     * Greedy generation after continuous thinking.
     *
     * @param prefix the prompt
     * @param thoughts continuous thoughts
     * @param maxNew the most tokens
     * @param end the end token
     * @return the generated tokens
     */
    public int[] generate(int[] prefix, int thoughts, int maxNew, int end) {
        Var h = h0;
        for (int t : prefix) {
            h = cell(h, embed(new int[] {t}));
        }
        for (int k = 0; k < thoughts; k++) {
            h = cell(h, h);
        }
        int[] out = new int[maxNew];
        int n = 0;
        while (n < maxNew) {
            float[] logits = h.matmul(wo).add(bo).value().toArray();
            int tok = LlmTensors.argmax(logits);
            out[n++] = tok;
            if (tok == end) {
                break;
            }
            h = cell(h, embed(new int[] {tok}));
        }
        return Arrays.copyOf(out, n);
    }
}
