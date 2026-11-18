package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Generative retrieval with semantic IDs, TIGER (Rajput et al., 2023,
 * <a href="https://arxiv.org/abs/2305.05065">arXiv:2305.05065</a>). A user's history is written as
 * the semantic-ID tokens ({@link SemanticIdTokenizer}) of its last items; a transformer
 * encoder-decoder ({@link TransformerSeq2Seq}) generates the next item's ID token by token.
 * Retrieval becomes generation: a beam search, constrained by a prefix trie to IDs of real items,
 * returns the top-k items. Because IDs come from content, the model can also emit items it never
 * saw as targets, if their content codes are where it learned to go (cold start).
 *
 * <p>The 2025 scaling study (<a href="https://arxiv.org/abs/2509.25522">arXiv:2509.25522</a>) finds
 * semantic-ID models saturate early in model size, while ID-based recommenders keep improving.
 */
public final class GenerativeRetriever {

    private final SemanticIdTokenizer ids;
    private final int history;
    private final TransformerSeq2Seq model;
    private final Map<Long, int[]> children = new HashMap<>();
    private final Map<Long, Integer> leaves = new HashMap<>();
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    /**
     * Creates the model and the prefix trie over all items' IDs.
     *
     * @param ids the items' semantic IDs
     * @param history items of history the encoder reads
     * @param width the transformer width
     * @param heads attention heads
     * @param layers encoder and decoder blocks
     * @param seed the seed
     */
    public GenerativeRetriever(SemanticIdTokenizer ids, int history, int width, int heads, int layers, long seed) {
        this.ids = ids;
        this.history = history;
        this.model = TransformerSeq2Seq.of(ids.vocabulary(), ids.vocabulary(), width, heads, layers, seed);
        Map<Long, java.util.Set<Integer>> next = new HashMap<>();
        for (int item = 0; item < ids.items(); item++) {
            int[] t = ids.tokens(item);
            long prefix = 0;
            for (int token : t) {
                next.computeIfAbsent(prefix, k -> new java.util.TreeSet<>()).add(token);
                prefix = extend(prefix, token);
            }
            leaves.put(prefix, item);
        }
        next.forEach((k, v) -> children.put(k, v.stream().mapToInt(Integer::intValue).toArray()));
    }

    private long extend(long prefix, int token) {
        return prefix * (ids.vocabulary() + 1) + token + 1;
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public GenerativeRetriever optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    private int[][] source(int[][] histories) {
        int len = ids.length();
        int[][] src = new int[histories.length][history * len];
        for (int b = 0; b < histories.length; b++) {
            int[] h = histories[b];
            for (int i = 0; i < history; i++) {
                System.arraycopy(ids.tokens(h[h.length - history + i]), 0, src[b], i * len, len);
            }
        }
        return src;
    }

    /**
     * One teacher-forced step.
     *
     * @param histories item histories, oldest first, at least {@code history} long
     * @param next each history's next item
     * @return the loss
     */
    public double trainStep(int[][] histories, int[] next) {
        int[][] target = new int[next.length][];
        for (int b = 0; b < next.length; b++) {
            target[b] = ids.tokens(next[b]);
        }
        Var loss = model.loss(source(histories), target);
        loss.backward();
        optimizer.step(model.parameters());
        return loss.value().item();
    }

    /**
     * Constrained beam search: only prefixes of real items' IDs survive.
     *
     * @param histories item histories, oldest first
     * @param k the beam width and the number of items returned
     * @return {@code [batch][<= k]} items, most likely first
     */
    public int[][] recommend(int[][] histories, int k) {
        return Var.noGrad(() -> search(histories, k));
    }

    private int[][] search(int[][] histories, int k) {
        int b = histories.length;
        int len = ids.length();
        int vocab = ids.vocabulary();
        Var memory = model.encode(source(histories), false);
        float[] rows = new float[b * k];
        for (int i = 0; i < b * k; i++) {
            rows[i] = i / k;
        }
        Var beamMemory = memory.gather(0, Tensor.of(Shape.of(b * k), rows));
        int[][][] beams = new int[b][1][0];
        double[][] score = new double[b][];
        for (int i = 0; i < b; i++) {
            score[i] = new double[] {0};
        }
        for (int step = 0; step < len; step++) {
            int[][] in = new int[b * k][step + 1];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < k; j++) {
                    int[] beam = beams[i][Math.min(j, beams[i].length - 1)];
                    in[i * k + j][0] = vocab;
                    System.arraycopy(beam, 0, in[i * k + j], 1, step);
                }
            }
            Var logits = model.decode(beamMemory, in, false).slice(1, step, step + 1).reshape(b * k, vocab);
            float[] logp = Activation.LOG_SOFTMAX.forward(logits).value().toArray();
            for (int i = 0; i < b; i++) {
                List<double[]> cand = new ArrayList<>();
                for (int j = 0; j < beams[i].length; j++) {
                    long prefix = 0;
                    for (int t : beams[i][j]) {
                        prefix = extend(prefix, t);
                    }
                    for (int t : children.getOrDefault(prefix, new int[0])) {
                        cand.add(new double[] {score[i][j] + logp[(i * k + j) * vocab + t], j, t});
                    }
                }
                cand.sort((x, y) -> Double.compare(y[0], x[0]));
                int keep = Math.min(k, cand.size());
                int[][] nb = new int[keep][];
                double[] ns = new double[keep];
                for (int c = 0; c < keep; c++) {
                    double[] e = cand.get(c);
                    int[] old = beams[i][(int) e[1]];
                    nb[c] = Arrays.copyOf(old, old.length + 1);
                    nb[c][old.length] = (int) e[2];
                    ns[c] = e[0];
                }
                beams[i] = nb;
                score[i] = ns;
            }
        }
        int[][] out = new int[b][];
        for (int i = 0; i < b; i++) {
            out[i] = new int[beams[i].length];
            for (int j = 0; j < beams[i].length; j++) {
                long prefix = 0;
                for (int t : beams[i][j]) {
                    prefix = extend(prefix, t);
                }
                out[i][j] = leaves.get(prefix);
            }
        }
        return out;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return model.parameters();
    }
}
