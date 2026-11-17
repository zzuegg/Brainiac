package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An interpolated Kneser-Ney n-gram language model over token ids (Kneser and Ney 1995;
 * Chen and Goodman 1998, "An Empirical Study of Smoothing Techniques"), the strongest
 * classical baseline and still the backbone of infini-gram style retrieval: counts of
 * {@code n}-grams with an absolute discount {@code D}, the freed mass given to the next
 * lower order, and lower orders counting <em>continuations</em> (in how many different
 * contexts a word appears) rather than occurrences, so "Francisco" is not a likely word
 * after an unseen context just because "San Francisco" is frequent. The lowest order is
 * mixed with a uniform distribution so no token has probability zero.
 *
 * <p>It is a {@link LanguageModel} (its "logits" are log-probabilities), so the core
 * {@code Generator} samples from it and it can serve as a free draft model or amateur.
 */
public final class KneserNeyLm implements LanguageModel {

    private final int order;
    private final int vocab;
    private final double discount;
    /** Per order {@code n} (1-based): counts of {@code n}-grams (highest: raw; lower: continuation). */
    private final List<Map<List<Integer>, Integer>> counts;
    /** Per order: for each context, the total of its counts and its number of distinct followers. */
    private final List<Map<List<Integer>, int[]>> contexts;

    private KneserNeyLm(int order, int vocab, double discount, List<Map<List<Integer>, Integer>> counts,
            List<Map<List<Integer>, int[]>> contexts) {
        this.order = order;
        this.vocab = vocab;
        this.discount = discount;
        this.counts = counts;
        this.contexts = contexts;
    }

    /**
     * Counts a corpus.
     *
     * @param corpus token sequences
     * @param order the n-gram order, at least 1
     * @param vocab the vocabulary size
     * @param discount the absolute discount, usually 0.75
     * @return the model
     */
    public static KneserNeyLm train(int[][] corpus, int order, int vocab, double discount) {
        List<Map<List<Integer>, Integer>> counts = new java.util.ArrayList<>();
        for (int n = 0; n <= order; n++) {
            counts.add(new HashMap<>());
        }
        Map<List<Integer>, Integer> top = counts.get(order);
        for (int[] seq : corpus) {
            for (int n = 1; n <= order; n++) {
                for (int i = 0; i + n <= seq.length; i++) {
                    List<Integer> g = gram(seq, i, i + n);
                    if (n == order || i == 0) {
                        // highest order, and n-grams at a sequence start (no left context), count raw
                        counts.get(n).merge(g, 1, Integer::sum);
                    }
                }
            }
        }
        // continuation counts for lower orders: distinct left extensions
        for (int n = order; n >= 2; n--) {
            Set<List<Integer>> seen = new HashSet<>();
            for (int[] seq : corpus) {
                for (int i = 0; i + n <= seq.length; i++) {
                    seen.add(gram(seq, i, i + n));
                }
            }
            for (List<Integer> g : seen) {
                counts.get(n - 1).merge(g.subList(1, n), 1, Integer::sum);
            }
        }
        List<Map<List<Integer>, int[]>> contexts = new java.util.ArrayList<>();
        for (int n = 0; n <= order; n++) {
            Map<List<Integer>, int[]> c = new HashMap<>();
            if (n > 0) {
                for (Map.Entry<List<Integer>, Integer> e : counts.get(n).entrySet()) {
                    int[] stat = c.computeIfAbsent(e.getKey().subList(0, n - 1), k -> new int[2]);
                    stat[0] += e.getValue();
                    stat[1]++;
                }
            }
            contexts.add(c);
        }
        if (top.isEmpty() && order > 1) {
            throw new IllegalArgumentException("the corpus has no " + order + "-grams");
        }
        return new KneserNeyLm(order, vocab, discount, counts, contexts);
    }

    private static List<Integer> gram(int[] seq, int from, int to) {
        Integer[] g = new Integer[to - from];
        for (int i = from; i < to; i++) {
            g[i - from] = seq[i];
        }
        return List.of(g);
    }

    /** @return the n-gram order */
    public int order() {
        return order;
    }

    /**
     * @param context the tokens before (only the last {@code order - 1} matter)
     * @param token the next token
     * @return its probability
     */
    public double probability(int[] context, int token) {
        int k = Math.min(order - 1, context.length);
        int[] h = Arrays.copyOfRange(context, context.length - k, context.length);
        return prob(h, token);
    }

    private double prob(int[] h, int w) {
        int n = h.length + 1;
        if (n == 1) {
            int[] stat = contexts.get(1).get(List.of());
            if (stat == null) {
                return 1.0 / vocab;
            }
            Integer c = counts.get(1).get(List.of(w));
            double p = Math.max((c == null ? 0 : c) - discount, 0) / stat[0];
            double lambda = discount * stat[1] / stat[0];
            return p + lambda / vocab;
        }
        List<Integer> ctx = gram(h, 0, h.length);
        int[] stat = contexts.get(n).get(ctx);
        int[] lower = Arrays.copyOfRange(h, 1, h.length);
        if (stat == null) {
            return prob(lower, w);
        }
        Integer[] g = new Integer[n];
        for (int i = 0; i < h.length; i++) {
            g[i] = h[i];
        }
        g[h.length] = w;
        Integer c = counts.get(n).get(List.of(g));
        double p = Math.max((c == null ? 0 : c) - discount, 0) / stat[0];
        double lambda = discount * stat[1] / stat[0];
        return p + lambda * prob(lower, w);
    }

    /**
     * @param context the tokens before
     * @return the next-token distribution
     */
    public double[] distribution(int[] context) {
        double[] p = new double[vocab];
        for (int t = 0; t < vocab; t++) {
            p[t] = probability(context, t);
        }
        return p;
    }

    /**
     * @param data token sequences
     * @return the perplexity of every token after the first
     */
    public double perplexity(int[][] data) {
        double nll = 0;
        int count = 0;
        for (int[] seq : data) {
            for (int i = 1; i < seq.length; i++) {
                nll -= Math.log(probability(Arrays.copyOf(seq, i), seq[i]));
                count++;
            }
        }
        return Math.exp(nll / Math.max(1, count));
    }

    private static final class Tracked implements State {
        int[][] tokens;

        Tracked(int batch) {
            tokens = new int[batch][0];
        }

        @Override
        public int length() {
            return tokens[0].length;
        }

        @Override
        public State select(int[] rows) {
            int[][] t = new int[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                t[i] = tokens[rows[i]].clone();
            }
            tokens = t;
            return this;
        }

        @Override
        public State truncate(int length) {
            for (int b = 0; b < tokens.length; b++) {
                tokens[b] = Arrays.copyOf(tokens[b], length);
            }
            return this;
        }
    }

    @Override
    public int vocabSize() {
        return vocab;
    }

    @Override
    public State newState(int batch) {
        return new Tracked(batch);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Tracked t = (Tracked) state;
        float[] out = new float[tokens.length * vocab];
        for (int b = 0; b < tokens.length; b++) {
            int[] n = Arrays.copyOf(t.tokens[b], t.tokens[b].length + tokens[b].length);
            System.arraycopy(tokens[b], 0, n, t.tokens[b].length, tokens[b].length);
            t.tokens[b] = n;
            double[] p = distribution(n);
            for (int v = 0; v < vocab; v++) {
                out[b * vocab + v] = (float) Math.log(p[v]);
            }
        }
        return Tensor.of(Shape.of(tokens.length, vocab), out);
    }

    @Override
    public String toString() {
        return "KneserNeyLm(order " + order + ", " + counts.get(order).size() + " " + order + "-grams)";
    }
}
