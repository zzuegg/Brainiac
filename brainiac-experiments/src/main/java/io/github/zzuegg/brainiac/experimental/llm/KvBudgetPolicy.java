package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.experimental.nn.SnapKv;

import java.util.Arrays;

/**
 * How a compressed KV cache spends a total budget of cached entries over layers and heads:
 * the shared extension point of head-level KV compression. A policy reads a
 * {@link KvSnapshot} (keys and scoring queries per head) and returns, per layer and head,
 * which cached tokens stay. The total kept, summed over every layer and head, is at most the
 * budget, so policies are compared at equal memory.
 *
 * <ul>
 * <li>{@link #uniform}: SnapKV with the same budget for every head, the baseline.</li>
 * <li>{@link #pyramid}: PyramidKV, budgets shrinking linearly with depth, because lower
 *     layers spread attention widely and upper layers concentrate it on a few tokens.</li>
 * <li>{@link #adaptive}: Ada-KV, each layer's budget split among its heads by one top-k over
 *     all heads' scores, so concentrated heads give room to dispersed ones.</li>
 * </ul>
 *
 * <p>{@link ReconstructionKvScorer} (KVzip) and {@link DuoHeadCache} (DuoAttention) are
 * policies of this interface too. The core {@code KvCache.Strategy} is sealed and keeps one
 * set of positions for all heads; this interface is the hook head-level policies need.
 *
 * <p>Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information
 * Funneling" (2024), <a href="https://arxiv.org/abs/2406.02069">arXiv:2406.02069</a>; Feng et
 * al., "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation" (2024),
 * <a href="https://arxiv.org/abs/2407.11550">arXiv:2407.11550</a>.
 */
@FunctionalInterface
public interface KvBudgetPolicy {

    /**
     * Chooses the entries to keep.
     *
     * @param snapshot the cached keys and scoring queries
     * @param budget the most entries kept, summed over every layer, head and token
     * @return {@code [layer][head][token]} whether the entry stays
     */
    boolean[][][] select(KvSnapshot snapshot, int budget);

    /**
     * Counts the kept entries of a selection.
     *
     * @param keep a selection
     * @return the entries kept
     */
    static int kept(boolean[][][] keep) {
        int n = 0;
        for (boolean[][] layer : keep) {
            for (boolean[] head : layer) {
                for (boolean b : head) {
                    n += b ? 1 : 0;
                }
            }
        }
        return n;
    }

    /**
     * SnapKV with an equal budget per head.
     *
     * @param window the observation window (the snapshot's queries are its last tokens)
     * @param kernel the pooling width, odd
     * @return the policy
     */
    static KvBudgetPolicy uniform(int window, int kernel) {
        SnapKv snap = SnapKv.of(window, kernel);
        return (s, budget) -> {
            int per = budget / (s.layers() * s.heads());
            boolean[][][] out = new boolean[s.layers()][s.heads()][];
            for (int l = 0; l < s.layers(); l++) {
                for (int h = 0; h < s.heads(); h++) {
                    out[l][h] = snap.select(s.attention(l, h), per);
                }
            }
            return out;
        };
    }

    /**
     * PyramidKV: SnapKV selection with per-layer budgets falling linearly from the first layer
     * to the last.
     *
     * @param window the observation window
     * @param kernel the pooling width, odd
     * @param ratio the first layer's budget over the last layer's, at least 1
     * @return the policy
     */
    static KvBudgetPolicy pyramid(int window, int kernel, double ratio) {
        if (ratio < 1) {
            throw new IllegalArgumentException("ratio must be at least 1");
        }
        SnapKv snap = SnapKv.of(window, kernel);
        return (s, budget) -> {
            int layers = s.layers();
            double[] weight = new double[layers];
            double total = 0;
            for (int l = 0; l < layers; l++) {
                weight[l] = layers == 1 ? 1 : ratio + (1 - ratio) * l / (layers - 1.0);
                total += weight[l];
            }
            boolean[][][] out = new boolean[layers][s.heads()][];
            for (int l = 0; l < layers; l++) {
                int per = (int) Math.floor(budget * weight[l] / total / s.heads());
                for (int h = 0; h < s.heads(); h++) {
                    out[l][h] = snap.select(s.attention(l, h), per);
                }
            }
            return out;
        };
    }

    /**
     * Ada-KV: each layer gets an equal share; inside a layer the window tokens of every head
     * stay and the rest of the share goes to the best pooled scores over all heads together,
     * with at least {@code floor} of an even split per head (the paper's safeguard).
     *
     * @param window the observation window
     * @param kernel the pooling width, odd
     * @param floor the share of an even split every head keeps at least, in [0, 1]
     * @return the policy
     */
    static KvBudgetPolicy adaptive(int window, int kernel, double floor) {
        SnapKv snap = SnapKv.of(window, kernel);
        return (s, budget) -> {
            int layers = s.layers();
            int heads = s.heads();
            int n = s.length();
            boolean[][][] out = new boolean[layers][heads][];
            for (int l = 0; l < layers; l++) {
                int share = budget / layers;
                float[][][] votes = new float[heads][][];
                double[][] scores = new double[heads][];
                int fixed = 0;
                for (int h = 0; h < heads; h++) {
                    votes[h] = s.attention(l, h);
                    scores[h] = pooled(votes[h], kernel);
                    fixed += Math.min(n, votes[h].length);
                }
                int room = Math.max(0, share - fixed);
                int[] extra = new int[heads];
                int guaranteed = (int) Math.floor(floor * room / heads);
                int left = room;
                for (int h = 0; h < heads; h++) {
                    extra[h] = Math.min(guaranteed, scores[h].length);
                    left -= extra[h];
                }
                // the rest by one top-k over every head's remaining scores
                int total = 0;
                for (double[] sc : scores) {
                    total += sc.length;
                }
                long[] order = new long[total];
                double[] flat = new double[total];
                int at = 0;
                for (int h = 0; h < heads; h++) {
                    double[] sorted = scores[h].clone();
                    Arrays.sort(sorted);
                    for (int j = 0; j < sorted.length; j++) {
                        // rank within the head, best first; the guaranteed ones are already taken
                        int rank = sorted.length - 1 - j;
                        flat[at] = rank < extra[h] ? Double.NEGATIVE_INFINITY : sorted[j];
                        order[at] = ((long) h << 32) | at;
                        at++;
                    }
                }
                Integer[] idx = new Integer[total];
                for (int i = 0; i < total; i++) {
                    idx[i] = i;
                }
                Arrays.sort(idx, (a, b) -> Double.compare(flat[b], flat[a]));
                for (int i = 0; i < Math.min(left, total); i++) {
                    if (flat[idx[i]] == Double.NEGATIVE_INFINITY) {
                        break;
                    }
                    extra[(int) (order[idx[i]] >>> 32)]++;
                }
                for (int h = 0; h < heads; h++) {
                    out[l][h] = snap.select(votes[h], Math.min(n, votes[h].length) + extra[h]);
                }
            }
            return out;
        };
    }

    /**
     * SnapKV's score of every token before the observation window: votes summed over the
     * window's queries, max-pooled over {@code kernel} neighbouring positions.
     *
     * @param votes per window query, its attention over every token
     * @param kernel the pooling width, odd
     * @return the score of each token before the window
     */
    static double[] pooled(float[][] votes, int kernel) {
        int n = votes.length == 0 ? 0 : votes[0].length;
        int prefix = Math.max(0, n - votes.length);
        double[] score = new double[prefix];
        for (float[] v : votes) {
            for (int j = 0; j < prefix; j++) {
                score[j] += v[j];
            }
        }
        double[] out = new double[prefix];
        int r = kernel / 2;
        for (int j = 0; j < prefix; j++) {
            double m = 0;
            for (int d = Math.max(0, j - r); d <= Math.min(prefix - 1, j + r); d++) {
                m = Math.max(m, score[d]);
            }
            out[j] = m + 1e-9 * score[j];
        }
        return out;
    }
}
