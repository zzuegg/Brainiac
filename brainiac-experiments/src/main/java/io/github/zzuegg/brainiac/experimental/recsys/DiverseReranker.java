package io.github.zzuegg.brainiac.experimental.recsys;

/**
 * Diversity re-ranking of a candidate list: choose {@code k} items that are relevant and not
 * redundant.
 * <ul>
 *   <li>{@link #mmr} (Carbonell and Goldstein, SIGIR 1998,
 *       <a href="https://doi.org/10.1145/290941.291025">doi:10.1145/290941.291025</a>): maximal
 *       marginal relevance, greedily adding the item maximising
 *       {@code lambda * relevance - (1 - lambda) * max similarity to the chosen items}.</li>
 *   <li>{@link #dpp} (Chen, Zhang and Zhou, NeurIPS 2018, <a href="https://arxiv.org/abs/1709.05135">
 *       arXiv:1709.05135</a>): a determinantal point process scores a whole set by
 *       {@code det(L_S)} with the kernel {@code L = diag(q) S diag(q)} (quality times similarity),
 *       which rewards sets that are relevant and spread out. MAP inference is NP-hard; the fast
 *       greedy algorithm adds the item with the largest marginal gain in {@code log det}, kept up
 *       to date by incremental Cholesky updates in {@code O(k)} per candidate and step. The
 *       relevance/diversity trade-off uses Chen et al.'s {@code q = exp(alpha r)} with
 *       {@code alpha = theta / (2 (1 - theta))}.</li>
 * </ul>
 * Similarities come from item embeddings (cosine; the Gram matrix of unit vectors, so the DPP
 * kernel is positive semi-definite).
 */
public final class DiverseReranker {

    private DiverseReranker() {
    }

    private static double[][] unit(double[][] embeddings) {
        double[][] u = new double[embeddings.length][];
        for (int i = 0; i < u.length; i++) {
            double n = 0;
            for (double v : embeddings[i]) {
                n += v * v;
            }
            n = Math.sqrt(n) + 1e-12;
            u[i] = new double[embeddings[i].length];
            for (int j = 0; j < u[i].length; j++) {
                u[i][j] = embeddings[i][j] / n;
            }
        }
        return u;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * Maximal marginal relevance.
     *
     * @param relevance candidate relevance scores
     * @param embeddings candidate embeddings (cosine similarity)
     * @param k items to choose
     * @param lambda 1 for pure relevance, 0 for pure novelty
     * @return chosen candidate indices, in order
     */
    public static int[] mmr(double[] relevance, double[][] embeddings, int k, double lambda) {
        int n = relevance.length;
        double[][] u = unit(embeddings);
        int[] chosen = new int[Math.min(k, n)];
        boolean[] used = new boolean[n];
        double[] maxSim = new double[n];
        java.util.Arrays.fill(maxSim, Double.NEGATIVE_INFINITY);
        for (int step = 0; step < chosen.length; step++) {
            int best = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (used[i]) {
                    continue;
                }
                double penalty = step == 0 ? 0 : maxSim[i];
                double s = lambda * relevance[i] - (1 - lambda) * penalty;
                if (s > bestScore) {
                    bestScore = s;
                    best = i;
                }
            }
            chosen[step] = best;
            used[best] = true;
            for (int i = 0; i < n; i++) {
                maxSim[i] = Math.max(maxSim[i], dot(u[i], u[best]));
            }
        }
        return chosen;
    }

    /**
     * Greedy DPP MAP inference.
     *
     * @param relevance candidate relevance scores
     * @param embeddings candidate embeddings
     * @param k items to choose
     * @param theta the relevance weight in {@code [0, 1)}: 0 ignores relevance, near 1 is nearly
     *     relevance-sorted
     * @return chosen candidate indices, in order (fewer than {@code k} if the kernel's rank runs out)
     */
    public static int[] dpp(double[] relevance, double[][] embeddings, int k, double theta) {
        int n = relevance.length;
        double alpha = theta / (2 * (1 - theta));
        double max = Double.NEGATIVE_INFINITY;
        for (double r : relevance) {
            max = Math.max(max, r);
        }
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = Math.exp(alpha * (relevance[i] - max));
        }
        double[][] u = unit(embeddings);
        int steps = Math.min(k, n);
        double[][] c = new double[n][steps];
        double[] d2 = new double[n];
        for (int i = 0; i < n; i++) {
            d2[i] = q[i] * q[i] + 1e-9;
        }
        boolean[] used = new boolean[n];
        int[] chosen = new int[steps];
        int count = 0;
        for (int step = 0; step < steps; step++) {
            int j = -1;
            double best = 1e-12;
            for (int i = 0; i < n; i++) {
                if (!used[i] && d2[i] > best) {
                    best = d2[i];
                    j = i;
                }
            }
            if (j < 0) {
                break;
            }
            chosen[count++] = j;
            used[j] = true;
            double dj = Math.sqrt(d2[j]);
            for (int i = 0; i < n; i++) {
                if (used[i]) {
                    continue;
                }
                double lji = q[j] * q[i] * dot(u[j], u[i]);
                double proj = 0;
                for (int m = 0; m < step; m++) {
                    proj += c[j][m] * c[i][m];
                }
                double e = (lji - proj) / dj;
                c[i][step] = e;
                d2[i] -= e * e;
            }
        }
        return java.util.Arrays.copyOf(chosen, count);
    }
}
