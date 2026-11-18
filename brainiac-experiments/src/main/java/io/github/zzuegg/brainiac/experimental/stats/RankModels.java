package io.github.zzuegg.brainiac.experimental.stats;

import java.util.List;

/**
 * Rankings from comparisons and rank aggregation; every method returns a score per item
 * (higher = better).
 *
 * <ul>
 *   <li>{@link Method#BRADLEY_TERRY_MM}: the Bradley-Terry MLE {@code P(i beats j) =
 *       g_i / (g_i + g_j)} by Hunter's MM iteration (Annals 2004), with a pseudo-comparison
 *       against a unit-strength reference so items without wins or losses stay finite.</li>
 *   <li>{@link Method#RANK_CENTRALITY} (Negahban, Oh and Shah, Operations Research 2017,
 *       <a href="https://arxiv.org/abs/1209.1688">arXiv:1209.1688</a>): the stationary
 *       distribution of a random walk that moves from each item to the items that beat it.</li>
 *   <li>{@link Method#PLACKETT_LUCE}: the Plackett-Luce MLE on full or partial (top-k) rankings by
 *       Hunter's MM; pairs are rankings of length two.</li>
 *   <li>{@link Method#BORDA}: Borda counts; {@link Method#KEMENY_LOCAL_SEARCH}: adjacent swaps
 *       from the Borda order while they lower the summed Kendall distance to the inputs (a local
 *       optimum of the Kemeny consensus).</li>
 * </ul>
 */
public final class RankModels {

    /** The method. */
    public enum Method {
        /** Bradley-Terry by MM. */
        BRADLEY_TERRY_MM,
        /** Rank Centrality. */
        RANK_CENTRALITY,
        /** Plackett-Luce by MM. */
        PLACKETT_LUCE,
        /** Borda count. */
        BORDA,
        /** Kemeny consensus by local search. */
        KEMENY_LOCAL_SEARCH
    }

    private RankModels() {
    }

    /**
     * Scores items from pairwise outcomes.
     *
     * @param method the method (Borda and Kemeny treat each pair as a ranking)
     * @param items the number of items
     * @param games {@code [winner, loser]} pairs
     * @return a score per item
     */
    public static double[] fromPairs(Method method, int items, List<int[]> games) {
        return switch (method) {
            case BRADLEY_TERRY_MM -> bradleyTerry(items, games, 500);
            case RANK_CENTRALITY -> rankCentrality(items, games);
            case PLACKETT_LUCE, BORDA, KEMENY_LOCAL_SEARCH -> fromRankings(method, items, games);
        };
    }

    /**
     * Scores items from (possibly partial) rankings, best first.
     *
     * @param method {@link Method#PLACKETT_LUCE}, {@link Method#BORDA} or
     *     {@link Method#KEMENY_LOCAL_SEARCH}
     * @param items the number of items
     * @param rankings the rankings
     * @return a score per item
     */
    public static double[] fromRankings(Method method, int items, List<int[]> rankings) {
        return switch (method) {
            case PLACKETT_LUCE -> plackettLuce(items, rankings, 300);
            case BORDA -> borda(items, rankings);
            case KEMENY_LOCAL_SEARCH -> kemeny(items, rankings);
            default -> throw new IllegalArgumentException("pairwise-only method: " + method);
        };
    }

    /**
     * The win rate of every item (0.5 for items that never played).
     *
     * @param items the number of items
     * @param games {@code [winner, loser]} pairs
     * @return the win rates
     */
    public static double[] winRate(int items, List<int[]> games) {
        double[] w = new double[items];
        double[] n = new double[items];
        for (int[] g : games) {
            w[g[0]]++;
            n[g[0]]++;
            n[g[1]]++;
        }
        double[] r = new double[items];
        for (int i = 0; i < items; i++) {
            r[i] = n[i] == 0 ? 0.5 : w[i] / n[i];
        }
        return r;
    }

    private static double[] bradleyTerry(int items, List<int[]> games, int iterations) {
        double[] wins = new double[items];
        double[][] n = new double[items][items];
        for (int[] g : games) {
            wins[g[0]]++;
            n[g[0]][g[1]]++;
            n[g[1]][g[0]]++;
        }
        double[] gam = new double[items];
        java.util.Arrays.fill(gam, 1);
        for (int it = 0; it < iterations; it++) {
            double change = 0;
            double[] next = new double[items];
            for (int i = 0; i < items; i++) {
                double den = 1 / (gam[i] + 1); // the pseudo-comparison with the reference
                for (int j = 0; j < items; j++) {
                    if (n[i][j] > 0) {
                        den += n[i][j] / (gam[i] + gam[j]);
                    }
                }
                next[i] = (wins[i] + 0.5) / den;
            }
            double geo = 0;
            for (double v : next) {
                geo += Math.log(v) / items;
            }
            for (int i = 0; i < items; i++) {
                next[i] /= Math.exp(geo);
                change = Math.max(change, Math.abs(Math.log(next[i] / gam[i])));
            }
            gam = next;
            if (change < 1e-9) {
                break;
            }
        }
        double[] s = new double[items];
        for (int i = 0; i < items; i++) {
            s[i] = Math.log(gam[i]);
        }
        return s;
    }

    private static double[] rankCentrality(int items, List<int[]> games) {
        double[][] beat = new double[items][items]; // beat[i][j]: j beat i
        double[][] n = new double[items][items];
        for (int[] g : games) {
            beat[g[1]][g[0]]++;
            n[g[0]][g[1]]++;
            n[g[1]][g[0]]++;
        }
        int dmax = 1;
        for (int i = 0; i < items; i++) {
            int d = 0;
            for (int j = 0; j < items; j++) {
                if (n[i][j] > 0) {
                    d++;
                }
            }
            dmax = Math.max(dmax, d);
        }
        double[][] p = new double[items][items];
        for (int i = 0; i < items; i++) {
            double out = 0;
            for (int j = 0; j < items; j++) {
                if (n[i][j] > 0) {
                    // regularised share of comparisons j won against i
                    p[i][j] = (beat[i][j] + 0.5) / (n[i][j] + 1) / dmax;
                    out += p[i][j];
                }
            }
            p[i][i] = 1 - out;
        }
        double[] pi = new double[items];
        java.util.Arrays.fill(pi, 1.0 / items);
        for (int it = 0; it < 5000; it++) {
            double[] next = new double[items];
            for (int i = 0; i < items; i++) {
                if (pi[i] == 0) {
                    continue;
                }
                for (int j = 0; j < items; j++) {
                    if (p[i][j] != 0) {
                        next[j] += pi[i] * p[i][j];
                    }
                }
            }
            double change = 0;
            for (int i = 0; i < items; i++) {
                change += Math.abs(next[i] - pi[i]);
            }
            pi = next;
            if (change < 1e-12) {
                break;
            }
        }
        double[] s = new double[items];
        for (int i = 0; i < items; i++) {
            s[i] = Math.log(Math.max(pi[i], 1e-300));
        }
        return s;
    }

    private static double[] plackettLuce(int items, List<int[]> rankings, int iterations) {
        double[] wins = new double[items]; // times an item is chosen at a stage (not last)
        for (int[] r : rankings) {
            for (int k = 0; k < r.length - 1; k++) {
                wins[r[k]]++;
            }
        }
        double[] g = new double[items];
        java.util.Arrays.fill(g, 1);
        for (int it = 0; it < iterations; it++) {
            double[] den = new double[items];
            for (int i = 0; i < items; i++) {
                den[i] = 1 / (g[i] + 1); // pseudo-comparison prior
            }
            for (int[] r : rankings) {
                double tail = 0;
                double[] suffix = new double[r.length];
                for (int k = r.length - 1; k >= 0; k--) {
                    tail += g[r[k]];
                    suffix[k] = tail;
                }
                double acc = 0;
                for (int k = 0; k < r.length; k++) {
                    if (k < r.length - 1) {
                        acc += 1 / suffix[k];
                    }
                    den[r[k]] += acc;
                }
            }
            double change = 0;
            double geo = 0;
            double[] next = new double[items];
            for (int i = 0; i < items; i++) {
                next[i] = (wins[i] + 0.5) / den[i];
                geo += Math.log(next[i]) / items;
            }
            for (int i = 0; i < items; i++) {
                next[i] /= Math.exp(geo);
                change = Math.max(change, Math.abs(Math.log(next[i] / g[i])));
            }
            g = next;
            if (change < 1e-9) {
                break;
            }
        }
        double[] s = new double[items];
        for (int i = 0; i < items; i++) {
            s[i] = Math.log(g[i]);
        }
        return s;
    }

    private static double[] borda(int items, List<int[]> rankings) {
        double[] s = new double[items];
        for (int[] r : rankings) {
            for (int k = 0; k < r.length; k++) {
                s[r[k]] += (r.length - 1 - k) / (double) Math.max(1, r.length - 1);
            }
        }
        return s;
    }

    private static double[] kemeny(int items, List<int[]> rankings) {
        // pairwise preference counts: pref[a][b] = rankings placing a above b
        double[][] pref = new double[items][items];
        for (int[] r : rankings) {
            for (int a = 0; a < r.length; a++) {
                for (int b = a + 1; b < r.length; b++) {
                    pref[r[a]][r[b]]++;
                }
            }
        }
        double[] bs = borda(items, rankings);
        Integer[] order = new Integer[items];
        for (int i = 0; i < items; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(bs[b], bs[a]));
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int k = 0; k + 1 < items; k++) {
                int a = order[k];
                int b = order[k + 1];
                // swapping changes the disagreement count by pref[a][b] - pref[b][a]
                if (pref[b][a] > pref[a][b]) {
                    order[k] = b;
                    order[k + 1] = a;
                    improved = true;
                }
            }
        }
        double[] s = new double[items];
        for (int k = 0; k < items; k++) {
            s[order[k]] = items - k;
        }
        return s;
    }
}
