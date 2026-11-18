package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Unsupervised community detection on undirected graphs given as neighbour lists.
 *
 * <ul>
 *   <li>{@link Method#LOUVAIN} (Blondel et al., J. Stat. Mech. 2008,
 *       <a href="https://arxiv.org/abs/0803.0476">arXiv:0803.0476</a>): greedy local moves of single
 *       nodes to the neighbouring community with the largest modularity gain, then contraction of
 *       the communities into nodes, repeated while modularity grows. It picks the number of
 *       communities itself.</li>
 *   <li>{@link Method#SPECTRAL_NON_BACKTRACKING}: spectral clustering of the Bethe Hessian
 *       {@code H(r) = (r^2 - 1) I - r A + D} with {@code r = sqrt(sum d^2 / sum d - 1)}, the
 *       symmetric stand-in for the non-backtracking operator (Saade, Krzakala and Zdeborova,
 *       NeurIPS 2014, <a href="https://arxiv.org/abs/1406.1880">arXiv:1406.1880</a>): its
 *       {@code k} smallest eigenvectors are clustered by k-means. Unlike the adjacency or the
 *       Laplacian it is not swamped by high-degree nodes on sparse graphs and works down to the
 *       detectability threshold.</li>
 *   <li>{@link Method#SBM_BP} (Decelle, Krzakala, Moore and Zdeborova, PRE 2011,
 *       <a href="https://arxiv.org/abs/1109.3041">arXiv:1109.3041</a>): belief propagation for the
 *       stochastic block model with the external-field approximation for non-edges, with the
 *       group sizes and affinities learnt by EM from several starting contrasts, the restart with
 *       the lowest Bethe free energy winning. Messages start tilted towards the Bethe-Hessian
 *       partition.</li>
 * </ul>
 */
public final class CommunityDetection {

    /** The method. */
    public enum Method {
        /** Louvain modularity maximisation */
        LOUVAIN,
        /** k-means on the Bethe Hessian's smallest eigenvectors */
        SPECTRAL_NON_BACKTRACKING,
        /** belief propagation on the stochastic block model with EM */
        SBM_BP
    }

    private CommunityDetection() {
    }

    /**
     * Neighbour lists of a dense 0/1 adjacency.
     *
     * @param adjacency {@code n x n}, row-major, symmetric
     * @param n the number of nodes
     * @return sorted neighbours of each node (no self-loops)
     */
    public static int[][] neighbours(float[] adjacency, int n) {
        int[][] nb = new int[n][];
        for (int i = 0; i < n; i++) {
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && adjacency[i * n + j] != 0) {
                    c++;
                }
            }
            nb[i] = new int[c];
            c = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && adjacency[i * n + j] != 0) {
                    nb[i][c++] = j;
                }
            }
        }
        return nb;
    }

    /**
     * Detects communities.
     *
     * @param method the method
     * @param nb sorted neighbour lists
     * @param k the number of communities (ignored by Louvain)
     * @param seed the seed
     * @return the community of each node
     */
    public static int[] detect(Method method, int[][] nb, int k, long seed) {
        return switch (method) {
            case LOUVAIN -> louvain(nb, seed);
            case SPECTRAL_NON_BACKTRACKING -> betheHessian(nb, k, seed);
            case SBM_BP -> beliefPropagation(nb, k, betheHessian(nb, k, seed), seed);
        };
    }

    /**
     * The Newman-Girvan modularity of a partition.
     *
     * @param nb neighbour lists
     * @param labels the partition
     * @return {@code Q}
     */
    public static double modularity(int[][] nb, int[] labels) {
        int c = Arrays.stream(labels).max().orElse(0) + 1;
        double[] in = new double[c];
        double[] tot = new double[c];
        double m2 = 0;
        for (int i = 0; i < nb.length; i++) {
            tot[labels[i]] += nb[i].length;
            m2 += nb[i].length;
            for (int j : nb[i]) {
                if (labels[j] == labels[i]) {
                    in[labels[i]]++;
                }
            }
        }
        double q = 0;
        for (int a = 0; a < c; a++) {
            q += in[a] / m2 - (tot[a] / m2) * (tot[a] / m2);
        }
        return q;
    }

    // ------------------------------------------------------------------ Louvain

    private static int[] louvain(int[][] nb, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = nb.length;
        List<Map<Integer, Double>> a = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Map<Integer, Double> row = new HashMap<>();
            for (int j : nb[i]) {
                row.put(j, 1.0);
            }
            a.add(row);
        }
        int[] membership = new int[n];
        for (int i = 0; i < n; i++) {
            membership[i] = i;
        }
        while (true) {
            int size = a.size();
            double[] deg = new double[size];
            double m2 = 0;
            for (int i = 0; i < size; i++) {
                for (double v : a.get(i).values()) {
                    deg[i] += v;
                }
                m2 += deg[i];
            }
            if (m2 == 0) {
                break;
            }
            int[] comm = new int[size];
            double[] tot = new double[size];
            for (int i = 0; i < size; i++) {
                comm[i] = i;
                tot[i] = deg[i];
            }
            boolean moved = true;
            boolean anyMove = false;
            int[] order = new int[size];
            for (int i = 0; i < size; i++) {
                order[i] = i;
            }
            for (int pass = 0; pass < 100 && moved; pass++) {
                moved = false;
                for (int i = size - 1; i > 0; i--) {
                    int j = r.nextInt(i + 1);
                    int t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
                for (int i : order) {
                    Map<Integer, Double> links = new HashMap<>();
                    for (Map.Entry<Integer, Double> e : a.get(i).entrySet()) {
                        if (e.getKey() != i) {
                            links.merge(comm[e.getKey()], e.getValue(), Double::sum);
                        }
                    }
                    int c0 = comm[i];
                    tot[c0] -= deg[i];
                    int best = c0;
                    double bestGain = links.getOrDefault(c0, 0.0) - tot[c0] * deg[i] / m2;
                    for (Map.Entry<Integer, Double> e : links.entrySet()) {
                        double gain = e.getValue() - tot[e.getKey()] * deg[i] / m2;
                        if (gain > bestGain + 1e-12) {
                            bestGain = gain;
                            best = e.getKey();
                        }
                    }
                    comm[i] = best;
                    tot[best] += deg[i];
                    if (best != c0) {
                        moved = true;
                        anyMove = true;
                    }
                }
            }
            if (!anyMove) {
                break;
            }
            int[] id = new int[size];
            Arrays.fill(id, -1);
            int next = 0;
            for (int i = 0; i < size; i++) {
                if (id[comm[i]] < 0) {
                    id[comm[i]] = next++;
                }
            }
            List<Map<Integer, Double>> b = new ArrayList<>();
            for (int c = 0; c < next; c++) {
                b.add(new HashMap<>());
            }
            for (int i = 0; i < size; i++) {
                for (Map.Entry<Integer, Double> e : a.get(i).entrySet()) {
                    b.get(id[comm[i]]).merge(id[comm[e.getKey()]], e.getValue(), Double::sum);
                }
            }
            for (int v = 0; v < n; v++) {
                membership[v] = id[comm[membership[v]]];
            }
            a = b;
            if (next == size) {
                break;
            }
        }
        return membership;
    }

    // ------------------------------------------------------------------ Bethe Hessian

    private static int[] betheHessian(int[][] nb, int k, long seed) {
        int n = nb.length;
        double sd = 0;
        double sd2 = 0;
        for (int[] row : nb) {
            sd += row.length;
            sd2 += (double) row.length * row.length;
        }
        double rr = Math.sqrt(Math.max(1.0, sd2 / Math.max(sd, 1) - 1));
        double[][] h = new double[n][n];
        for (int i = 0; i < n; i++) {
            h[i][i] = rr * rr - 1 + nb[i].length;
            for (int j : nb[i]) {
                h[i][j] = -rr;
            }
        }
        MlMatrices.Eigen e = MlMatrices.eigh(h);
        double[][] u = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                u[i][c] = e.vectors()[i][n - 1 - c];
            }
        }
        return KMeans.fit(u, k, KMeans.Init.PLUS_PLUS, 10, 100, seed).labels();
    }

    // ------------------------------------------------------------------ belief propagation

    /** Contrasts {@code c_out / c_in} tried by {@link #beliefPropagation}. */
    private static final double[] CONTRASTS = {0.05, 0.1, 0.2, 0.3, 0.5};
    /** EM rounds (BP, then affinities from the edge marginals) per starting contrast. */
    private static final int EM_ROUNDS = 6;

    /**
     * SBM belief propagation (what {@link Method#SBM_BP} runs on the Bethe-Hessian partition).
     * The parameters are unknown: from each starting contrast {@code c_out / c_in} on a grid
     * (equal group sizes, the observed mean degree), BP with messages tilted towards
     * {@code init} alternates with EM updates of the group sizes and affinities from the node
     * and edge marginals, and the end point with the lowest Bethe free energy is kept, as
     * Decelle et al. choose among EM restarts.
     *
     * @param nb sorted neighbour lists
     * @param k the number of groups
     * @param init the initial partition, labels in {@code [0, k)}
     * @param seed the seed of the update order
     * @return the argmax of the BP marginals
     */
    public static int[] beliefPropagation(int[][] nb, int k, int[] init, long seed) {
        int n = nb.length;
        int[] start = new int[n + 1];
        for (int i = 0; i < n; i++) {
            start[i + 1] = start[i] + nb[i].length;
        }
        int edges = start[n];
        // rev[e] for e = (i -> nb[i][t]) is the index of (nb[i][t] -> i)
        int[] rev = new int[edges];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < nb[i].length; t++) {
                int j = nb[i][t];
                rev[start[i] + t] = start[j] + Arrays.binarySearch(nb[j], i);
            }
        }
        double mean = (double) edges / n;
        double bestF = Double.POSITIVE_INFINITY;
        double[][] best = null;
        for (double eps : CONTRASTS) {
            double cIn = k * mean / (1 + (k - 1) * eps);
            double[][] c = new double[k][k];
            for (int a = 0; a < k; a++) {
                for (int b = 0; b < k; b++) {
                    c[a][b] = a == b ? cIn : eps * cIn;
                }
            }
            double[][] marg = new double[n][k];
            double[][] msg = new double[edges][];
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < k; a++) {
                    marg[i][a] = a == init[i] ? 0.6 : 0.4 / (k - 1);
                }
                for (int t = start[i]; t < start[i + 1]; t++) {
                    msg[t] = marg[i].clone();
                }
            }
            SplittableRandom r = new SplittableRandom(seed);
            double[] na = new double[k];
            Arrays.fill(na, 1.0 / k);
            double f = 0;
            for (int em = 0; em < EM_ROUNDS; em++) {
                f = run(nb, k, start, rev, c, na, marg, msg, 30, r);
                if (em < EM_ROUNDS - 1) {
                    reestimate(nb, k, start, rev, c, na, marg, msg);
                }
            }
            if (f < bestF) {
                bestF = f;
                best = marg;
            }
        }
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            int b = 0;
            for (int a = 1; a < k; a++) {
                if (best[i][a] > best[i][b]) {
                    b = a;
                }
            }
            labels[i] = b;
        }
        return labels;
    }

    /** The M step: {@code n_a} from the marginals, {@code c_ab = sum_(i->j) P_ij(a, b) / (N n_a n_b)}. */
    private static void reestimate(int[][] nb, int k, int[] start, int[] rev, double[][] c, double[] na,
            double[][] marg, double[][] msg) {
        int n = nb.length;
        Arrays.fill(na, 0);
        for (double[] m : marg) {
            for (int a = 0; a < k; a++) {
                na[a] += m[a] / n;
            }
        }
        for (int a = 0; a < k; a++) {
            na[a] = Math.max(na[a], 1e-3);
        }
        double[][] cnt = new double[k][k];
        double[][] pair = new double[k][k];
        for (int i = 0; i < n; i++) {
            for (int t = start[i]; t < start[i + 1]; t++) {
                double[] pi = msg[t];
                double[] pj = msg[rev[t]];
                double z = 0;
                for (int a = 0; a < k; a++) {
                    for (int b = 0; b < k; b++) {
                        pair[a][b] = c[a][b] * pi[a] * pj[b];
                        z += pair[a][b];
                    }
                }
                for (int a = 0; a < k; a++) {
                    for (int b = 0; b < k; b++) {
                        cnt[a][b] += pair[a][b] / Math.max(z, 1e-300);
                    }
                }
            }
        }
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                c[a][b] = Math.max(1e-3, cnt[a][b] / (n * na[a] * na[b]));
            }
        }
    }

    /** BP sweeps with fixed parameters; returns the Bethe free energy per node (up to a constant). */
    private static double run(int[][] nb, int k, int[] start, int[] rev, double[][] c, double[] na,
            double[][] marg, double[][] msg, int sweeps, SplittableRandom r) {
        int n = nb.length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        double[] field = new double[k];
        double[] logTot = new double[k];
        double logZnodes = 0;
        for (int sweep = 0; sweep <= sweeps; sweep++) {
            boolean last = sweep == sweeps;
            Arrays.fill(field, 0);
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < k; a++) {
                    for (int b = 0; b < k; b++) {
                        field[a] += c[a][b] * marg[i][b] / n;
                    }
                }
            }
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            logZnodes = 0;
            for (int i : order) {
                int deg = nb[i].length;
                double[][] logIn = new double[deg][k];
                for (int a = 0; a < k; a++) {
                    logTot[a] = Math.log(na[a]) - field[a];
                }
                for (int t = 0; t < deg; t++) {
                    double[] in = msg[rev[start[i] + t]];
                    for (int a = 0; a < k; a++) {
                        double s = 0;
                        for (int b = 0; b < k; b++) {
                            s += c[a][b] * in[b];
                        }
                        logIn[t][a] = Math.log(Math.max(s, 1e-300));
                        logTot[a] += logIn[t][a];
                    }
                }
                double mx = Double.NEGATIVE_INFINITY;
                double[] m = new double[k];
                for (int a = 0; a < k; a++) {
                    m[a] = logTot[a];
                    mx = Math.max(mx, m[a]);
                }
                double z = 0;
                for (int a = 0; a < k; a++) {
                    z += Math.exp(m[a] - mx);
                }
                logZnodes += mx + Math.log(z);
                if (last) {
                    continue;
                }
                for (int t = 0; t < deg; t++) {
                    double[] out = new double[k];
                    double mo = Double.NEGATIVE_INFINITY;
                    for (int a = 0; a < k; a++) {
                        out[a] = logTot[a] - logIn[t][a];
                        mo = Math.max(mo, out[a]);
                    }
                    normaliseExp(out, mo);
                    double[] old = msg[start[i] + t];
                    for (int a = 0; a < k; a++) {
                        old[a] = 0.5 * old[a] + 0.5 * out[a];
                    }
                }
                normaliseExp(m, mx);
                marg[i] = m;
            }
        }
        // f = -(1/N) sum_i log Z_i + (1/N) sum_(ij) log Z_ij + (1/2) sum_ab c_ab n_a n_b
        double logZedges = 0;
        for (int i = 0; i < n; i++) {
            for (int t = start[i]; t < start[i + 1]; t++) {
                if (nb[i][t - start[i]] < i) {
                    continue;
                }
                double[] pi = msg[t];
                double[] pj = msg[rev[t]];
                double z = 0;
                for (int a = 0; a < k; a++) {
                    for (int b = 0; b < k; b++) {
                        z += c[a][b] * pi[a] * pj[b];
                    }
                }
                logZedges += Math.log(Math.max(z, 1e-300));
            }
        }
        double cBar = 0;
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                cBar += c[a][b] * na[a] * na[b];
            }
        }
        return (-logZnodes + logZedges) / n + cBar / 2;
    }

    private static void normaliseExp(double[] v, double mx) {
        double s = 0;
        for (int a = 0; a < v.length; a++) {
            v[a] = Math.exp(v[a] - mx);
            s += v[a];
        }
        for (int a = 0; a < v.length; a++) {
            v[a] /= s;
        }
    }
}
