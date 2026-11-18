package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The hierarchical Dirichlet process topic model (Teh, Jordan, Beal and Blei, JASA 2006): every
 * document draws its topic mixture from {@code DP(alpha, beta)} around a shared global measure
 * {@code beta ~ GEM(gamma)}, so all documents share one countable set of topics and the corpus
 * chooses how many it uses.
 *
 * <p>Inference is the direct-assignment Gibbs sampler (section 5.3 of the paper) with topics
 * {@code Dirichlet(eta)} over words integrated out. Each token takes an existing topic with
 * probability {@code ∝ (n_jk + alpha beta_k) (n_kw + eta) / (n_k + V eta)} or a new one with
 * {@code ∝ alpha beta_u / V}; a new topic splits the unassigned mass {@code beta_u} by a
 * {@code Beta(1, gamma)} stick. After each sweep the table counts {@code m_jk} are drawn from
 * their Antoniak distribution and {@code beta ~ Dir(m_.1, ..., m_.K, gamma)}; empty topics are
 * dropped.
 */
public final class HierarchicalDirichletProcess {

    /**
     * The final sample.
     *
     * @param topics the posterior-mean topics {@code [K][V]}
     * @param mixtures the posterior-mean document mixtures {@code [D][K]}
     * @param topicTrace the number of topics after each sweep
     */
    public record Result(double[][] topics, double[][] mixtures, int[] topicTrace) {

        /**
         * As a {@link TopicModel}, for perplexity and recovery.
         *
         * @return the topic model
         */
        public TopicModel model() {
            return new TopicModel(topics, mixtures);
        }
    }

    private HierarchicalDirichletProcess() {
    }

    /**
     * Runs the direct-assignment sampler from one topic.
     *
     * @param docs documents as word ids
     * @param vocabulary the vocabulary size {@code V}
     * @param alpha the document-level concentration
     * @param gamma the corpus-level concentration
     * @param eta the topic-word Dirichlet parameter
     * @param sweeps Gibbs sweeps
     * @param seed the seed
     * @return the final sample
     */
    public static Result gibbs(int[][] docs, int vocabulary, double alpha, double gamma, double eta, int sweeps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int nd = docs.length;
        int[][] z = new int[nd][];
        List<int[]> nkw = new ArrayList<>();
        List<int[]> njk = new ArrayList<>();
        List<Integer> nk = new ArrayList<>();
        // per-topic document counts kept as columns: njk.get(k)[j]
        nkw.add(new int[vocabulary]);
        njk.add(new int[nd]);
        nk.add(0);
        for (int j = 0; j < nd; j++) {
            z[j] = new int[docs[j].length];
            for (int t = 0; t < docs[j].length; t++) {
                nkw.get(0)[docs[j][t]]++;
                njk.get(0)[j]++;
                nk.set(0, nk.get(0) + 1);
            }
        }
        List<Double> beta = new ArrayList<>();
        beta.add(0.5);
        double betaU = 0.5;
        double veta = vocabulary * eta;
        int[] trace = new int[sweeps];
        double[] p = new double[16];
        for (int s = 0; s < sweeps; s++) {
            for (int j = 0; j < nd; j++) {
                for (int t = 0; t < docs[j].length; t++) {
                    int w = docs[j][t];
                    int old = z[j][t];
                    nkw.get(old)[w]--;
                    njk.get(old)[j]--;
                    nk.set(old, nk.get(old) - 1);
                    int k = nk.size();
                    if (p.length < k + 1) {
                        p = new double[2 * (k + 1)];
                    }
                    double total = 0;
                    for (int c = 0; c < k; c++) {
                        double v = (njk.get(c)[j] + alpha * beta.get(c)) * (nkw.get(c)[w] + eta) / (nk.get(c) + veta);
                        total += v;
                        p[c] = total;
                    }
                    total += alpha * betaU / vocabulary;
                    p[k] = total;
                    double u = r.nextDouble() * total;
                    int pick = 0;
                    while (pick < k && p[pick] < u) {
                        pick++;
                    }
                    if (pick == k) {
                        nkw.add(new int[vocabulary]);
                        njk.add(new int[nd]);
                        nk.add(0);
                        double b = betaDraw(1, gamma, r);
                        beta.add(b * betaU);
                        betaU *= 1 - b;
                    }
                    z[j][t] = pick;
                    nkw.get(pick)[w]++;
                    njk.get(pick)[j]++;
                    nk.set(pick, nk.get(pick) + 1);
                }
            }
            // drop empty topics and relabel
            int k = nk.size();
            int[] map = new int[k];
            List<int[]> nkw2 = new ArrayList<>();
            List<int[]> njk2 = new ArrayList<>();
            List<Integer> nk2 = new ArrayList<>();
            List<Double> beta2 = new ArrayList<>();
            for (int c = 0; c < k; c++) {
                if (nk.get(c) > 0) {
                    map[c] = nk2.size();
                    nkw2.add(nkw.get(c));
                    njk2.add(njk.get(c));
                    nk2.add(nk.get(c));
                    beta2.add(beta.get(c));
                } else {
                    map[c] = -1;
                    betaU += beta.get(c);
                }
            }
            for (int j = 0; j < nd; j++) {
                for (int t = 0; t < z[j].length; t++) {
                    z[j][t] = map[z[j][t]];
                }
            }
            nkw = nkw2;
            njk = njk2;
            nk = nk2;
            beta = beta2;
            k = nk.size();
            // table counts by the Antoniak distribution, then beta ~ Dir(m_.k, gamma)
            double[] m = new double[k + 1];
            for (int c = 0; c < k; c++) {
                double ab = alpha * beta.get(c);
                int[] col = njk.get(c);
                for (int j = 0; j < nd; j++) {
                    for (int l = 0; l < col[j]; l++) {
                        if (r.nextDouble() < ab / (ab + l)) {
                            m[c]++;
                        }
                    }
                }
            }
            m[k] = gamma;
            double[] g = new double[k + 1];
            double sum = 0;
            for (int c = 0; c <= k; c++) {
                g[c] = m[c] > 0 ? IndianBuffetProcess.gamma(m[c], r) : 0;
                sum += g[c];
            }
            for (int c = 0; c < k; c++) {
                beta.set(c, g[c] / sum);
            }
            betaU = g[k] / sum;
            trace[s] = k;
        }
        int k = nk.size();
        double[][] topics = new double[k][vocabulary];
        for (int c = 0; c < k; c++) {
            for (int w = 0; w < vocabulary; w++) {
                topics[c][w] = (nkw.get(c)[w] + eta) / (nk.get(c) + veta);
            }
        }
        double[][] mix = new double[nd][k];
        for (int j = 0; j < nd; j++) {
            double tot = 0;
            for (int c = 0; c < k; c++) {
                mix[j][c] = njk.get(c)[j] + alpha * beta.get(c);
                tot += mix[j][c];
            }
            for (int c = 0; c < k; c++) {
                mix[j][c] /= tot;
            }
        }
        return new Result(topics, mix, trace);
    }

    private static double betaDraw(double a, double b, SplittableRandom r) {
        double x = IndianBuffetProcess.gamma(a, r);
        double y = IndianBuffetProcess.gamma(b, r);
        return x / (x + y);
    }
}
