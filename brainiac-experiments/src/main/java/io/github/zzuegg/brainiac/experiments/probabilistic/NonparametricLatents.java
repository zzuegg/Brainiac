package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.DirichletProcessMixture;
import io.github.zzuegg.brainiac.experimental.ml.HierarchicalDirichletProcess;
import io.github.zzuegg.brainiac.experimental.ml.IndianBuffetProcess;
import io.github.zzuegg.brainiac.experimental.ml.TopicModel;
import io.github.zzuegg.brainiac.experiments.classic.TopicRecovery;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Bayesian nonparametric latent structure (wave 12, item 12.5.16).
 *
 * <ul>
 *   <li>{@link #latentFeatures}: the "Cambridge" toy of Griffiths and Ghahramani, 6x6 images that
 *       each sum a random subset of four hidden 3x3 patterns plus noise of standard deviation 0.5.
 *       The IBP linear-Gaussian model against a Dirichlet-process mixture (one cluster per pattern
 *       combination) and NMF with the right {@code K = 4}; scored by the number of
 *       features/clusters, the RMSE of the reconstruction against the noise-free images and the
 *       mean best cosine between true and recovered patterns.</li>
 *   <li>{@link #topics}: a planted corpus ({@link TopicRecovery#corpus}) with eight topics and
 *       sparse document mixtures; the HDP against LDA at {@code K = 4, 8, 16}, scored by the
 *       number of topics, held-out fold-in perplexity and topic recovery.</li>
 * </ul>
 */
public final class NonparametricLatents {

    private NonparametricLatents() {
    }

    /**
     * The four hidden 6x6 patterns, one 3x3 shape per quadrant.
     *
     * @return {@code [4][36]}
     */
    static double[][] cambridgePatterns() {
        String[] shapes = {
            "###" + "#.#" + "###",
            ".#." + "###" + ".#.",
            "#.." + "#.." + "###",
            "#.#" + ".#." + "#.#"};
        double[][] p = new double[4][36];
        for (int f = 0; f < 4; f++) {
            int r0 = f / 2 * 3;
            int c0 = f % 2 * 3;
            for (int i = 0; i < 9; i++) {
                if (shapes[f].charAt(i) == '#') {
                    p[f][(r0 + i / 3) * 6 + c0 + i % 3] = 1;
                }
            }
        }
        return p;
    }

    /**
     * IBP against a DP mixture and NMF on the Cambridge toy.
     *
     * @param n images
     * @param sweeps Gibbs sweeps for the IBP and the DP mixture
     * @param seed the seed
     * @return method to {features or clusters, reconstruction RMSE against clean images, mean best
     *     cosine to the true patterns (NaN for the DP mixture)}
     */
    public static Map<String, double[]> latentFeatures(int n, int sweeps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] patterns = cambridgePatterns();
        double[][] clean = new double[n][36];
        double[][] x = new double[n][36];
        for (int i = 0; i < n; i++) {
            for (int f = 0; f < 4; f++) {
                if (r.nextDouble() < 0.5) {
                    for (int j = 0; j < 36; j++) {
                        clean[i][j] += patterns[f][j];
                    }
                }
            }
            for (int j = 0; j < 36; j++) {
                x[i][j] = clean[i][j] + 0.5 * r.nextGaussian();
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        IndianBuffetProcess.Result ibp = IndianBuffetProcess.gibbs(x, 0.5, 1, 1, sweeps, seed);
        out.put("IBP linear-Gaussian", new double[] {ibp.features(), rmse(ibp.reconstruction(), clean),
            bestCosine(patterns, ibp.loadings())});
        DirichletProcessMixture.Result dp = DirichletProcessMixture.gibbs(x, 1,
                DirichletProcessMixture.Prior.of(x, 0.25), sweeps, null, seed);
        double[][] dpRecon = new double[n][];
        for (int i = 0; i < n; i++) {
            dpRecon[i] = dp.means()[dp.labels()[i]];
        }
        out.put("DP mixture", new double[] {dp.clusters(), rmse(dpRecon, clean), Double.NaN});
        double[][][] wh = nmf(x, 4, 300, seed);
        double[][] nmfRecon = new double[n][36];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 4; k++) {
                for (int j = 0; j < 36; j++) {
                    nmfRecon[i][j] += wh[0][i][k] * wh[1][k][j];
                }
            }
        }
        out.put("NMF K=4", new double[] {4, rmse(nmfRecon, clean), bestCosine(patterns, wh[1])});
        return out;
    }

    /**
     * HDP against LDA at three fixed {@code K} on a planted eight-topic corpus.
     *
     * @param docs training documents
     * @param sweeps Gibbs sweeps
     * @param seed the seed
     * @return model to {topics, held-out perplexity, recovery (mean best cosine to the true topics)}
     */
    public static Map<String, double[]> topics(int docs, int sweeps, long seed) {
        int k = 8;
        int vocabulary = 100;
        TopicRecovery.Corpus corpus = TopicRecovery.corpus(k, vocabulary, docs, 40, 0.2, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        TopicModel truth = new TopicModel(corpus.topics(), new double[0][]);
        out.put("true topics", new double[] {k, truth.perplexity(corpus.test(), 30), 1});
        HierarchicalDirichletProcess.Result hdp = HierarchicalDirichletProcess.gibbs(corpus.train(), vocabulary, 1, 1,
                0.05, sweeps, seed);
        TopicModel hm = hdp.model();
        out.put("HDP", new double[] {hdp.topics().length, hm.perplexity(corpus.test(), 30),
            hm.recovery(corpus.topics())});
        for (int kk : new int[] {4, 8, 16}) {
            TopicModel lda = TopicModel.lda(corpus.train(), vocabulary, kk, 0.1, 0.05, sweeps, seed);
            out.put("LDA K=" + kk, new double[] {kk, lda.perplexity(corpus.test(), 30), lda.recovery(corpus.topics())});
        }
        return out;
    }

    private static double rmse(double[][] a, double[][] b) {
        double s = 0;
        int c = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                s += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
                c++;
            }
        }
        return Math.sqrt(s / c);
    }

    private static double bestCosine(double[][] truth, double[][] found) {
        double total = 0;
        for (double[] t : truth) {
            double best = 0;
            for (double[] f : found) {
                double dot = 0;
                double nt = 0;
                double nf = 0;
                for (int j = 0; j < t.length; j++) {
                    dot += t[j] * f[j];
                    nt += t[j] * t[j];
                    nf += f[j] * f[j];
                }
                best = Math.max(best, dot / Math.sqrt(nt * nf + 1e-300));
            }
            total += best / truth.length;
        }
        return total;
    }

    /** Multiplicative-update NMF (Lee and Seung) on the positive part of {@code x}. */
    private static double[][][] nmf(double[][] x, int k, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        double[][] v = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                v[i][j] = Math.max(x[i][j], 0);
            }
        }
        double[][] w = new double[n][k];
        double[][] h = new double[k][d];
        for (double[] row : w) {
            for (int c = 0; c < k; c++) {
                row[c] = 0.1 + r.nextDouble();
            }
        }
        for (double[] row : h) {
            for (int j = 0; j < d; j++) {
                row[j] = 0.1 + r.nextDouble();
            }
        }
        for (int it = 0; it < iterations; it++) {
            double[][] wh = multiply(w, h);
            for (int c = 0; c < k; c++) {
                for (int j = 0; j < d; j++) {
                    double num = 0;
                    double den = 0;
                    for (int i = 0; i < n; i++) {
                        num += w[i][c] * v[i][j];
                        den += w[i][c] * wh[i][j];
                    }
                    h[c][j] *= num / (den + 1e-12);
                }
            }
            wh = multiply(w, h);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < k; c++) {
                    double num = 0;
                    double den = 0;
                    for (int j = 0; j < d; j++) {
                        num += v[i][j] * h[c][j];
                        den += wh[i][j] * h[c][j];
                    }
                    w[i][c] *= num / (den + 1e-12);
                }
            }
        }
        return new double[][][] {w, h};
    }

    private static double[][] multiply(double[][] a, double[][] b) {
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < b.length; k++) {
                double v = a[i][k];
                for (int j = 0; j < b[0].length; j++) {
                    c[i][j] += v * b[k][j];
                }
            }
        }
        return c;
    }
}
