package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Topic models of bag-of-words corpora: a set of topics (distributions over words) and each
 * document's mixture of them.
 * <ul>
 *   <li>{@link #lda}: latent Dirichlet allocation (Blei, Ng and Jordan 2003,
 *       <a href="https://jmlr.org/papers/v3/blei03a.html">JMLR 3</a>) fitted by collapsed Gibbs
 *       sampling (Griffiths and Steyvers 2004): every token's topic is resampled from
 *       {@code (n_dk + alpha) (n_kw + beta) / (n_k + V beta)} with the topic proportions and
 *       topics integrated out.</li>
 *   <li>{@link #nmf}: non-negative matrix factorisation of the document-word counts by the
 *       multiplicative updates of Lee and Seung (1999,
 *       <a href="https://www.nature.com/articles/44565">Nature 401</a>) for the generalised KL
 *       divergence, whose rows become topics once normalised.</li>
 * </ul>
 * {@link #perplexity} scores any set of topics on held-out documents the same way: each
 * document's mixture is fitted by EM with the topics fixed.
 *
 * @param topics the topics {@code [K][V]}, each row a distribution over words
 * @param mixtures the training documents' topic proportions {@code [D][K]}
 */
public record TopicModel(double[][] topics, double[][] mixtures) {

    /**
     * Fits LDA by collapsed Gibbs sampling.
     *
     * @param docs documents as word ids
     * @param vocabulary the vocabulary size
     * @param k the number of topics
     * @param alpha the document-topic Dirichlet prior, for example {@code 0.1}
     * @param beta the topic-word Dirichlet prior, for example {@code 0.01}
     * @param sweeps Gibbs sweeps over all tokens
     * @param seed the seed
     * @return the posterior-mean topics and mixtures of the last sweep
     */
    public static TopicModel lda(int[][] docs, int vocabulary, int k, double alpha, double beta, int sweeps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = docs.length;
        int[][] z = new int[d][];
        int[][] ndk = new int[d][k];
        int[][] nkw = new int[k][vocabulary];
        int[] nk = new int[k];
        for (int i = 0; i < d; i++) {
            z[i] = new int[docs[i].length];
            for (int t = 0; t < docs[i].length; t++) {
                int topic = r.nextInt(k);
                z[i][t] = topic;
                ndk[i][topic]++;
                nkw[topic][docs[i][t]]++;
                nk[topic]++;
            }
        }
        double[] p = new double[k];
        double vb = vocabulary * beta;
        for (int s = 0; s < sweeps; s++) {
            for (int i = 0; i < d; i++) {
                for (int t = 0; t < docs[i].length; t++) {
                    int w = docs[i][t];
                    int old = z[i][t];
                    ndk[i][old]--;
                    nkw[old][w]--;
                    nk[old]--;
                    double total = 0;
                    for (int c = 0; c < k; c++) {
                        total += (ndk[i][c] + alpha) * (nkw[c][w] + beta) / (nk[c] + vb);
                        p[c] = total;
                    }
                    double u = r.nextDouble() * total;
                    int topic = 0;
                    while (topic < k - 1 && p[topic] < u) {
                        topic++;
                    }
                    z[i][t] = topic;
                    ndk[i][topic]++;
                    nkw[topic][w]++;
                    nk[topic]++;
                }
            }
        }
        double[][] phi = new double[k][vocabulary];
        for (int c = 0; c < k; c++) {
            for (int w = 0; w < vocabulary; w++) {
                phi[c][w] = (nkw[c][w] + beta) / (nk[c] + vb);
            }
        }
        double[][] theta = new double[d][k];
        for (int i = 0; i < d; i++) {
            for (int c = 0; c < k; c++) {
                theta[i][c] = (ndk[i][c] + alpha) / (docs[i].length + k * alpha);
            }
        }
        return new TopicModel(phi, theta);
    }

    /**
     * Fits NMF with KL-divergence multiplicative updates.
     *
     * @param docs documents as word ids
     * @param vocabulary the vocabulary size
     * @param k the number of topics
     * @param iterations update rounds
     * @param seed the seed
     * @return the normalised topics and mixtures
     */
    public static TopicModel nmf(int[][] docs, int vocabulary, int k, int iterations, long seed) {
        int d = docs.length;
        double[][] x = counts(docs, vocabulary);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] w = new double[d][k];
        double[][] h = new double[k][vocabulary];
        for (double[] row : w) {
            for (int c = 0; c < k; c++) {
                row[c] = 0.5 + r.nextDouble();
            }
        }
        for (double[] row : h) {
            for (int v = 0; v < vocabulary; v++) {
                row[v] = 0.5 + r.nextDouble();
            }
        }
        double[][] ratio = new double[d][vocabulary];
        for (int it = 0; it < iterations; it++) {
            fillRatio(x, w, h, ratio);
            // H <- H * (W^T (X / WH)) / (W^T 1)
            for (int c = 0; c < k; c++) {
                double colSum = 0;
                for (int i = 0; i < d; i++) {
                    colSum += w[i][c];
                }
                for (int v = 0; v < vocabulary; v++) {
                    double num = 0;
                    for (int i = 0; i < d; i++) {
                        num += w[i][c] * ratio[i][v];
                    }
                    h[c][v] *= num / Math.max(colSum, 1e-300);
                }
            }
            fillRatio(x, w, h, ratio);
            // W <- W * ((X / WH) H^T) / (1 H^T)
            for (int c = 0; c < k; c++) {
                double rowSum = 0;
                for (int v = 0; v < vocabulary; v++) {
                    rowSum += h[c][v];
                }
                for (int i = 0; i < d; i++) {
                    double num = 0;
                    for (int v = 0; v < vocabulary; v++) {
                        num += ratio[i][v] * h[c][v];
                    }
                    w[i][c] *= num / Math.max(rowSum, 1e-300);
                }
            }
        }
        double[][] phi = new double[k][];
        double[] scale = new double[k];
        for (int c = 0; c < k; c++) {
            phi[c] = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) {
                scale[c] += h[c][v];
            }
            for (int v = 0; v < vocabulary; v++) {
                phi[c][v] = (h[c][v] + 1e-12) / (scale[c] + vocabulary * 1e-12);
            }
        }
        double[][] theta = new double[d][k];
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int c = 0; c < k; c++) {
                theta[i][c] = w[i][c] * scale[c];
                s += theta[i][c];
            }
            for (int c = 0; c < k; c++) {
                theta[i][c] /= Math.max(s, 1e-300);
            }
        }
        return new TopicModel(phi, theta);
    }

    private static void fillRatio(double[][] x, double[][] w, double[][] h, double[][] ratio) {
        int k = h.length;
        for (int i = 0; i < x.length; i++) {
            for (int v = 0; v < x[i].length; v++) {
                if (x[i][v] == 0) {
                    ratio[i][v] = 0;
                    continue;
                }
                double s = 0;
                for (int c = 0; c < k; c++) {
                    s += w[i][c] * h[c][v];
                }
                ratio[i][v] = x[i][v] / Math.max(s, 1e-300);
            }
        }
    }

    /**
     * Document-word count rows.
     *
     * @param docs documents as word ids
     * @param vocabulary the vocabulary size
     * @return {@code [D][V]}
     */
    public static double[][] counts(int[][] docs, int vocabulary) {
        double[][] x = new double[docs.length][vocabulary];
        for (int i = 0; i < docs.length; i++) {
            for (int w : docs[i]) {
                x[i][w]++;
            }
        }
        return x;
    }

    /**
     * Held-out perplexity: each document's topic mixture is fitted by EM with the topics fixed
     * (fold-in), then {@code exp(-sum log p(w) / tokens)}.
     *
     * @param docs held-out documents
     * @param iterations EM iterations per document
     * @return the perplexity, lower is better
     */
    public double perplexity(int[][] docs, int iterations) {
        int k = topics.length;
        double ll = 0;
        long tokens = 0;
        for (int[] doc : docs) {
            double[] theta = new double[k];
            java.util.Arrays.fill(theta, 1.0 / k);
            double[] resp = new double[k];
            for (int it = 0; it < iterations; it++) {
                double[] next = new double[k];
                for (int w : doc) {
                    double s = 0;
                    for (int c = 0; c < k; c++) {
                        resp[c] = theta[c] * topics[c][w];
                        s += resp[c];
                    }
                    for (int c = 0; c < k; c++) {
                        next[c] += resp[c] / Math.max(s, 1e-300);
                    }
                }
                for (int c = 0; c < k; c++) {
                    theta[c] = (next[c] + 0.01) / (doc.length + 0.01 * k);
                }
            }
            for (int w : doc) {
                double p = 0;
                for (int c = 0; c < k; c++) {
                    p += theta[c] * topics[c][w];
                }
                ll += Math.log(Math.max(p, 1e-300));
                tokens++;
            }
        }
        return Math.exp(-ll / tokens);
    }

    /**
     * How well planted topics are recovered: for every true topic, the best cosine similarity
     * with any fitted topic, averaged.
     *
     * @param truth the true topics {@code [K][V]}
     * @return the mean best cosine in {@code [0, 1]}
     */
    public double recovery(double[][] truth) {
        double total = 0;
        for (double[] t : truth) {
            double best = 0;
            for (double[] f : topics) {
                best = Math.max(best, MlMatrices.dot(t, f)
                        / Math.sqrt(MlMatrices.dot(t, t) * MlMatrices.dot(f, f) + 1e-300));
            }
            total += best / truth.length;
        }
        return total;
    }
}
