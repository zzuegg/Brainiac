package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.TopicModel;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Topic models on a synthetic corpus with planted topics: LDA by collapsed Gibbs sampling and
 * NMF ({@link TopicModel}) against k-means on bag-of-words rows (each centroid read as a topic).
 * Every document mixes several topics (Dirichlet proportions), which a hard clustering cannot
 * express. Reported: topic recovery (mean best cosine with the planted topics) and held-out
 * perplexity with fold-in mixtures.
 */
public final class TopicRecovery {

    private TopicRecovery() {
    }

    /**
     * A planted corpus.
     *
     * @param topics the true topics {@code [K][V]}
     * @param train training documents
     * @param test held-out documents
     */
    public record Corpus(double[][] topics, int[][] train, int[][] test) {
    }

    /**
     * Generates documents from {@code k} topics, each putting most of its mass on its own block
     * of 10 words and a little everywhere.
     *
     * @param k the number of topics
     * @param vocabulary the vocabulary size, at least {@code 10 k}
     * @param docs training documents
     * @param length words per document
     * @param alpha the Dirichlet concentration of the mixtures
     * @param seed the seed
     * @return the corpus, with {@code docs / 3} held-out documents
     */
    public static Corpus corpus(int k, int vocabulary, int docs, int length, double alpha, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] topics = new double[k][vocabulary];
        for (int c = 0; c < k; c++) {
            double s = 0;
            for (int w = 0; w < vocabulary; w++) {
                boolean own = w >= c * 10 && w < c * 10 + 10;
                topics[c][w] = own ? 1 + r.nextDouble() * 2 : 0.02;
                s += topics[c][w];
            }
            for (int w = 0; w < vocabulary; w++) {
                topics[c][w] /= s;
            }
        }
        return new Corpus(topics, documents(topics, docs, length, alpha, r),
                documents(topics, docs / 3, length, alpha, r));
    }

    private static int[][] documents(double[][] topics, int n, int length, double alpha, SplittableRandom r) {
        int k = topics.length;
        int[][] docs = new int[n][length];
        for (int i = 0; i < n; i++) {
            double[] theta = dirichlet(k, alpha, r);
            for (int t = 0; t < length; t++) {
                int c = sample(theta, r);
                docs[i][t] = sample(topics[c], r);
            }
        }
        return docs;
    }

    private static int sample(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        for (int i = 0; i < p.length - 1; i++) {
            u -= p[i];
            if (u < 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    private static double[] dirichlet(int k, double alpha, SplittableRandom r) {
        double[] g = new double[k];
        double s = 0;
        for (int i = 0; i < k; i++) {
            g[i] = gamma(alpha, r);
            s += g[i];
        }
        for (int i = 0; i < k; i++) {
            g[i] /= s;
        }
        return g;
    }

    /** Marsaglia-Tsang gamma variates, boosted for shapes below 1. */
    private static double gamma(double shape, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = r.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    /**
     * Fits the three models.
     *
     * @param corpus the corpus
     * @param sweeps LDA Gibbs sweeps
     * @param seed the seed
     * @return method to {topic recovery, held-out perplexity}; "true topics" gives the floor
     */
    public static Map<String, double[]> compare(Corpus corpus, int sweeps, long seed) {
        int k = corpus.topics().length;
        int v = corpus.topics()[0].length;
        Map<String, double[]> out = new LinkedHashMap<>();
        TopicModel truth = new TopicModel(corpus.topics(), new double[0][]);
        out.put("true topics", new double[] {1, truth.perplexity(corpus.test(), 30)});
        TopicModel lda = TopicModel.lda(corpus.train(), v, k, 0.1, 0.01, sweeps, seed);
        out.put("LDA (Gibbs)", new double[] {lda.recovery(corpus.topics()), lda.perplexity(corpus.test(), 30)});
        TopicModel nmf = TopicModel.nmf(corpus.train(), v, k, 150, seed);
        out.put("NMF", new double[] {nmf.recovery(corpus.topics()), nmf.perplexity(corpus.test(), 30)});
        double[][] rows = TopicModel.counts(corpus.train(), v);
        for (double[] row : rows) {
            double s = 0;
            for (double x : row) {
                s += x;
            }
            for (int w = 0; w < v; w++) {
                row[w] /= s;
            }
        }
        KMeans km = KMeans.fit(rows, k, KMeans.Init.PLUS_PLUS, 5, 100, seed);
        double[][] centroidTopics = new double[k][v];
        for (int c = 0; c < k; c++) {
            double s = 0;
            for (int w = 0; w < v; w++) {
                centroidTopics[c][w] = km.centres()[c][w] + 1e-3;
                s += centroidTopics[c][w];
            }
            for (int w = 0; w < v; w++) {
                centroidTopics[c][w] /= s;
            }
        }
        TopicModel kmeans = new TopicModel(centroidTopics, new double[0][]);
        out.put("k-means on BoW", new double[] {kmeans.recovery(corpus.topics()),
            kmeans.perplexity(corpus.test(), 30)});
        return out;
    }
}
