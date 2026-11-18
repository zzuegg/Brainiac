package io.github.zzuegg.brainiac.experimental.recsys;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An inverted index over token-id documents with three scoring modes.
 * <ul>
 *   <li>{@link #bm25}: Okapi BM25 (Robertson and Zaragoza, 2009, "The Probabilistic Relevance
 *       Framework: BM25 and Beyond"): {@code sum_t idf(t) tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))}
 *       over the query terms, with term-frequency saturation {@code k1} and length normalisation
 *       {@code b};</li>
 *   <li>{@link #tfIdf}: cosine similarity of TF-IDF vectors, the classic baseline;</li>
 *   <li>{@link #impact}: an impact index for learned sparse representations (SPLADE,
 *       {@link SparseLexicalEncoder}): documents are added with a weight per term and a query
 *       scores {@code sum_t w_q(t) w_d(t)}.</li>
 * </ul>
 * Scoring walks only the posting lists of the query's terms. Token 0 is padding and is ignored.
 */
public final class Bm25Index {

    private enum Mode {
        BM25, TF_IDF, IMPACT
    }

    private record Posting(int doc, float weight) {
    }

    private final Mode mode;
    private final double k1;
    private final double b;
    private final Map<Integer, List<Posting>> postings = new HashMap<>();
    private final List<Integer> lengths = new ArrayList<>();
    private final List<Map<Integer, Integer>> counts = new ArrayList<>();
    private long totalLength;
    private double[] norms;

    private Bm25Index(Mode mode, double k1, double b) {
        this.mode = mode;
        this.k1 = k1;
        this.b = b;
    }

    /**
     * @param k1 term-frequency saturation, such as 1.2
     * @param b length normalisation in {@code [0, 1]}, such as 0.75
     * @return an empty BM25 index
     */
    public static Bm25Index bm25(double k1, double b) {
        return new Bm25Index(Mode.BM25, k1, b);
    }

    /** @return an empty TF-IDF cosine index */
    public static Bm25Index tfIdf() {
        return new Bm25Index(Mode.TF_IDF, 0, 0);
    }

    /** @return an empty impact index for weighted terms */
    public static Bm25Index impact() {
        return new Bm25Index(Mode.IMPACT, 0, 0);
    }

    /**
     * Adds a document of tokens (BM25 and TF-IDF modes).
     *
     * @param tokens the document's tokens
     * @return its id
     */
    public int add(int[] tokens) {
        if (mode == Mode.IMPACT) {
            throw new IllegalStateException("an impact index takes weighted terms");
        }
        int id = lengths.size();
        Map<Integer, Integer> tf = new HashMap<>();
        int len = 0;
        for (int t : tokens) {
            if (t != 0) {
                tf.merge(t, 1, Integer::sum);
                len++;
            }
        }
        tf.forEach((t, c) -> postings.computeIfAbsent(t, k -> new ArrayList<>()).add(new Posting(id, c)));
        lengths.add(len);
        counts.add(tf);
        totalLength += len;
        norms = null;
        return id;
    }

    /**
     * Adds a document of weighted terms (impact mode).
     *
     * @param terms term ids
     * @param weights their weights
     * @return its id
     */
    public int add(int[] terms, float[] weights) {
        if (mode != Mode.IMPACT) {
            throw new IllegalStateException("weighted terms need an impact index");
        }
        int id = lengths.size();
        for (int i = 0; i < terms.length; i++) {
            if (weights[i] != 0) {
                postings.computeIfAbsent(terms[i], k -> new ArrayList<>()).add(new Posting(id, weights[i]));
            }
        }
        lengths.add(terms.length);
        counts.add(Map.of());
        return id;
    }

    /** @return the number of documents */
    public int size() {
        return lengths.size();
    }

    /**
     * @param term a term
     * @return its inverse document frequency, {@code log(1 + (N - n + 0.5) / (n + 0.5))}
     */
    public double idf(int term) {
        int n = postings.getOrDefault(term, List.of()).size();
        return Math.log(1 + (size() - n + 0.5) / (n + 0.5));
    }

    private void computeNorms() {
        norms = new double[size()];
        for (int d = 0; d < size(); d++) {
            double s = 0;
            for (Map.Entry<Integer, Integer> e : counts.get(d).entrySet()) {
                double w = e.getValue() * idf(e.getKey());
                s += w * w;
            }
            norms[d] = Math.sqrt(s) + 1e-12;
        }
    }

    /**
     * Scores every document for a token query (BM25 and TF-IDF modes).
     *
     * @param query the query's tokens
     * @return one score per document, 0 for documents sharing no term
     */
    public float[] scores(int[] query) {
        float[] out = new float[size()];
        Map<Integer, Integer> qtf = new HashMap<>();
        for (int t : query) {
            if (t != 0) {
                qtf.merge(t, 1, Integer::sum);
            }
        }
        if (mode == Mode.TF_IDF && norms == null) {
            computeNorms();
        }
        double avg = totalLength / (double) Math.max(1, size());
        double qnorm = 0;
        for (Map.Entry<Integer, Integer> e : qtf.entrySet()) {
            int t = e.getKey();
            double idf = idf(t);
            qnorm += Math.pow(e.getValue() * idf, 2);
            for (Posting p : postings.getOrDefault(t, List.of())) {
                double tf = p.weight();
                if (mode == Mode.BM25) {
                    out[p.doc()] += (float) (idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths.get(p.doc()) / avg)));
                } else if (mode == Mode.TF_IDF) {
                    out[p.doc()] += (float) (e.getValue() * idf * tf * idf);
                } else {
                    out[p.doc()] += e.getValue() * p.weight();
                }
            }
        }
        if (mode == Mode.TF_IDF) {
            double qn = Math.sqrt(qnorm) + 1e-12;
            for (int d = 0; d < out.length; d++) {
                out[d] /= (float) (qn * norms[d]);
            }
        }
        return out;
    }

    /**
     * Scores every document for a weighted query (impact mode).
     *
     * @param terms the query's terms
     * @param weights their weights
     * @return one score per document
     */
    public float[] scores(int[] terms, float[] weights) {
        float[] out = new float[size()];
        for (int i = 0; i < terms.length; i++) {
            for (Posting p : postings.getOrDefault(terms[i], List.of())) {
                out[p.doc()] += weights[i] * p.weight();
            }
        }
        return out;
    }

    /**
     * @param query the query's tokens
     * @param k how many
     * @return the best {@code k} documents, best first
     */
    public int[] search(int[] query, int k) {
        return RankingMetrics.topK(scores(query), k, null);
    }

    /** @return the average number of postings per document */
    public double averageTermsPerDocument() {
        long n = 0;
        for (List<Posting> p : postings.values()) {
            n += p.size();
        }
        return n / (double) Math.max(1, size());
    }

    @Override
    public String toString() {
        return mode + " index of " + size() + " documents";
    }
}
