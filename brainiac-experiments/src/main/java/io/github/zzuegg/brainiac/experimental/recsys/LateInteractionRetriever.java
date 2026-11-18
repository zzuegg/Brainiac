package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Late interaction, ColBERT (Khattab and Zaharia, 2020, <a href="https://arxiv.org/abs/2004.12832">
 * arXiv:2004.12832</a>): queries and documents keep one small unit vector per token, and relevance
 * is MaxSim, {@code sum_i max_j q_i . d_j}: every query token finds its best match in the
 * document. This is more expressive than one vector per document (two query terms can be matched
 * by two different document tokens), yet documents are still encoded offline.
 *
 * <p>{@link #index} stores the document token vectors either in full precision or compressed as in
 * ColBERTv2 (Santhanam et al., 2022, <a href="https://arxiv.org/abs/2112.01488">arXiv:2112.01488</a>):
 * the id of the nearest of a set of k-means centroids plus a 2-bit code per dimension for the
 * residual. Candidate documents for a query come from a {@link KnnIndex} over all token vectors;
 * the candidates are then re-scored exactly by MaxSim.
 */
public final class LateInteractionRetriever {

    private final TokenEncoder encoder;
    private final Var projection;
    private final int dim;
    private float temperature = 0.1f;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float[][][] store;
    private KnnIndex tokenIndex;
    private int[] tokenDoc;
    private double bytesPerDocument;

    /**
     * Creates the retriever.
     *
     * @param vocabulary the vocabulary size, including padding token 0
     * @param maxLength the longest query or document
     * @param width the encoder width
     * @param dim the per-token vector size
     * @param heads attention heads
     * @param layers transformer blocks
     * @param seed the seed
     */
    public LateInteractionRetriever(int vocabulary, int maxLength, int width, int dim, int heads, int layers,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.encoder = new TokenEncoder(vocabulary, maxLength, width, heads, layers, r);
        this.projection = RecsysOps.matrix(r, width, dim);
        this.dim = dim;
    }

    /**
     * @param o the optimizer
     * @return this retriever
     */
    public LateInteractionRetriever optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /**
     * Per-token unit vectors; padding tokens get zero vectors.
     *
     * @param tokens {@code [batch][length]}
     * @param training whether dropout-style layers train
     * @return {@code [batch, length, dim]}
     */
    public Var tokenVectors(int[][] tokens, boolean training) {
        int b = tokens.length;
        int len = tokens[0].length;
        Var h = encoder.encode(tokens, training).reshape(b * len, encoder.dim()).matmul(projection);
        return RecsysOps.normalize(h).reshape(b, len, dim).mul(TokenEncoder.mask(tokens));
    }

    /**
     * MaxSim scores of every query against every document.
     *
     * @param q {@code [bq, lq, dim]}
     * @param d {@code [bd, ld, dim]}
     * @return {@code [bq, bd]}
     */
    public static Var maxSim(Var q, Var d) {
        int bq = q.shape().dim(0);
        int lq = q.shape().dim(1);
        int bd = d.shape().dim(0);
        int ld = d.shape().dim(1);
        int k = q.shape().dim(2);
        Var s = q.reshape(bq * lq, k).matmul(d.reshape(bd * ld, k).transpose()).reshape(bq, lq, bd, ld);
        return s.max(3, false).sum(1, false);
    }

    /**
     * One InfoNCE step over MaxSim scores with in-batch (and optional hard) negatives.
     *
     * @param queries {@code [batch][length]}
     * @param positives {@code [batch][length]}
     * @param hardNegatives {@code [batch][length]}, or {@code null}
     * @return the loss
     */
    public double trainStep(int[][] queries, int[][] positives, int[][] hardNegatives) {
        Var q = tokenVectors(queries, true);
        Var d = tokenVectors(positives, true);
        if (hardNegatives != null) {
            d = Var.concat(0, d, tokenVectors(hardNegatives, true));
        }
        Var loss = Functional.crossEntropy(maxSim(q, d).div(temperature), Tensor.arange(queries.length));
        loss.backward();
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.add(projection);
        optimizer.step(p);
        return loss.value().item();
    }

    /**
     * Encodes and stores documents.
     *
     * @param documents {@code [n][length]} token ids
     * @param centroids k-means centroids for compression, or 0 to keep full-precision vectors
     * @param seed the k-means seed
     */
    public void index(int[][] documents, int centroids, long seed) {
        float[] all = Var.noGrad(() -> tokenVectors(documents, false)).value().toArray();
        int len = documents[0].length;
        store = new float[documents.length][][];
        List<float[]> vectors = new ArrayList<>();
        for (int i = 0; i < documents.length; i++) {
            List<float[]> doc = new ArrayList<>();
            for (int j = 0; j < len; j++) {
                if (documents[i][j] != 0) {
                    float[] v = Arrays.copyOfRange(all, (i * len + j) * dim, (i * len + j + 1) * dim);
                    doc.add(v);
                    vectors.add(v);
                }
            }
            store[i] = doc.toArray(float[][]::new);
        }
        double tokensPerDoc = vectors.size() / (double) documents.length;
        if (centroids > 0) {
            compress(vectors, centroids, seed);
            bytesPerDocument = tokensPerDoc * (Math.ceil(Math.log(centroids) / Math.log(2) / 8) + dim * 2 / 8.0);
        } else {
            bytesPerDocument = tokensPerDoc * dim * 4;
        }
        tokenIndex = KnnIndex.create(dim, 1, KnnIndex.Metric.DOT);
        tokenDoc = new int[vectors.size()];
        int row = 0;
        for (int i = 0; i < store.length; i++) {
            for (float[] v : store[i]) {
                tokenIndex.add(v, new float[] {i});
                tokenDoc[row++] = i;
            }
        }
    }

    /** Replaces every stored vector by centroid + 2-bit-per-dimension residual (ColBERTv2). */
    private void compress(List<float[]> vectors, int centroids, long seed) {
        double[][] x = new double[vectors.size()][dim];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < dim; j++) {
                x[i][j] = vectors.get(i)[j];
            }
        }
        KMeans km = KMeans.fit(x, Math.min(centroids, x.length), KMeans.Init.PLUS_PLUS, 1, 20, seed);
        double[][] residual = new double[x.length][dim];
        double[] sorted = new double[x.length * dim];
        for (int i = 0; i < x.length; i++) {
            double[] c = km.centres()[km.labels()[i]];
            for (int j = 0; j < dim; j++) {
                residual[i][j] = x[i][j] - c[j];
                sorted[i * dim + j] = residual[i][j];
            }
        }
        Arrays.sort(sorted);
        double[] cut = {sorted[sorted.length / 4], sorted[sorted.length / 2], sorted[3 * sorted.length / 4]};
        double[] level = new double[4];
        int[] count = new int[4];
        for (double v : sorted) {
            int bucket = bucket(v, cut);
            level[bucket] += v;
            count[bucket]++;
        }
        for (int q = 0; q < 4; q++) {
            level[q] = count[q] == 0 ? 0 : level[q] / count[q];
        }
        for (int i = 0; i < x.length; i++) {
            double[] c = km.centres()[km.labels()[i]];
            float[] v = vectors.get(i);
            for (int j = 0; j < dim; j++) {
                v[j] = (float) (c[j] + level[bucket(residual[i][j], cut)]);
            }
        }
    }

    private static int bucket(double v, double[] cut) {
        return v < cut[0] ? 0 : v < cut[1] ? 1 : v < cut[2] ? 2 : 3;
    }

    /** @return the bytes of stored token vectors per document of the last {@link #index} */
    public double bytesPerDocument() {
        return bytesPerDocument;
    }

    private float[] queryVectors(int[] query) {
        return Var.noGrad(() -> tokenVectors(new int[][] {query}, false)).value().toArray();
    }

    private float score(float[] q, int[] query, int doc) {
        float s = 0;
        for (int i = 0; i < query.length; i++) {
            if (query[i] == 0) {
                continue;
            }
            float best = Float.NEGATIVE_INFINITY;
            for (float[] d : store[doc]) {
                float dot = 0;
                for (int j = 0; j < dim; j++) {
                    dot += q[i * dim + j] * d[j];
                }
                best = Math.max(best, dot);
            }
            s += best;
        }
        return s;
    }

    /**
     * Exhaustive MaxSim over the stored documents.
     *
     * @param query the query's tokens
     * @return one score per document
     */
    public float[] scores(int[] query) {
        float[] q = queryVectors(query);
        float[] out = new float[store.length];
        for (int d = 0; d < store.length; d++) {
            out[d] = score(q, query, d);
        }
        return out;
    }

    /**
     * Candidate generation then exact re-scoring: each query token retrieves its {@code probe}
     * nearest document tokens; their documents are scored by MaxSim.
     *
     * @param query the query's tokens
     * @param k how many documents
     * @param probe document tokens retrieved per query token
     * @return up to {@code k} documents, best first
     */
    public int[] search(int[] query, int k, int probe) {
        float[] q = queryVectors(query);
        Set<Integer> candidates = new HashSet<>();
        for (int i = 0; i < query.length; i++) {
            if (query[i] != 0) {
                for (KnnIndex.Hit h : tokenIndex.search(Arrays.copyOfRange(q, i * dim, (i + 1) * dim), probe)) {
                    candidates.add(tokenDoc[h.row()]);
                }
            }
        }
        float[] s = new float[store.length];
        Arrays.fill(s, Float.NEGATIVE_INFINITY);
        for (int d : candidates) {
            s[d] = score(q, query, d);
        }
        return RankingMetrics.topK(s, k, null);
    }
}
