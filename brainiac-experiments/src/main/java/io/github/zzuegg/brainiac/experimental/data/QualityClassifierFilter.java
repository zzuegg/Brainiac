package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Model-based quality filtering, as in DCLM (Li et al. 2024, "DataComp-LM",
 * <a href="https://arxiv.org/abs/2406.11794">arXiv:2406.11794</a>) and FineWeb-Edu: a small
 * classifier learns to tell a seed set of high-quality documents (DCLM: instruction data
 * and ELI5 answers; FineWeb-Edu: LLM-graded educational pages) from random documents of
 * the raw pool, and only the top-scoring share of the pool is kept for pretraining. DCLM
 * found such a fastText bigram classifier beat every heuristic and perplexity filter.
 *
 * <p>This is that classifier for token documents: hashed n-gram counts (orders 1 to
 * {@code order}, normalised by length) into {@code buckets} features, then logistic
 * regression trained with Adam. {@link #filter} keeps the top fraction of a pool.
 */
public final class QualityClassifierFilter {

    private final int buckets;
    private final int order;
    private final Var weight;
    private final Var bias;

    private QualityClassifierFilter(int buckets, int order, long seed) {
        if (buckets < 1 || order < 1) {
            throw new IllegalArgumentException("need buckets >= 1 and order >= 1");
        }
        this.buckets = buckets;
        this.order = order;
        this.weight = Var.param(Tensor.normal(Shape.of(buckets, 1), 0, 0.01f, new SplittableRandom(seed)));
        this.bias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    /**
     * Trains a classifier.
     *
     * @param positives the high-quality seed documents
     * @param negatives documents drawn from the raw pool
     * @param buckets the hashed feature count, for example 4096
     * @param order the longest n-gram, 2 for fastText's bigrams
     * @param epochs full-batch Adam steps
     * @param seed the seed
     * @return the trained classifier
     */
    public static QualityClassifierFilter train(List<int[]> positives, List<int[]> negatives, int buckets, int order,
            int epochs, long seed) {
        QualityClassifierFilter f = new QualityClassifierFilter(buckets, order, seed);
        List<int[]> docs = new ArrayList<>(positives);
        docs.addAll(negatives);
        float[] labels = new float[docs.size()];
        Arrays.fill(labels, 0, positives.size(), 1);
        Tensor x = f.features(docs);
        Tensor y = Tensor.of(Shape.of(docs.size(), 1), labels);
        Optimizer opt = Optimizer.adam(0.05f);
        List<Var> params = List.of(f.weight, f.bias);
        for (int e = 0; e < epochs; e++) {
            Var z = Var.of(x).matmul(f.weight).add(f.bias);
            // logistic loss: softplus(z) - y z, plus a little L2
            z.softplus().sub(z.mul(y)).mean().add(f.weight.square().sum().mul(1e-4f)).backward();
            opt.step(params);
        }
        return f;
    }

    private int hash(int[] doc, int end, int n) {
        long h = 0x165667B19E3779F9L * n;
        for (int t = end - n + 1; t <= end; t++) {
            h = (h ^ doc[t]) * 0x9E3779B97F4A7C15L;
            h ^= h >>> 31;
        }
        return (int) Math.floorMod(h, (long) buckets);
    }

    /**
     * The hashed n-gram features of documents.
     *
     * @param docs the documents
     * @return {@code [docs, buckets]}
     */
    public Tensor features(List<int[]> docs) {
        float[] f = new float[docs.size() * buckets];
        for (int d = 0; d < docs.size(); d++) {
            int[] doc = docs.get(d);
            float w = 1f / Math.max(1, doc.length);
            for (int t = 0; t < doc.length; t++) {
                for (int n = 1; n <= Math.min(order, t + 1); n++) {
                    f[d * buckets + hash(doc, t, n)] += w;
                }
            }
        }
        return Tensor.of(Shape.of(docs.size(), buckets), f);
    }

    /**
     * @param docs the documents
     * @return the classifier's logit of "high quality" per document
     */
    public double[] scores(List<int[]> docs) {
        float[] z = features(docs).matmul(weight.value()).add(bias.value()).toArray();
        double[] out = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            out[i] = z[i];
        }
        return out;
    }

    /**
     * The top-scoring share of a pool.
     *
     * @param pool the documents
     * @param keepFraction the share kept
     * @return the kept documents, in pool order
     */
    public List<int[]> filter(List<int[]> pool, double keepFraction) {
        return keepTop(pool, scores(pool), keepFraction);
    }

    /**
     * The documents with the highest scores.
     *
     * @param pool the documents
     * @param scores a score per document, higher is better
     * @param keepFraction the share kept
     * @return the kept documents, in pool order
     */
    public static List<int[]> keepTop(List<int[]> pool, double[] scores, double keepFraction) {
        Integer[] order = new Integer[pool.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        int keep = Math.max(1, (int) Math.round(keepFraction * pool.size()));
        boolean[] kept = new boolean[pool.size()];
        for (int i = 0; i < keep; i++) {
            kept[order[i]] = true;
        }
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < pool.size(); i++) {
            if (kept[i]) {
                out.add(pool.get(i));
            }
        }
        return out;
    }
}
