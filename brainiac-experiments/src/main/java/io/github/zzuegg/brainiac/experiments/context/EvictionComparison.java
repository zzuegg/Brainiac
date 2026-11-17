package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.EvictingKvCache;
import io.github.zzuegg.brainiac.experimental.nn.LearnedEvictionScorer;
import io.github.zzuegg.brainiac.experimental.nn.SnapKv;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Experiment X7: which tokens should a full KV cache drop? A stream mixes filler tokens,
 * rare "fact" tokens (an unusual key direction) and rare "question" tokens whose queries look
 * for facts. Facts receive almost no attention until a question arrives, often long after
 * they were written, so recency (sliding window) and attention-so-far (H2O) both evict them;
 * a {@link LearnedEvictionScorer} trained on a different stream learns that unusual keys
 * will be needed and keeps them. The score is the error of the layer's outputs at question
 * steps against a cache that keeps everything. A second part compresses a prompt once with
 * {@link SnapKv}: the question at the end of the prompt votes for the facts it needs.
 */
public final class EvictionComparison {

    private static final int F = 8;

    private EvictionComparison() {
    }

    /**
     * A one-head attention layer whose queries look for facts: query dimension 0 reads input
     * dimension 1 (the question flag), keys and values are the input.
     *
     * @return the layer
     */
    static MultiHeadAttention layer() {
        MultiHeadAttention a = (MultiHeadAttention) MultiHeadAttention.of(1).headSize(F).bias(false).causal(true)
                .create(Shape.of(8, F), new SplittableRandom(1));
        float[] q = new float[F * F];
        float[] eye = new float[F * F];
        for (int i = 0; i < F; i++) {
            eye[i * F + i] = 1;
            q[i * F + i] = i < 2 ? 0 : 1;
        }
        q[F] = 2;   // input dim 1 -> query dim 0
        List<Var> p = a.parameters();
        p.get(0).assign(Tensor.of(Shape.of(F, F), q));
        p.get(1).assign(Tensor.of(Shape.of(F, F), eye));
        p.get(2).assign(Tensor.of(Shape.of(F, F), eye));
        p.get(3).assign(Tensor.of(Shape.of(F, F), eye));
        return a;
    }

    /**
     * A stream of filler, facts and questions.
     *
     * @param n the tokens
     * @param seed the seed
     * @param kinds receives 0 filler, 1 fact, 2 question per token
     * @return the token features
     */
    static List<float[]> stream(int n, long seed, int[] kinds) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> out = new ArrayList<>();
        for (int t = 0; t < n; t++) {
            float[] x = new float[F];
            for (int i = 2; i < F; i++) {
                x[i] = (float) (r.nextGaussian() * 0.5);
            }
            double u = r.nextDouble();
            int kind = t > 0 && u < 0.03 ? 1 : t > 20 && u < 0.05 ? 2 : 0;
            if (kind == 1) {
                x[0] = 3;
            } else if (kind == 2) {
                x[1] = 3;
            }
            kinds[t] = kind;
            out.add(x);
        }
        return out;
    }

    /**
     * Runs the comparison of eviction policies.
     *
     * @param budget the tokens each cache keeps
     * @param length the stream length
     * @param seed the seed of the test stream (the scorer trains on another)
     * @return the mean squared output error at question steps, per policy name
     */
    public static Map<String, Double> run(int budget, int length, long seed) {
        MultiHeadAttention attention = layer();
        int[] trainKinds = new int[length];
        LearnedEvictionScorer scorer = new LearnedEvictionScorer(16, 1).recent(budget / 4);
        scorer.fit(LearnedEvictionScorer.harvest(attention, stream(length, seed + 1000, trainKinds), 48, 6), 200);
        int[] kinds = new int[length];
        List<float[]> test = stream(length, seed, kinds);
        Map<String, EvictingKvCache.EvictionPolicy> policies = new LinkedHashMap<>();
        policies.put("sliding", EvictingKvCache.sliding());
        policies.put("sinks", EvictingKvCache.sinks(4));
        policies.put("h2o", EvictingKvCache.heavyHitters(budget / 4));
        policies.put("learned", scorer);
        EvictingKvCache full = EvictingKvCache.of(attention);
        List<float[]> reference = new ArrayList<>();
        for (float[] x : test) {
            reference.add(full.append(x));
        }
        Map<String, Double> errors = new LinkedHashMap<>();
        for (var e : policies.entrySet()) {
            EvictingKvCache cache = EvictingKvCache.of(attention).budget(budget, e.getValue());
            double err = 0;
            int count = 0;
            for (int t = 0; t < length; t++) {
                float[] y = cache.append(test.get(t));
                if (kinds[t] == 2 && t > budget) {
                    err += mse(y, reference.get(t));
                    count++;
                }
            }
            errors.put(e.getKey(), err / Math.max(1, count));
        }
        return errors;
    }

    /**
     * Compresses a prompt of filler with a few facts, ending in questions, to {@code keep}
     * tokens with SnapKV or by keeping the most recent, and measures how well the next
     * question is answered.
     *
     * @param keep the tokens kept
     * @param seed the seed
     * @return the squared output error for "snapkv" and "recent"
     */
    public static Map<String, Double> snapKv(int keep, long seed) {
        MultiHeadAttention attention = layer();
        SplittableRandom r = new SplittableRandom(seed);
        int n = 80;
        List<float[]> prompt = new ArrayList<>();
        for (int t = 0; t < n; t++) {
            float[] x = new float[F];
            for (int i = 2; i < F; i++) {
                x[i] = (float) (r.nextGaussian() * 0.5);
            }
            if (t % 23 == 5) {
                x[0] = 3;          // a fact
            }
            if (t >= n - 4) {
                x[1] = 3;          // the question at the end of the prompt
            }
            prompt.add(x);
        }
        float[] next = prompt.get(n - 1).clone();
        Map<String, Double> out = new LinkedHashMap<>();
        EvictingKvCache full = EvictingKvCache.of(attention);
        prompt.forEach(full::append);
        float[] reference = full.append(next);
        EvictingKvCache snap = EvictingKvCache.of(attention);
        prompt.forEach(snap::append);
        snap.compress(SnapKv.of(4, 3), keep);
        out.put("snapkv", mse(snap.append(next), reference));
        EvictingKvCache recent = EvictingKvCache.of(attention).budget(keep + 1, EvictingKvCache.sliding());
        prompt.forEach(recent::append);
        out.put("recent", mse(recent.append(next), reference));
        return out;
    }

    private static double mse(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s / a.length;
    }
}
