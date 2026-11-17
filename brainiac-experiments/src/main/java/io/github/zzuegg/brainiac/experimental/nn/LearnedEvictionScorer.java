package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A tiny learned KV-cache eviction policy: an MLP that predicts, from what a cached token
 * looks like now ({@link EvictingKvCache.SlotInfo}: attention received so far and lately,
 * age, key and value norms, whether it is the first token), how much attention it will
 * receive in the next {@code horizon} steps, and evicts the token with the lowest prediction
 * (apart from the most recent ones). It is trained by regression on traces of a full cache,
 * where the future is known (an oracle, as in learned cache policies such as
 * "Learning to evict", and the data-driven scorers studied after H2O,
 * <a href="https://arxiv.org/abs/2306.14048">arXiv:2306.14048</a>). Unlike H2O it can keep a
 * token that has not been attended yet but looks like the tokens that will be, for example
 * one with an unusual key.
 */
public final class LearnedEvictionScorer implements EvictingKvCache.EvictionPolicy {

    /** The number of features per token. */
    public static final int FEATURES = 7;

    private final Dense hidden;
    private final Dense out;
    private int recent = 4;

    /**
     * An untrained scorer.
     *
     * @param units the hidden units
     * @param seed the random seed
     */
    public LearnedEvictionScorer(int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        hidden = new Dense(FEATURES, units, Activation.TANH, Init.XAVIER, r);
        out = new Dense(units, 1, Activation.IDENTITY, Init.XAVIER, r);
    }

    /**
     * @param tokens the most recent tokens never evicted
     * @return this scorer
     */
    public LearnedEvictionScorer recent(int tokens) {
        this.recent = tokens;
        return this;
    }

    /**
     * The features of a cached token.
     *
     * @param s the token
     * @return {@link #FEATURES} values
     */
    public static float[] features(EvictingKvCache.SlotInfo s) {
        return new float[] {
            (float) Math.log1p(s.cumulativeAttention()),
            s.meanAttention() * s.cached(),
            s.lastAttention() * s.cached(),
            s.keyNorm(),
            s.valueNorm(),
            s.age() / (s.age() + 16f),
            s.position() == 0 ? 1 : 0
        };
    }

    /**
     * Predicts a token's future attention.
     *
     * @param s the token
     * @return the score, larger is more worth keeping
     */
    public float score(EvictingKvCache.SlotInfo s) {
        Var x = Var.of(Tensor.of(Shape.of(1, FEATURES), features(s)));
        return out.forward(hidden.forward(x)).value().item();
    }

    @Override
    public int evict(List<EvictingKvCache.SlotInfo> slots) {
        int last = Math.max(1, slots.size() - recent);
        float[] x = new float[last * FEATURES];
        for (int i = 0; i < last; i++) {
            System.arraycopy(features(slots.get(i)), 0, x, i * FEATURES, FEATURES);
        }
        float[] s = out.forward(hidden.forward(Var.of(Tensor.of(Shape.of(last, FEATURES), x)))).value().toArray();
        int best = 0;
        for (int i = 1; i < last; i++) {
            if (s[i] < s[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Training examples from a full cache.
     *
     * @param features {@code [n][FEATURES]}
     * @param targets the attention each token received over the following steps, times the cache size
     */
    public record Examples(float[][] features, float[] targets) {
    }

    /**
     * Runs a stream through a layer with a cache that keeps everything and records, every
     * {@code stride} steps, each cached token's features and the attention it then receives
     * over the next {@code horizon} steps.
     *
     * @param attention the layer
     * @param stream the token features
     * @param horizon the steps of future attention to predict
     * @param stride record every this many steps
     * @return the examples
     */
    public static Examples harvest(MultiHeadAttention attention, List<float[]> stream, int horizon, int stride) {
        EvictingKvCache cache = EvictingKvCache.of(attention).record(true);
        List<float[]> feats = new ArrayList<>();
        List<int[]> at = new ArrayList<>();   // {step, position}
        for (int t = 0; t < stream.size(); t++) {
            cache.append(stream.get(t));
            if (t % stride == 0 && t + horizon < stream.size()) {
                for (EvictingKvCache.SlotInfo s : cache.slotInfo()) {
                    feats.add(features(s));
                    at.add(new int[] {t, s.position()});
                }
            }
        }
        List<EvictingKvCache.Step> steps = cache.steps();
        List<Map<Integer, Float>> weights = new ArrayList<>();
        for (EvictingKvCache.Step s : steps) {
            Map<Integer, Float> m = new HashMap<>();
            for (int i = 0; i < s.keys().length; i++) {
                m.put(s.keys()[i], s.weights()[i]);
            }
            weights.add(m);
        }
        float[] targets = new float[at.size()];
        for (int i = 0; i < at.size(); i++) {
            int t = at.get(i)[0];
            int p = at.get(i)[1];
            double sum = 0;
            for (int f = t + 1; f <= t + horizon; f++) {
                sum += weights.get(f).getOrDefault(p, 0f) * steps.get(f).keys().length;
            }
            targets[i] = (float) (sum / horizon);
        }
        return new Examples(feats.toArray(float[][]::new), targets);
    }

    /**
     * Fits the scorer to examples by mean squared error.
     *
     * @param examples the examples
     * @param iterations full-batch Adam steps
     * @return the final loss
     */
    public float fit(Examples examples, int iterations) {
        Tensor x = Tensor.of(examples.features());
        Tensor y = Tensor.of(Shape.of(examples.targets().length, 1), examples.targets());
        Optimizer opt = Optimizer.adam(0.01f);
        List<Var> params = new ArrayList<>(hidden.parameters());
        params.addAll(out.parameters());
        float loss = 0;
        for (int i = 0; i < iterations; i++) {
            Var l = out.forward(hidden.forward(Var.of(x))).sub(y).square().mean();
            loss = l.value().item();
            l.backward();
            opt.step(params);
        }
        return loss;
    }

    @Override
    public String toString() {
        return "LearnedEvictionScorer(recent " + recent + ")";
    }
}
