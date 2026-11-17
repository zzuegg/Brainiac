package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A key/value cache for one {@link MultiHeadAttention} layer with a pluggable eviction policy
 * and full bookkeeping of what each token read: the testbed for KV-cache compression. It
 * runs the layer's own weights step by step (with all tokens kept, its outputs equal the
 * layer's causal forward pass) and, when it holds more than {@code budget} tokens, asks an
 * {@link EvictionPolicy} which to drop. Policies see per-token statistics ({@link SlotInfo}:
 * age, attention received so far and lately, key and value norms) and can be the classic
 * ones ({@link #sliding()}, {@link #sinks(int)}, {@link #heavyHitters(int)}, the core
 * {@code KvCache.Strategy} set) or a {@link LearnedEvictionScorer}. A prompt can also be
 * compressed once with {@link SnapKv} voting.
 *
 * <p>With {@link #record(boolean) recording} on, every step's attention weights and
 * evictions are kept for {@link AttentionCacheViewer}. The core {@code KvCache.Strategy} is a
 * sealed interface, so these policies cannot yet be passed to the core cache; this class
 * mirrors its behaviour for one layer instead. Inference only, batch of one, float32.
 */
public final class EvictingKvCache {

    /**
     * What a policy knows about a cached token.
     *
     * @param position the token's position in the stream
     * @param age steps since it was added
     * @param cumulativeAttention the attention it received so far, summed over steps (mean over heads)
     * @param meanAttention cumulative attention divided by the steps it was cached for
     * @param lastAttention the attention it received at the last step
     * @param keyNorm the norm of its key (all key/value heads)
     * @param valueNorm the norm of its value
     * @param cached how many tokens the cache holds now
     */
    public record SlotInfo(int position, int age, float cumulativeAttention, float meanAttention, float lastAttention,
            float keyNorm, float valueNorm, int cached) {
    }

    /** Chooses which cached token to drop. */
    @FunctionalInterface
    public interface EvictionPolicy {
        /**
         * @param slots the cached tokens, oldest first; the newest one is last and must stay
         * @return the index in {@code slots} of the token to drop
         */
        int evict(List<SlotInfo> slots);
    }

    /**
     * One recorded step.
     *
     * @param position the new token's position
     * @param label the token's label, for display
     * @param keys the positions it attended to
     * @param weights its attention weight on each, averaged over heads
     * @param evicted the positions dropped at this step
     */
    public record Step(int position, String label, int[] keys, float[] weights, int[] evicted) {
    }

    private static final class Slot {
        final int position;
        final float[] key;
        final float[] value;
        final float keyNorm;
        final float valueNorm;
        float cumulative;
        float last;
        int steps;

        Slot(int position, float[] key, float[] value) {
            this.position = position;
            this.key = key;
            this.value = value;
            this.keyNorm = norm(key);
            this.valueNorm = norm(value);
        }
    }

    private final int features;
    private final int heads;
    private final int kvHeads;
    private final int headSize;
    private final int valueSize;
    private final int outputSize;
    private final float rotaryBase;
    private final float scale;
    private final float[] wq;
    private final float[] wk;
    private final float[] wv;
    private final float[] wo;
    private final float[] bq;
    private final float[] bk;
    private final float[] bv;
    private final float[] bo;
    private final List<Slot> slots = new ArrayList<>();
    private final List<Step> steps = new ArrayList<>();
    private final List<float[][]> windowAttention = new ArrayList<>();
    private int budget = Integer.MAX_VALUE;
    private EvictionPolicy policy = sliding();
    private boolean record;
    private int position;

    private EvictingKvCache(MultiHeadAttention attention) {
        MultiHeadAttention.Settings s = attention.settings();
        if (s.queryFeatures() != s.kvFeatures() || s.alibi() || s.keyMask()) {
            throw new IllegalArgumentException(attention + " is not plain self-attention");
        }
        features = s.queryFeatures();
        heads = s.heads();
        kvHeads = s.kvHeads();
        headSize = s.headSize();
        valueSize = s.valueSize();
        outputSize = s.outputSize();
        rotaryBase = s.rotaryBase();
        scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(headSize));
        List<Var> p = attention.parameters();
        wq = p.get(0).value().toArray();
        wk = p.get(1).value().toArray();
        wv = p.get(2).value().toArray();
        wo = p.get(3).value().toArray();
        boolean bias = p.size() > 4;
        bq = bias ? p.get(4).value().toArray() : null;
        bk = bias ? p.get(5).value().toArray() : null;
        bv = bias ? p.get(6).value().toArray() : null;
        bo = bias ? p.get(7).value().toArray() : null;
    }

    /**
     * A cache for a self-attention layer, reading its current weights.
     *
     * @param attention the layer
     * @return an empty cache that keeps everything
     */
    public static EvictingKvCache of(MultiHeadAttention attention) {
        return new EvictingKvCache(attention);
    }

    /**
     * @param tokens the most tokens kept
     * @param rule which token goes when the cache is over budget
     * @return this cache
     */
    public EvictingKvCache budget(int tokens, EvictionPolicy rule) {
        if (tokens < 1) {
            throw new IllegalArgumentException("budget must be positive");
        }
        this.budget = tokens;
        this.policy = Objects.requireNonNull(rule, "rule");
        return this;
    }

    /**
     * @param on whether to keep every step's attention for {@link #steps()}
     * @return this cache
     */
    public EvictingKvCache record(boolean on) {
        this.record = on;
        return this;
    }

    /** @return drop the oldest token */
    public static EvictionPolicy sliding() {
        return slots -> 0;
    }

    /**
     * @param count the first tokens kept for good
     * @return keep the first tokens, then drop the oldest (StreamingLLM)
     */
    public static EvictionPolicy sinks(int count) {
        return slots -> {
            for (int i = 0; i < slots.size() - 1; i++) {
                if (slots.get(i).position() >= count) {
                    return i;
                }
            }
            return 0;
        };
    }

    /**
     * @param recent the most recent tokens always kept
     * @return drop the older token that received the least attention so far (H2O)
     */
    public static EvictionPolicy heavyHitters(int recent) {
        return slots -> {
            int last = Math.max(1, slots.size() - recent);
            int best = 0;
            for (int i = 1; i < last; i++) {
                if (slots.get(i).cumulativeAttention() < slots.get(best).cumulativeAttention()) {
                    best = i;
                }
            }
            return best;
        };
    }

    /** @return how many tokens the cache holds */
    public int size() {
        return slots.size();
    }

    /** @return the positions of the cached tokens, oldest first */
    public int[] positions() {
        return slots.stream().mapToInt(s -> s.position).toArray();
    }

    /** @return the statistics of the cached tokens, oldest first */
    public List<SlotInfo> slotInfo() {
        List<SlotInfo> out = new ArrayList<>(slots.size());
        for (Slot s : slots) {
            out.add(new SlotInfo(s.position, position - s.position, s.cumulative,
                    s.steps == 0 ? 0 : s.cumulative / s.steps, s.last, s.keyNorm, s.valueNorm, slots.size()));
        }
        return out;
    }

    /** @return the recorded steps */
    public List<Step> steps() {
        return List.copyOf(steps);
    }

    /** @return the tokens fed so far */
    public int length() {
        return position;
    }

    /**
     * Feeds one token.
     *
     * @param x the token's features
     * @return the layer's output for it
     */
    public float[] append(float[] x) {
        return append(x, Integer.toString(position));
    }

    /**
     * Feeds one token with a display label.
     *
     * @param x the token's features
     * @param label the label shown by the viewer
     * @return the layer's output for it
     */
    public float[] append(float[] x, String label) {
        if (x.length != features) {
            throw new IllegalArgumentException("expected " + features + " features, got " + x.length);
        }
        float[] q = rotate(project(x, wq, bq, heads * headSize), heads, position);
        float[] k = rotate(project(x, wk, bk, kvHeads * headSize), kvHeads, position);
        float[] v = project(x, wv, bv, kvHeads * valueSize);
        slots.add(new Slot(position, k, v));
        List<Integer> evicted = new ArrayList<>();
        while (slots.size() > budget) {
            int i = policy.evict(slotInfo());
            if (i < 0 || i >= slots.size() - 1) {
                throw new IllegalStateException("policy chose " + i + ", must be below " + (slots.size() - 1));
            }
            evicted.add(slots.remove(i).position);
        }
        int n = slots.size();
        float[] mean = new float[n];
        float[][] perHead = new float[heads][n];
        float[] concat = new float[heads * valueSize];
        int group = heads / kvHeads;
        for (int h = 0; h < heads; h++) {
            int kh = h / group;
            double max = Double.NEGATIVE_INFINITY;
            double[] sc = new double[n];
            for (int j = 0; j < n; j++) {
                sc[j] = scale * KnnIndex.dot(q, h * headSize, slots.get(j).key, kh * headSize, headSize);
                max = Math.max(max, sc[j]);
            }
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sc[j] = Math.exp(sc[j] - max);
                sum += sc[j];
            }
            for (int j = 0; j < n; j++) {
                float w = (float) (sc[j] / sum);
                perHead[h][j] = w;
                mean[j] += w / heads;
                float[] val = slots.get(j).value;
                for (int c = 0; c < valueSize; c++) {
                    concat[h * valueSize + c] += w * val[kh * valueSize + c];
                }
            }
        }
        for (int j = 0; j < n; j++) {
            Slot s = slots.get(j);
            s.cumulative += mean[j];
            s.last = mean[j];
            s.steps++;
        }
        windowAttention.add(new float[][] {positionsAsFloats(), mean});
        if (windowAttention.size() > 64) {
            windowAttention.remove(0);
        }
        if (record) {
            steps.add(new Step(position, label, positions(), mean, evicted.stream().mapToInt(Integer::intValue)
                    .toArray()));
        }
        position++;
        return project(concat, wo, bo, outputSize);
    }

    private float[] positionsAsFloats() {
        float[] p = new float[slots.size()];
        for (int i = 0; i < p.length; i++) {
            p[i] = slots.get(i).position;
        }
        return p;
    }

    /**
     * Compresses the cache once with SnapKV: the attention of the last {@code window} tokens
     * votes for which older tokens stay (see {@link SnapKv}).
     *
     * @param snap the voting rule
     * @param keep the tokens kept, window included
     * @return the positions dropped
     */
    public int[] compress(SnapKv snap, int keep) {
        int w = Math.min(snap.window(), windowAttention.size());
        float[][] votes = new float[w][];
        int n = slots.size();
        int[] pos = positions();
        for (int i = 0; i < w; i++) {
            float[][] step = windowAttention.get(windowAttention.size() - w + i);
            float[] aligned = new float[n];
            for (int j = 0; j < step[0].length; j++) {
                int at = java.util.Arrays.binarySearch(pos, (int) step[0][j]);
                if (at >= 0) {
                    aligned[at] = step[1][j];
                }
            }
            votes[i] = aligned;
        }
        boolean[] kept = snap.select(votes, keep);
        List<Integer> dropped = new ArrayList<>();
        List<Slot> stay = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            if (kept[j]) {
                stay.add(slots.get(j));
            } else {
                dropped.add(slots.get(j).position);
            }
        }
        slots.clear();
        slots.addAll(stay);
        return dropped.stream().mapToInt(Integer::intValue).toArray();
    }

    private float[] project(float[] x, float[] w, float[] b, int out) {
        int in = x.length;
        float[] y = new float[out];
        for (int i = 0; i < in; i++) {
            float xi = x[i];
            if (xi == 0) {
                continue;
            }
            for (int o = 0; o < out; o++) {
                y[o] += xi * w[i * out + o];
            }
        }
        if (b != null) {
            for (int o = 0; o < out; o++) {
                y[o] += b[o];
            }
        }
        return y;
    }

    /** Rotary embedding by absolute position, in the half-split layout of {@code Functional.rotary}. */
    private float[] rotate(float[] x, int count, int pos) {
        if (rotaryBase <= 0) {
            return x;
        }
        int half = headSize / 2;
        float[] y = x.clone();
        for (int h = 0; h < count; h++) {
            int o = h * headSize;
            for (int i = 0; i < half; i++) {
                double angle = pos * Math.pow(rotaryBase, -2.0 * i / headSize);
                float c = (float) Math.cos(angle);
                float s = (float) Math.sin(angle);
                float a = x[o + i];
                float b = x[o + half + i];
                y[o + i] = a * c - b * s;
                y[o + half + i] = b * c + a * s;
            }
        }
        return y;
    }

    private static float norm(float[] x) {
        double s = 0;
        for (float v : x) {
            s += v * v;
        }
        return (float) Math.sqrt(s);
    }

    /** Forgets everything. */
    public void reset() {
        slots.clear();
        steps.clear();
        windowAttention.clear();
        position = 0;
    }

    @Override
    public String toString() {
        return "EvictingKvCache(" + slots.size() + " of " + (budget == Integer.MAX_VALUE ? "all" : budget) + ")";
    }
}
