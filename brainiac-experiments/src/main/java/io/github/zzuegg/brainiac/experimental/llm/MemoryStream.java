package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * The memory stream of Generative Agents (Park et al.,
 * <a href="https://arxiv.org/abs/2304.03442">2304.03442</a>): an agent's long-term memory as
 * a time-ordered list of natural-language records, retrieved by a score that adds three
 * normalized signals:
 * <ul>
 *   <li><b>recency</b>: an exponential decay over the time since the memory was last
 *       retrieved, so memories in use stay fresh;</li>
 *   <li><b>importance</b>: how poignant the memory is (1 to 10, rated when it is written,
 *       by the model in the paper and by any function here);</li>
 *   <li><b>relevance</b>: the cosine similarity of the memory's embedding to the query,
 *       found with a core {@link VectorIndex}.</li>
 * </ul>
 * Each signal is min-max normalized over the candidates before the weighted sum, as in the
 * paper. When the importance of the observations since the last reflection passes a
 * threshold, {@link #maybeReflect} asks a {@link Reflector} for higher-level insights about
 * the recent memories and writes them back into the stream as memories of their own
 * (with pointers to their evidence), so later queries can retrieve the abstraction instead
 * of piecing it together from many observations.
 */
public final class MemoryStream {

    /** The kind of a memory. */
    public enum Kind {
        /** Something the agent perceived. */
        OBSERVATION,
        /** An insight the agent derived from other memories. */
        REFLECTION
    }

    /**
     * One memory.
     *
     * @param id its index in the stream
     * @param text the record
     * @param kind observation or reflection
     * @param created when it was written
     * @param importance its importance, 1 to 10
     * @param evidence for a reflection, the ids it was derived from
     */
    public record Memory(int id, String text, Kind kind, double created, double importance, List<Integer> evidence) {
    }

    /**
     * A retrieved memory with its score parts (each normalized to 0..1).
     *
     * @param memory the memory
     * @param score the weighted total
     * @param recency the recency part
     * @param importance the importance part
     * @param relevance the relevance part
     */
    public record Retrieved(Memory memory, double score, double recency, double importance, double relevance) {
    }

    /**
     * Writes insights about recent memories, as the model's reflection prompt does.
     */
    @FunctionalInterface
    public interface Reflector {
        /**
         * @param recent the memories since the last reflection, oldest first
         * @return insights, each with the ids of the memories it rests on
         */
        List<Insight> reflect(List<Memory> recent);
    }

    /**
     * A reflection to store.
     *
     * @param text the insight
     * @param importance its importance
     * @param evidence the memories it rests on
     */
    public record Insight(String text, double importance, List<Integer> evidence) {
    }

    private final Function<String, float[]> embedder;
    private final int dim;
    private final VectorIndex index;
    private final List<Memory> memories = new ArrayList<>();
    private final List<Double> lastAccess = new ArrayList<>();
    private ToDoubleFunction<String> importanceRater = s -> 5;
    private double decay = 0.995;
    private double wRecency = 1;
    private double wImportance = 1;
    private double wRelevance = 1;
    private double reflectThreshold = Double.POSITIVE_INFINITY;
    private double importanceSinceReflection;
    private int reflectedUpTo;

    private MemoryStream(Function<String, float[]> embedder, int dim) {
        this.embedder = Objects.requireNonNull(embedder);
        this.dim = dim;
        this.index = VectorIndex.flat(dim, VectorIndex.Metric.COSINE, Device.cpu());
    }

    /**
     * @param embedder embeds a text for relevance
     * @param dim the embedding width
     * @return an empty stream with equal weights, decay 0.995 per time unit and no reflection
     */
    public static MemoryStream of(Function<String, float[]> embedder, int dim) {
        return new MemoryStream(embedder, dim);
    }

    /**
     * A cheap embedder for tests and toys: words hashed into buckets, with signs.
     *
     * @param dim the width
     * @return the embedder
     */
    public static Function<String, float[]> hashedBagOfWords(int dim) {
        return text -> {
            float[] v = new float[dim];
            for (String w : text.toLowerCase(Locale.ROOT).split("[^a-z0-9*+-]+")) {
                if (!w.isEmpty()) {
                    int h = w.hashCode() * 0x9E3779B1;
                    v[Math.floorMod(h, dim)] += (h >>> 31) == 0 ? 1 : -1;
                }
            }
            return v;
        };
    }

    /**
     * @param rater rates a new observation's importance, 1 to 10 (5 by default)
     * @return this stream
     */
    public MemoryStream importance(ToDoubleFunction<String> rater) {
        this.importanceRater = Objects.requireNonNull(rater);
        return this;
    }

    /**
     * @param factor the recency multiplier per time unit since last access (0.995 per game hour in the paper)
     * @return this stream
     */
    public MemoryStream decay(double factor) {
        this.decay = factor;
        return this;
    }

    /**
     * @param recency the weight of recency
     * @param importance the weight of importance
     * @param relevance the weight of relevance
     * @return this stream
     */
    public MemoryStream weights(double recency, double importance, double relevance) {
        this.wRecency = recency;
        this.wImportance = importance;
        this.wRelevance = relevance;
        return this;
    }

    /**
     * @param threshold the summed importance of new observations that triggers a reflection (150 in the paper)
     * @return this stream
     */
    public MemoryStream reflectAfter(double threshold) {
        this.reflectThreshold = threshold;
        return this;
    }

    /**
     * Records an observation, rating its importance.
     *
     * @param text the observation
     * @param time the current time
     * @return the memory
     */
    public Memory observe(String text, double time) {
        return add(text, Kind.OBSERVATION, time, importanceRater.applyAsDouble(text), List.of());
    }

    /**
     * Records a memory with a given importance.
     *
     * @param text the record
     * @param kind its kind
     * @param time the current time
     * @param importance its importance
     * @param evidence the memories it rests on
     * @return the memory
     */
    public Memory add(String text, Kind kind, double time, double importance, List<Integer> evidence) {
        float[] e = embedder.apply(text);
        if (e.length != dim) {
            throw new IllegalArgumentException("embedding has " + e.length + " values, expected " + dim);
        }
        Memory m = new Memory(memories.size(), text, kind, time, importance, List.copyOf(evidence));
        memories.add(m);
        lastAccess.add(time);
        index.add(Tensor.of(Shape.of(1, dim), e, Device.cpu()));
        if (kind == Kind.OBSERVATION) {
            importanceSinceReflection += importance;
        }
        return m;
    }

    /**
     * Reflects if enough has happened since the last reflection.
     *
     * @param reflector writes the insights
     * @param time the current time
     * @return the reflections written (empty if below the threshold)
     */
    public List<Memory> maybeReflect(Reflector reflector, double time) {
        if (importanceSinceReflection < reflectThreshold) {
            return List.of();
        }
        return reflect(reflector, time);
    }

    /**
     * Reflects now over the memories since the last reflection.
     *
     * @param reflector writes the insights
     * @param time the current time
     * @return the reflections written
     */
    public List<Memory> reflect(Reflector reflector, double time) {
        List<Memory> recent = List.copyOf(memories.subList(reflectedUpTo, memories.size()));
        List<Memory> out = new ArrayList<>();
        for (Insight in : reflector.reflect(recent)) {
            out.add(add(in.text(), Kind.REFLECTION, time, in.importance(), in.evidence()));
        }
        reflectedUpTo = memories.size();
        importanceSinceReflection = 0;
        return out;
    }

    /**
     * Retrieves the top memories for a query and marks them as accessed.
     *
     * @param query the query text
     * @param k how many
     * @param time the current time
     * @return the best {@code k}, best first
     */
    public List<Retrieved> retrieve(String query, int k, double time) {
        int n = memories.size();
        if (n == 0) {
            return List.of();
        }
        double[] rel = new double[n];
        for (VectorIndex.Hit h : index.search(Tensor.of(embedder.apply(query)), n)) {
            rel[h.id()] = h.score();
        }
        double[] rec = new double[n];
        double[] imp = new double[n];
        for (int i = 0; i < n; i++) {
            rec[i] = Math.pow(decay, Math.max(0, time - lastAccess.get(i)));
            imp[i] = memories.get(i).importance();
        }
        normalize(rec);
        normalize(imp);
        normalize(rel);
        List<Retrieved> all = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            all.add(new Retrieved(memories.get(i), wRecency * rec[i] + wImportance * imp[i] + wRelevance * rel[i],
                    rec[i], imp[i], rel[i]));
        }
        all.sort(Comparator.comparingDouble(Retrieved::score).reversed());
        List<Retrieved> top = all.subList(0, Math.min(k, n));
        for (Retrieved r : top) {
            lastAccess.set(r.memory().id(), time);
        }
        return List.copyOf(top);
    }

    private static void normalize(double[] v) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        for (int i = 0; i < v.length; i++) {
            v[i] = hi > lo ? (v[i] - lo) / (hi - lo) : 0;
        }
    }

    /** @return every memory, oldest first */
    public List<Memory> memories() {
        return List.copyOf(memories);
    }

    /** @return the number of memories */
    public int size() {
        return memories.size();
    }

    @Override
    public String toString() {
        long reflections = memories.stream().filter(m -> m.kind() == Kind.REFLECTION).count();
        return "MemoryStream(" + memories.size() + " memories, " + reflections + " reflections)";
    }
}
