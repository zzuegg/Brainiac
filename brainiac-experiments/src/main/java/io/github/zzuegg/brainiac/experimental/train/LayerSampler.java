package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * LISA, layerwise importance sampled AdamW: full-parameter fine-tuning in which only a few
 * randomly chosen middle layers (plus the always-trained embedding and head) are unfrozen at a
 * time. Every {@code period} steps a new set of {@code active} layers is drawn and a fresh inner
 * optimizer is started, so the optimizer keeps state only for the layers being trained: peak
 * optimizer memory is that of LoRA-sized training while every weight still gets updated over the
 * run. The paper found LoRA's per-layer weight-norm profile skewed to the embedding and head and
 * the middle layers nearly idle; sampling them uniformly matches or beats LoRA and often full
 * fine-tuning.
 *
 * <p>It is an {@link Optimizer} wrapper: {@link #step} receives every parameter (as any training
 * loop passes them) and updates only the active ones, so swapping it in is one line.
 * {@link #peakStateValues()} reports the most optimizer-state values held at once (two per
 * trained value for Adam-type inner optimizers).
 *
 * <p>Pan et al., "LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model
 * Fine-Tuning" (NeurIPS 2024), <a href="https://arxiv.org/abs/2403.17919">arXiv:2403.17919</a>.
 */
public final class LayerSampler implements Optimizer {

    private final Supplier<Optimizer> factory;
    private final List<List<Var>> layers;
    private final List<Var> always;
    private final int active;
    private final int period;
    private final int statesPerValue;
    private final SplittableRandom random;
    private Optimizer inner;
    private List<Var> trained = List.of();
    private long steps;
    private long peak;
    private float learningRate;

    /**
     * @param factory makes a fresh inner optimizer for every period
     * @param layers the sampled layers, each a list of its parameters
     * @param always parameters trained in every period (embedding, head)
     * @param active the layers unfrozen per period
     * @param period steps between resampling
     * @param statesPerValue optimizer-state values per trained value (2 for Adam), for the memory count
     * @param seed seeds the sampling
     */
    public LayerSampler(Supplier<Optimizer> factory, List<List<Var>> layers, List<Var> always, int active, int period,
            int statesPerValue, long seed) {
        if (active < 0 || active > layers.size() || period < 1) {
            throw new IllegalArgumentException("need 0 <= active <= layers and a positive period");
        }
        this.factory = Objects.requireNonNull(factory, "factory");
        this.layers = List.copyOf(layers);
        this.always = List.copyOf(always);
        this.active = active;
        this.period = period;
        this.statesPerValue = statesPerValue;
        this.random = new SplittableRandom(seed);
        this.learningRate = factory.get().learningRate();
    }

    /** @return the parameters trained in the current period */
    public List<Var> trained() {
        return trained;
    }

    /** @return the most optimizer-state values held at once so far */
    public long peakStateValues() {
        return peak;
    }

    private void resample() {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            order.add(i);
        }
        Collections.shuffle(order, new java.util.Random(random.nextLong()));
        List<Var> t = new ArrayList<>(always);
        for (int i = 0; i < active; i++) {
            t.addAll(layers.get(order.get(i)));
        }
        trained = List.copyOf(t);
        inner = factory.get();   // the old state is dropped: only the active layers hold any
        inner.learningRate(learningRate);
        long values = 0;
        for (Var v : trained) {
            values += v.shape().size();
        }
        peak = Math.max(peak, values * statesPerValue);
    }

    @Override
    public void step(List<Var> parameters) {
        if (steps % period == 0) {
            resample();
        }
        steps++;
        Map<Var, Boolean> given = new IdentityHashMap<>();
        for (Var v : parameters) {
            given.put(v, true);
        }
        List<Var> update = new ArrayList<>();
        for (Var v : trained) {
            if (given.containsKey(v)) {
                update.add(v);
            }
        }
        inner.step(update);
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
        if (inner != null) {
            inner.learningRate(learningRate);
        }
    }

    @Override
    public String toString() {
        return "LayerSampler(" + active + " of " + layers.size() + " layers every " + period + " steps)";
    }
}
