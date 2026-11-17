package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learning in wake and sleep phases, after complementary learning systems theory (McClelland
 * et al. 1995) and generative/experience replay (Shin et al. 2017,
 * <a href="https://arxiv.org/abs/1705.08690">arXiv:1705.08690</a>; Rolnick et al. 2019,
 * <a href="https://arxiv.org/abs/1811.11682">arXiv:1811.11682</a>). While awake, the model
 * learns online from what it sees now and every example also goes into a fast episodic
 * store (the "hippocampus"). In a {@link #sleep} phase the store is replayed, interleaved
 * across everything seen so far, into the slow weights (the "neocortex"), and then
 * {@linkplain Pruning pruned} to a fixed budget. Interleaved replay is what prevents
 * catastrophic forgetting when tasks arrive one after the other.
 *
 * <p>Use {@link #wake} for online steps, or train with a core
 * {@link io.github.zzuegg.brainiac.train.Trainer} and attach {@link #listener} so the model
 * sleeps every few epochs.
 */
public final class SleepConsolidation {

    /** Which memories survive pruning after sleep. */
    public enum Pruning {
        /** A uniform random sample. */
        RANDOM,
        /** The ones the model still gets most wrong (largest loss after sleep). */
        SURPRISE,
        /** Equal numbers per target class (argmax of the target), random within a class. */
        BALANCED
    }

    private record Memory(float[] x, float[] y) {
    }

    private final Model model;
    private final Loss loss;
    private final Optimizer optimizer;
    private final SplittableRandom random;
    private final List<Memory> store = new ArrayList<>();
    private int budget = 200;
    private Pruning pruning = Pruning.BALANCED;
    private int sleeps;

    private SleepConsolidation(Model model, Loss loss, Optimizer optimizer, long seed) {
        this.model = model;
        this.loss = loss;
        this.optimizer = optimizer;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Starts wake/sleep learning for a model.
     *
     * @param model the model
     * @param loss the loss
     * @param optimizer the optimizer for both phases
     * @param seed the seed for replay and pruning
     * @return the learner
     */
    public static SleepConsolidation of(Model model, Loss loss, Optimizer optimizer, long seed) {
        return new SleepConsolidation(model, loss, optimizer, seed);
    }

    /**
     * @param memories the most examples kept in the store after each sleep
     * @param rule which survive
     * @return this learner
     */
    public SleepConsolidation store(int memories, Pruning rule) {
        if (memories < 1) {
            throw new IllegalArgumentException("budget must be positive");
        }
        this.budget = memories;
        this.pruning = rule;
        return this;
    }

    /** @return the examples in the episodic store */
    public int storeSize() {
        return store.size();
    }

    /** @return how many times the model slept */
    public int sleeps() {
        return sleeps;
    }

    /**
     * One online learning step on a batch, which is also remembered.
     *
     * @param batch the examples seen now
     * @return the loss on them before the step
     */
    public float wake(Dataset batch) {
        for (int i = 0; i < batch.size(); i++) {
            store.add(new Memory(batch.featuresOf(i), batch.targetsOf(i)));
        }
        return step(batch.features(), batch.targets());
    }

    /**
     * Remembers examples without learning from them (as a listener does while a trainer learns).
     *
     * @param data the examples
     */
    public void remember(Dataset data) {
        for (int i = 0; i < data.size(); i++) {
            store.add(new Memory(data.featuresOf(i), data.targetsOf(i)));
        }
    }

    private float step(Tensor x, Tensor y) {
        Var l = loss.compute(model.forward(Var.of(x), true), y);
        float value = l.value().item();
        l.backward();
        optimizer.step(model.parameters());
        return value;
    }

    /**
     * Replays the store into the weights, then prunes it to the budget.
     *
     * @param steps the replay steps
     * @param batchSize the examples per replay step, drawn uniformly from the whole store
     * @return the mean replay loss
     */
    public float sleep(int steps, int batchSize) {
        if (store.isEmpty()) {
            return 0;
        }
        double total = 0;
        for (int s = 0; s < steps; s++) {
            int n = Math.min(batchSize, store.size());
            float[][] xs = new float[n][];
            float[][] ys = new float[n][];
            for (int i = 0; i < n; i++) {
                Memory m = store.get(random.nextInt(store.size()));
                xs[i] = m.x;
                ys[i] = m.y;
            }
            Dataset d = Dataset.of(xs, ys);
            total += step(d.features(), d.targets());
        }
        prune();
        sleeps++;
        return (float) (total / Math.max(1, steps));
    }

    private void prune() {
        if (store.size() <= budget) {
            return;
        }
        List<Memory> kept = new ArrayList<>(budget);
        switch (pruning) {
            case RANDOM -> {
                List<Memory> all = new ArrayList<>(store);
                for (int i = 0; i < budget; i++) {
                    kept.add(all.remove(random.nextInt(all.size())));
                }
            }
            case SURPRISE -> {
                float[] losses = new float[store.size()];
                for (int i = 0; i < store.size(); i++) {
                    Memory m = store.get(i);
                    Tensor x = Tensor.of(Shape.of(1, m.x.length), m.x);
                    losses[i] = loss.compute(model.forward(Var.of(x), false), Tensor.of(Shape.of(1, m.y.length), m.y))
                            .value().item();
                }
                Integer[] order = new Integer[store.size()];
                for (int i = 0; i < order.length; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, Comparator.comparingDouble(i -> -losses[i]));
                for (int i = 0; i < budget; i++) {
                    kept.add(store.get(order[i]));
                }
            }
            case BALANCED -> {
                Map<Integer, List<Memory>> byClass = new LinkedHashMap<>();
                for (Memory m : store) {
                    byClass.computeIfAbsent(argmax(m.y), c -> new ArrayList<>()).add(m);
                }
                List<List<Memory>> groups = new ArrayList<>(byClass.values());
                for (List<Memory> g : groups) {
                    // shuffle within the class
                    for (int i = g.size() - 1; i > 0; i--) {
                        int j = random.nextInt(i + 1);
                        Memory t = g.get(i);
                        g.set(i, g.get(j));
                        g.set(j, t);
                    }
                }
                for (int round = 0; kept.size() < budget; round++) {
                    for (List<Memory> g : groups) {
                        if (round < g.size() && kept.size() < budget) {
                            kept.add(g.get(round));
                        }
                    }
                }
            }
        }
        store.clear();
        store.addAll(kept);
    }

    private static int argmax(float[] y) {
        if (y.length == 1) {
            return Math.round(y[0]);
        }
        int best = 0;
        for (int i = 1; i < y.length; i++) {
            if (y[i] > y[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * A listener for a core trainer: the training data seen in an epoch should be passed to
     * {@link #remember} beforehand; every {@code everyEpochs} epochs the model sleeps.
     *
     * @param everyEpochs how often to sleep
     * @param steps the replay steps per sleep
     * @param batchSize the examples per replay step
     * @return the listener
     */
    public TrainingListener listener(int everyEpochs, int steps, int batchSize) {
        return (EpochResult result) -> {
            if (result.epoch() % everyEpochs == 0) {
                sleep(steps, batchSize);
            }
        };
    }
}
