package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Changes the structure of a model's {@link Brain} layers while a {@link Trainer} trains
 * it: add it with {@code trainer.listener(plasticity)}.
 *
 * <ul>
 *   <li><b>Rewiring</b>: every few steps, prune the weakest fraction of every projection's
 *       connections and grow as many new ones, where the loss gradient is largest (RigL,
 *       Evci et al. 2020) or at random (SET, Mocanu et al. 2018). The fraction shrinks to 0
 *       on a cosine schedule, so the wiring settles.</li>
 *   <li><b>Neurogenesis</b>: when the training loss stops improving, wake dormant neurons
 *       in the region that most wants them (largest gradient pressure).</li>
 *   <li><b>Neuron death</b>: put neurons whose activity stays near zero back to
 *       sleep.</li>
 * </ul>
 *
 * <pre>{@code
 * Plasticity plasticity = Plasticity.of(model)
 *         .rewire(100, 0.3)              // every 100 steps, swap 30% of connections
 *         .until(5_000)                  // anneal to no rewiring by step 5000
 *         .growth(Plasticity.Growth.GRADIENT)
 *         .neurogenesis(3, 0.01, 8)      // 3 epochs with < 1% improvement: add 8 neurons
 *         .pruneDeadNeurons(1e-3)
 *         .verbose(true);
 * Trainer.of(model).listener(plasticity).fit(data);
 * }</pre>
 */
public final class Plasticity implements TrainingListener {

    /** How new connections are chosen. */
    public enum Growth {
        /** where the loss gradient for the missing connection is largest (RigL) */
        GRADIENT,
        /** uniformly at random among missing connections (SET) */
        RANDOM
    }

    /**
     * A structural change.
     *
     * @param epoch the epoch it happened in
     * @param step the optimizer step it happened after
     * @param kind {@code rewire}, {@code grow neurons} or {@code remove neurons}
     * @param detail what changed
     */
    public record Event(int epoch, long step, String kind, String detail) {
        @Override
        public String toString() {
            return String.format("epoch %3d step %6d  %-14s %s", epoch, step, kind, detail);
        }
    }

    /**
     * The size of the brains after an epoch.
     *
     * @param epoch the epoch
     * @param loss its training loss
     * @param neurons active neurons in all brains
     * @param connections connections in all brains
     */
    public record Snapshot(int epoch, float loss, int neurons, long connections) {
    }

    private final List<Brain> brains;
    private int rewireEvery;
    private double rewireFraction;
    private long until = Long.MAX_VALUE;
    private Growth growth = Growth.GRADIENT;
    private int patience;
    private double minImprovement;
    private int newNeurons;
    private double deadThreshold;
    private boolean verbose;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Event> events = new ArrayList<>();
    private final List<Snapshot> history = new ArrayList<>();
    private float bestLoss = Float.POSITIVE_INFINITY;
    private int stale;
    private int epoch;
    private long step;
    private int nextRegion;

    private Plasticity(List<Brain> brains) {
        if (brains.isEmpty()) {
            throw new IllegalArgumentException("the model has no Brain layer");
        }
        this.brains = brains;
        for (Brain b : brains) {
            b.track(true);
        }
    }

    /**
     * @param model a model with one or more {@link Brain} layers
     * @return plasticity for all of them; nothing changes until options are set
     */
    public static Plasticity of(Model model) {
        List<Brain> found = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Brain b) {
                found.add(b);
            }
        }
        return new Plasticity(found);
    }

    /**
     * @param brain a brain
     * @return plasticity for it
     */
    public static Plasticity of(Brain brain) {
        return new Plasticity(List.of(Objects.requireNonNull(brain)));
    }

    /**
     * @param everySteps optimizer steps between rewirings
     * @param fraction the fraction of each projection's connections to swap at first
     * @return this
     */
    public Plasticity rewire(int everySteps, double fraction) {
        if (everySteps < 1 || fraction < 0 || fraction > 1) {
            throw new IllegalArgumentException("need everySteps >= 1 and fraction in [0, 1]");
        }
        this.rewireEvery = everySteps;
        this.rewireFraction = fraction;
        return this;
    }

    /**
     * @param step the step by which the rewiring fraction has fallen to 0 (cosine)
     * @return this
     */
    public Plasticity until(long step) {
        this.until = step;
        return this;
    }

    /**
     * @param g how new connections are chosen
     * @return this
     */
    public Plasticity growth(Growth g) {
        this.growth = Objects.requireNonNull(g);
        return this;
    }

    /**
     * @param patience epochs without enough improvement before growing
     * @param minImprovement the relative loss improvement that counts, for example 0.01
     * @param neurons how many neurons to add each time
     * @return this
     */
    public Plasticity neurogenesis(int patience, double minImprovement, int neurons) {
        this.patience = patience;
        this.minImprovement = minImprovement;
        this.newNeurons = neurons;
        return this;
    }

    /**
     * @param threshold neurons with mean absolute activity below this are removed after
     *     every epoch; 0 to keep them
     * @return this
     */
    public Plasticity pruneDeadNeurons(double threshold) {
        this.deadThreshold = threshold;
        return this;
    }

    /**
     * @param on whether to print every structural change
     * @return this
     */
    public Plasticity verbose(boolean on) {
        this.verbose = on;
        return this;
    }

    /**
     * @param seed the seed for random growth
     * @return this
     */
    public Plasticity seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return every structural change so far */
    public List<Event> events() {
        return Collections.unmodifiableList(events);
    }

    /** @return the size of the brains after every epoch */
    public List<Snapshot> history() {
        return Collections.unmodifiableList(history);
    }

    private void log(String kind, String detail) {
        Event e = new Event(epoch, step, kind, detail);
        events.add(e);
        if (verbose) {
            System.out.println(e);
        }
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        this.epoch = epoch;
        this.step = step;
        for (Brain b : brains) {
            b.collectGradients();
        }
        if (rewireEvery > 0 && step % rewireEvery == 0 && step < until) {
            double f = until == Long.MAX_VALUE ? rewireFraction
                    : rewireFraction * 0.5 * (1 + Math.cos(Math.PI * step / until));
            long swapped = 0;
            for (Brain b : brains) {
                int[] removed = b.prune(f);
                for (int j = 0; j < removed.length; j++) {
                    swapped += b.grow(j, removed[j], growth == Growth.GRADIENT, random);
                }
            }
            if (swapped > 0) {
                log("rewire", String.format("swapped %d connections (%.1f%%)", swapped, 100 * f));
            }
        }
    }

    @Override
    public void onEpoch(EpochResult result) {
        epoch = result.epoch();
        if (deadThreshold > 0 && result.epoch() > 1) {
            for (Brain b : brains) {
                Map<String, Integer> removed = b.removeDeadNeurons(deadThreshold);
                removed.forEach((region, n) -> log("remove neurons", n + " silent neurons in " + region));
            }
        }
        if (newNeurons > 0) {
            if (result.loss() < bestLoss * (1 - minImprovement)) {
                bestLoss = result.loss();
                stale = 0;
            } else if (++stale >= patience) {
                stale = 0;
                bestLoss = result.loss();
                for (Brain b : brains) {
                    String region = pick(b);
                    if (region != null) {
                        int added = b.addNeurons(region, newNeurons, random);
                        log("grow neurons", String.format("+%d in %s (loss %.4f stalled)", added, region,
                                result.loss()));
                    }
                }
            }
        }
        int neurons = 0;
        long connections = 0;
        for (Brain b : brains) {
            neurons += b.activeNeurons();
            connections += b.connections();
        }
        history.add(new Snapshot(result.epoch(), result.loss(), neurons, connections));
    }

    /** The region that most wants neurons and has room, or null. */
    private String pick(Brain b) {
        List<Brain.RegionStats> regions = b.regions();
        Map<String, Double> pressure = b.growthPressure();
        String best = null;
        double bestPressure = -1;
        if (growth == Growth.GRADIENT) {
            for (Brain.RegionStats r : regions) {
                double p = pressure.getOrDefault(r.name(), 0.0);
                if (r.active() < r.capacity() && p > bestPressure) {
                    best = r.name();
                    bestPressure = p;
                }
            }
            return best;
        }
        for (int k = 0; k < regions.size(); k++) {
            Brain.RegionStats r = regions.get((nextRegion + k) % regions.size());
            if (r.active() < r.capacity()) {
                nextRegion = (nextRegion + k + 1) % regions.size();
                return r.name();
            }
        }
        return null;
    }
}
