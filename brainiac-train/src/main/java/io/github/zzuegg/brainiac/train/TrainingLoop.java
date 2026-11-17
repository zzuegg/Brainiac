package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Step-by-step training for models that do not fit {@link Trainer}'s epochs of a {@link
 * io.github.zzuegg.brainiac.data.Dataset}, such as language models: you compute each loss,
 * the loop does the rest. It accumulates gradients over micro-batches, clips them, sets the
 * learning rate for each step, updates weight averages, frees each step's memory, and saves
 * and resumes checkpoints.
 *
 * <pre>{@code
 * TrainingLoop loop = TrainingLoop.of(model.namedParameters(), Optimizer.adamW(3e-4f, 0.1f))
 *         .accumulate(8)
 *         .clipGradientNorm(1)
 *         .schedule(LearningRateSchedule.warmup(100, LearningRateSchedule.cosine(10_000, 3e-5f)))
 *         .ema(0.999f);
 * loop.resumeIfPresent(Path.of("checkpoints/run1"));
 * while (loop.step() < 10_000) {
 *     float loss = loop.microBatch(() -> lossOn(nextBatch()));
 *     if (loop.step() % 500 == 0 && loop.stepped()) {
 *         loop.save(Path.of("checkpoints/run1"));
 *     }
 * }
 * }</pre>
 */
public final class TrainingLoop {

    private final Map<String, Var> parameters;
    private final List<Var> params;
    private final Optimizer optimizer;
    private int accumulate = 1;
    private float clipNorm;
    private LearningRateSchedule schedule;
    private float baseRate;
    private Ema ema;
    private int micro;
    private int step;
    private boolean stepped;
    private Tensor lastNorm;
    private java.util.function.Consumer<List<Var>> beforeStep;

    private TrainingLoop(Map<String, Var> parameters, Optimizer optimizer) {
        this.parameters = new LinkedHashMap<>(parameters);
        this.params = List.copyOf(this.parameters.values());
        this.optimizer = optimizer;
        this.baseRate = optimizer.learningRate();
    }

    /**
     * @param parameters the weights to train, by name (the names are used in checkpoints)
     * @param optimizer their optimizer
     * @return a loop
     */
    public static TrainingLoop of(Map<String, Var> parameters, Optimizer optimizer) {
        return new TrainingLoop(parameters, optimizer);
    }

    /**
     * @param parameters the weights to train, named by their index in checkpoints
     * @param optimizer their optimizer
     * @return a loop
     */
    public static TrainingLoop of(List<Var> parameters, Optimizer optimizer) {
        Map<String, Var> named = new LinkedHashMap<>();
        for (int i = 0; i < parameters.size(); i++) {
            named.put(Integer.toString(i), parameters.get(i));
        }
        return new TrainingLoop(named, optimizer);
    }

    /**
     * Sums the gradients of several micro-batches before each step, as one batch that many
     * times larger would, for batches that do not fit in memory at once. Each micro-batch
     * loss is divided by {@code microBatches}, so the step sees their mean.
     *
     * @param microBatches micro-batches per step
     * @return this loop
     */
    public TrainingLoop accumulate(int microBatches) {
        if (microBatches < 1) {
            throw new IllegalArgumentException("microBatches < 1");
        }
        this.accumulate = microBatches;
        return this;
    }

    /**
     * Scales the gradients down when their combined norm is larger than this, before each step.
     *
     * @param maxNorm the largest norm, such as {@code 1}
     * @return this loop
     */
    public TrainingLoop clipGradientNorm(float maxNorm) {
        this.clipNorm = maxNorm;
        return this;
    }

    /**
     * Sets the learning rate before each step from the step number (1 for the first step).
     *
     * @param schedule the schedule, given step numbers where it says epochs
     * @return this loop
     */
    public TrainingLoop schedule(LearningRateSchedule schedule) {
        this.schedule = schedule;
        return this;
    }

    /**
     * Keeps an exponential moving average of the weights, updated after each step.
     *
     * @param decay how much of the average each step keeps, such as {@code 0.999f}
     * @return this loop
     */
    public TrainingLoop ema(float decay) {
        this.ema = new Ema(params, decay);
        return this;
    }

    /** @return the averages, or {@code null} without {@link #ema} */
    public Ema ema() {
        return ema;
    }

    /** @return the optimizer steps taken so far */
    public int step() {
        return step;
    }

    /** @return whether the last {@link #microBatch} ended with an optimizer step */
    public boolean stepped() {
        return stepped;
    }

    /** @return the gradient norm before clipping at the last step, or NaN before any step or without clipping */
    public float lastGradientNorm() {
        return lastNorm == null ? Float.NaN : lastNorm.item();
    }

    /**
     * Trains on one micro-batch: computes its loss, back-propagates it, and after every
     * {@link #accumulate} micro-batches steps the optimizer. The values computed on the way
     * are freed afterwards, so the loss function must not keep them.
     *
     * @param loss computes the micro-batch's scalar loss
     * @return the loss
     */
    public float microBatch(Supplier<Var> loss) {
        current = loss;
        try {
            return run(loss);
        } finally {
            current = null;
        }
    }

    /** The micro-batch's loss, for optimizers that evaluate it again within a step. */
    private Supplier<Var> current;

    /**
     * Calls {@code hook} before each optimizer step, after clipping, with the parameters and
     * their gradients ({@link Var#grad}), for example to prune or regrow connections from
     * them or to change the gradients.
     *
     * @param hook receives the parameters
     * @return this loop
     */
    public TrainingLoop beforeStep(java.util.function.Consumer<List<Var>> hook) {
        this.beforeStep = hook;
        return this;
    }

    private float run(Supplier<Var> loss) {
        Var[] l = new Var[1];
        List<Tensor> work = Tensor.created(() -> l[0] = loss.get());
        Var scaled = accumulate == 1 ? l[0] : l[0].div(accumulate);
        Tensor value = l[0].value();
        scaled.backward();
        Tensor.evaluate(gradients());
        float out = value.item();
        scaled.release();
        List<Tensor> keep = new ArrayList<>(gradients());
        for (Var p : params) {
            keep.add(p.value());
        }
        Tensor.release(work, keep);
        micro++;
        stepped = false;
        if (micro == accumulate) {
            micro = 0;
            update();
            stepped = true;
        }
        return out;
    }

    /** The loss and gradients again at the current weights, for {@link Optimizer#step(List, Optimizer.Reevaluation)}. */
    private float reevaluate(Supplier<Var> loss) {
        for (Var p : params) {
            p.zeroGrad();
        }
        Var[] l = new Var[1];
        List<Tensor> work = Tensor.created(() -> l[0] = loss.get());
        l[0].backward();
        Tensor.evaluate(gradients());
        float out = l[0].value().item();
        l[0].release();
        List<Tensor> keep = new ArrayList<>(gradients());
        for (Var p : params) {
            keep.add(p.value());
        }
        Tensor.release(work, keep);
        return out;
    }

    private List<Tensor> gradients() {
        List<Tensor> g = new ArrayList<>(params.size());
        for (Var p : params) {
            if (p.grad() != null) {
                g.add(p.grad());
            }
        }
        return g;
    }

    private void update() {
        step++;
        if (schedule != null) {
            optimizer.learningRate(schedule.rate(step, baseRate));
        }
        if (clipNorm > 0) {
            clip();
        }
        List<Tensor> replaced = new ArrayList<>(params.size());
        for (Var p : params) {
            replaced.add(p.value());
        }
        if (beforeStep != null) {
            beforeStep.accept(params);
        }
        Supplier<Var> loss = current;
        if (accumulate == 1 && loss != null) {
            optimizer.step(params, () -> reevaluate(loss));
        } else {
            optimizer.step(params);
        }
        List<Tensor> kept = new ArrayList<>(params.size());
        for (Var p : params) {
            kept.add(p.value());
        }
        if (lastNorm != null) {
            kept.add(lastNorm);
        }
        Tensor.release(replaced, kept);
        if (ema != null) {
            ema.update();
        }
    }

    private void clip() {
        List<Tensor> grads = gradients();
        if (grads.isEmpty()) {
            return;
        }
        Tensor[] norm = new Tensor[1];
        List<Tensor> scaled = new ArrayList<>();
        List<Tensor> work = Tensor.created(() -> {
            Tensor sum = null;
            for (Tensor g : grads) {
                Tensor s = g.to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).square().sum();
                sum = sum == null ? s : sum.add(s);
            }
            norm[0] = sum.sqrt();
            // min(1, max / norm) without reading the norm back
            Tensor scale = norm[0].add(1e-6f).pow(-1).mul(clipNorm);
            scale = scale.minimum(Tensor.full(scale.shape(), 1, scale.dtype(), scale.device()));
            for (Var p : params) {
                if (p.grad() != null) {
                    Tensor g = p.grad().mul(scale.to(p.grad().dtype()));
                    p.setGrad(g);
                    scaled.add(g);
                }
            }
        });
        List<Tensor> keep = new ArrayList<>(scaled);
        keep.add(norm[0]);
        Tensor.evaluate(keep);
        work.addAll(grads);
        if (lastNorm != null) {
            work.add(lastNorm);
        }
        Tensor.release(work, keep);
        lastNorm = norm[0];
    }

    /**
     * Saves a checkpoint (see {@link Checkpoint}). Saving between the micro-batches of a step
     * loses that step's partial gradients.
     *
     * @param dir the directory
     */
    public void save(Path dir) {
        save(dir, Map.of());
    }

    /**
     * Saves a checkpoint with more values, such as where the data reader is.
     *
     * @param dir the directory
     * @param extra the values: strings, numbers or booleans
     */
    public void save(Path dir, Map<String, Object> extra) {
        Checkpoint.save(dir, parameters, optimizer, ema, step, extra);
    }

    /**
     * Continues from a checkpoint: weights, optimizer state, averages and step count.
     *
     * @param dir the directory {@link #save} wrote
     * @return the extra values saved with it
     */
    public Map<String, Object> resume(Path dir) {
        Checkpoint.Saved saved = Checkpoint.load(dir, parameters, optimizer, ema);
        step = saved.step();
        micro = 0;
        for (Var p : params) {
            p.zeroGrad();
        }
        if (schedule != null && step > 0) {
            optimizer.learningRate(schedule.rate(step, baseRate));
        }
        return saved.extra();
    }

    /**
     * Continues from a checkpoint if the directory has one.
     *
     * @param dir the directory
     * @return whether a checkpoint was read
     */
    public boolean resumeIfPresent(Path dir) {
        if (!java.nio.file.Files.isRegularFile(dir.resolve("trainer.json"))) {
            return false;
        }
        resume(dir);
        return true;
    }
}
