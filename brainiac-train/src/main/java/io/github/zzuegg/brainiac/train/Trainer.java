package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Trains a {@link Model} on a {@link Dataset}.
 *
 * <pre>{@code
 * TrainingResult result = Trainer.of(model)
 *         .loss(Loss.MSE)
 *         .optimizer(Optimizer.adam(0.01f))
 *         .batchSize(4)
 *         .epochs(2_000)
 *         .listener(Progress.every(100))
 *         .seed(42)
 *         .fit(data);
 * }</pre>
 *
 * <p>Defaults: MSE loss, Adam with learning rate 0.001, batch size 32, 100 epochs,
 * shuffling every epoch, no validation, no early stopping. Configuration methods return
 * this trainer; {@link #fit} may be called again to continue training.
 */
public final class Trainer {

    private final Model model;
    private Loss loss = Loss.MSE;
    private Optimizer optimizer;
    private int batchSize = 32;
    private int epochs = 100;
    private boolean shuffle = true;
    private io.github.zzuegg.brainiac.data.Augmentation augmentation;
    private long seed = 0;
    private Dataset validation;
    private int patience = -1;
    private final List<Metric> metrics = new ArrayList<>();
    private final List<TrainingListener> listeners = new ArrayList<>();
    private int epochsDone;
    private long stepsDone;
    private DType mixedType;
    private float lossScale = 1f;
    private int growthInterval;
    private int goodSteps;
    private int skippedSteps;
    private float l2;
    private float clipNorm;
    private float clipValue;
    private LearningRateSchedule schedule;
    private float baseLearningRate = Float.NaN;

    private Trainer(Model model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    /**
     * Starts configuring training for a model.
     *
     * @param model the model to train; its parameters are updated in place
     * @return a trainer with default settings
     */
    public static Trainer of(Model model) {
        return new Trainer(model);
    }

    /**
     * Sets the loss to minimize.
     *
     * @param loss the loss
     * @return this trainer
     */
    public Trainer loss(Loss loss) {
        this.loss = Objects.requireNonNull(loss, "loss");
        return this;
    }

    /**
     * Sets how parameters are updated.
     *
     * @param optimizer the optimizer
     * @return this trainer
     */
    public Trainer optimizer(Optimizer optimizer) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        return this;
    }

    /**
     * Sets how many samples are used per update.
     *
     * @param batchSize at least 1
     * @return this trainer
     */
    public Trainer batchSize(int batchSize) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1, got " + batchSize);
        }
        this.batchSize = batchSize;
        return this;
    }

    /**
     * Returns how many passes over the data {@link #fit} makes at most.
     *
     * @return the epochs
     */
    public int epochs() {
        return epochs;
    }

    /**
     * Sets how many passes over the data {@link #fit} makes at most.
     *
     * @param epochs at least 1
     * @return this trainer
     */
    public Trainer epochs(int epochs) {
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1, got " + epochs);
        }
        this.epochs = epochs;
        return this;
    }

    /**
     * Changes every training sample at random in every epoch, such as shifting and mirroring
     * images (see {@link io.github.zzuegg.brainiac.data.Augmentation#cropAndFlip}). Off by
     * default; validation and test data are never changed.
     *
     * @param augmentation the change, or null for none
     * @return this trainer
     */
    public Trainer augment(io.github.zzuegg.brainiac.data.Augmentation augmentation) {
        this.augmentation = augmentation;
        return this;
    }

    /**
     * Sets whether samples are reshuffled every epoch. On by default.
     *
     * @param shuffle whether to shuffle
     * @return this trainer
     */
    public Trainer shuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    /**
     * Sets the seed for shuffling, so runs are reproducible.
     *
     * @param seed the seed
     * @return this trainer
     */
    public Trainer seed(long seed) {
        this.seed = seed;
        return this;
    }

    /**
     * Adds metrics to compute each epoch.
     *
     * @param metrics the metrics
     * @return this trainer
     */
    public Trainer metric(Metric... metrics) {
        this.metrics.addAll(List.of(metrics));
        return this;
    }

    /**
     * Sets data to evaluate after every epoch without training on it.
     *
     * @param validation the validation data
     * @return this trainer
     */
    public Trainer validation(Dataset validation) {
        this.validation = Objects.requireNonNull(validation, "validation");
        return this;
    }

    /**
     * Stops when the loss (on validation data if given, else on training data) has not
     * improved for {@code patience} epochs, and restores the best weights.
     *
     * @param patience epochs to wait for an improvement
     * @return this trainer
     */
    public Trainer earlyStopping(int patience) {
        if (patience < 1) {
            throw new IllegalArgumentException("patience must be at least 1, got " + patience);
        }
        this.patience = patience;
        return this;
    }

    /**
     * Adds L2 regularization (weight decay): {@code lambda * sum(w^2)} over all weight
     * matrices (parameters with two or more dimensions; biases are not penalized) is added
     * to the loss. Keeps weights small and reduces overfitting.
     *
     * @param lambda the strength, for example {@code 1e-4f}
     * @return this trainer
     */
    public Trainer l2(float lambda) {
        if (lambda < 0) {
            throw new IllegalArgumentException("lambda must not be negative, got " + lambda);
        }
        this.l2 = lambda;
        return this;
    }

    /**
     * Changes the learning rate every epoch. The optimizer's learning rate at the first
     * {@link #fit} is the base rate the schedule scales.
     *
     * @param schedule the schedule, for example {@code LearningRateSchedule.cosine(100, 0.01f)}
     * @return this trainer
     */
    public Trainer schedule(LearningRateSchedule schedule) {
        this.schedule = Objects.requireNonNull(schedule, "schedule");
        return this;
    }

    /**
     * Rescales all gradients together whenever their global L2 norm exceeds
     * {@code maxNorm}, which keeps recurrent networks and transformers from diverging.
     *
     * @param maxNorm the largest allowed norm, positive
     * @return this trainer
     */
    public Trainer clipGradientNorm(float maxNorm) {
        if (!(maxNorm > 0)) {
            throw new IllegalArgumentException("maxNorm must be positive, got " + maxNorm);
        }
        this.clipNorm = maxNorm;
        return this;
    }

    /**
     * Clamps every gradient value to {@code [-limit, limit]}.
     *
     * @param limit the largest allowed magnitude, positive
     * @return this trainer
     */
    public Trainer clipGradientValue(float limit) {
        if (!(limit > 0)) {
            throw new IllegalArgumentException("limit must be positive, got " + limit);
        }
        this.clipValue = limit;
        return this;
    }

    /**
     * Puts the float32 master weights back and turns the scaled low-precision gradients
     * into float32 ones; adjusts the loss scale.
     *
     * @return whether the gradients are finite, so the step can be taken
     */
    private boolean unscale(List<Var> params, List<Tensor> masters) {
        Tensor check = null;
        for (int i = 0; i < params.size(); i++) {
            Var p = params.get(i);
            p.assign(masters.get(i));
            if (p.grad() != null) {
                Tensor g = p.grad().to(DType.FLOAT32).div(lossScale);
                p.setGrad(g);
                Tensor s = g.abs().sum();
                check = check == null ? s : check.add(s);
            }
        }
        if (check != null && !Float.isFinite(check.item())) {
            for (Var p : params) {
                p.zeroGrad();
            }
            lossScale = Math.max(lossScale / 2, Float.MIN_NORMAL);
            goodSteps = 0;
            skippedSteps++;
            return false;
        }
        if (growthInterval > 0 && ++goodSteps >= growthInterval) {
            lossScale = Math.min(lossScale * 2, 1 << 24);
            goodSteps = 0;
        }
        return true;
    }

    private void clip(List<Var> params) {
        if (clipValue > 0) {
            for (Var p : params) {
                if (p.grad() != null) {
                    Tensor lim = Tensor.full(Shape.of(), clipValue, p.grad().device());
                    p.setGrad(p.grad().minimum(lim).maximum(lim.neg()));
                }
            }
        }
        if (clipNorm > 0) {
            Tensor total = null;
            for (Var p : params) {
                if (p.grad() != null) {
                    Tensor sq = p.grad().square().sum();
                    total = total == null ? sq : total.add(sq);
                }
            }
            if (total == null) {
                return;
            }
            float norm = (float) Math.sqrt(total.item());
            if (norm > clipNorm) {
                float scale = clipNorm / norm;
                for (Var p : params) {
                    if (p.grad() != null) {
                        p.setGrad(p.grad().mul(scale));
                    }
                }
            }
        }
    }

    /**
     * Trains in mixed precision: every step runs the forward and backward pass with
     * weights converted to {@code type}, while the optimizer updates exact float32 master
     * weights. For {@code FLOAT16} the loss is multiplied by a scale so small gradients do
     * not round to zero; when a step's gradients overflow, the step is skipped and the
     * scale halved, and after {@code growthInterval} good steps in a row it doubles.
     * {@code BFLOAT16} has float32's range and trains with a scale of 1.
     *
     * @param type {@code FLOAT16} or {@code BFLOAT16}
     * @return this trainer
     */
    public Trainer mixedPrecision(DType type) {
        return mixedPrecision(type, type == DType.FLOAT16 ? 65536f : 1f, type == DType.FLOAT16 ? 2000 : 0);
    }

    /**
     * Trains in mixed precision with explicit loss scaling.
     *
     * @param type the type to compute in, such as {@code FLOAT16}
     * @param initialScale the first loss scale, positive
     * @param growthInterval good steps in a row before the scale doubles; 0 to keep it
     *     fixed except for halving on overflow
     * @return this trainer
     * @see #mixedPrecision(DType)
     */
    public Trainer mixedPrecision(DType type, float initialScale, int growthInterval) {
        if (type == DType.FLOAT32) {
            throw new IllegalArgumentException("mixed precision needs a type other than FLOAT32");
        }
        if (!(initialScale > 0) || growthInterval < 0) {
            throw new IllegalArgumentException("need a positive scale and a growth interval of at least 0");
        }
        this.mixedType = type;
        this.lossScale = initialScale;
        this.growthInterval = growthInterval;
        return this;
    }

    /** @return the current mixed-precision loss scale; 1 without mixed precision */
    public float lossScale() {
        return mixedType == null ? 1f : lossScale;
    }

    /** @return the steps skipped so far because mixed-precision gradients overflowed */
    public int skippedSteps() {
        return skippedSteps;
    }

    /**
     * Adds a listener called after every epoch.
     *
     * @param listener the listener
     * @return this trainer
     */
    public Trainer listener(TrainingListener listener) {
        this.listeners.add(Objects.requireNonNull(listener, "listener"));
        return this;
    }

    /**
     * Moves the model to a device; batches follow it automatically.
     *
     * @param device the device to train on
     * @return this trainer
     */
    public Trainer device(Device device) {
        model.to(device);
        return this;
    }

    /**
     * Trains the model. After each step, the memory of the values computed on the way and of
     * the parameter values it replaced is reused at once (see {@link Var#release}), so
     * layers and listeners must not keep tensors computed during a step, or a parameter
     * value, past that step.
     *
     * @param data the training data
     * @return the history and outcome
     */
    public TrainingResult fit(Dataset data) {
        checkSizes(data);
        if (validation != null) {
            checkSizes(validation);
        }
        if (optimizer == null) {
            optimizer = Optimizer.adam(0.001f);
        }
        Device device = model.device();
        List<Var> params = model.parameters();
        List<EpochResult> history = new ArrayList<>();
        List<Batch> fixedBatches = shuffle || augmentation != null ? null : data.batches(batchSize, device);

        float bestScore = Float.POSITIVE_INFINITY;
        int bestEpoch = 0;
        List<Tensor> bestWeights = null;
        boolean stoppedEarly = false;

        if (Float.isNaN(baseLearningRate)) {
            baseLearningRate = optimizer.learningRate();
        }
        for (int e = 1; e <= epochs; e++) {
            int epoch = ++epochsDone;
            if (schedule != null) {
                optimizer.learningRate(schedule.rate(epoch, baseLearningRate));
            }
            List<Batch> batches = augmentation != null
                    ? (shuffle ? data.shuffle(seed + epoch) : data).batches(batchSize, device, augmentation, seed + epoch)
                    : shuffle ? data.shuffle(seed + epoch).batches(batchSize, device) : fixedBatches;
            // per-batch values stay on the device until the epoch ends, so the host never
            // waits for the device between batches and can queue the next one meanwhile
            List<Tensor> batchValues = new ArrayList<>();
            List<Integer> batchSizes = new ArrayList<>();
            for (Batch batch : batches) {
                List<Tensor> masters = null;
                if (mixedType != null) {
                    masters = new ArrayList<>(params.size());
                    for (Var p : params) {
                        masters.add(p.value());
                        p.assign(p.value().to(mixedType));
                    }
                }
                Var[] forward = new Var[2];
                // what the forward pass computes, such as dropout masks, is only used this step
                List<Tensor> forwardWork = Tensor.created(() -> {
                    forward[0] = model.forward(Var.of(batch.features()), true);
                    forward[1] = loss.compute(forward[0], shaped(batch.targets(), forward[0]));
                    if (l2 > 0) {
                        Var penalty = l2Penalty(params);
                        if (penalty != null) {
                            forward[1] = forward[1].add(penalty.mul(l2));
                        }
                    }
                });
                Var out = forward[0];
                Var l = forward[1];
                Tensor targets = shaped(batch.targets(), out);
                int first = batchValues.size();
                batchValues.add(l.value());
                List<Tensor> metricWork = Tensor.created(() -> {
                    for (Metric m : metrics) {
                        batchValues.add(m.value(out.value(), targets));
                    }
                });
                Tensor.evaluate(batchValues.subList(first, batchValues.size()));
                releaseWork(metricWork, batchValues.subList(first, batchValues.size()), batch, targets);
                batchSizes.add(batch.size());
                if (masters != null) {
                    l.backward(Tensor.full(l.shape(), lossScale, l.dtype(), l.value().device()));
                    if (!unscale(params, masters)) {
                        continue;
                    }
                } else {
                    l.backward();
                }
                clip(params);
                List<Tensor> replaced = new ArrayList<>(params.size());
                if (masters == null) {   // (mixed precision keeps the masters itself)
                    for (Var p : params) {
                        replaced.add(p.value());
                    }
                }
                optimizer.step(params);
                // The step's intermediate values and the parameter values it replaced: their
                // memory is reused at once rather than after a garbage collection.
                List<Tensor> kept = new ArrayList<>(batchValues.subList(first, batchValues.size()));
                l.release(kept.toArray(Tensor[]::new));
                for (Var p : params) {
                    kept.add(p.value());
                }
                for (Var b : model.buffers()) {
                    kept.add(b.value());
                }
                if (bestWeights != null) {
                    kept.addAll(bestWeights);   // restored if later epochs are worse
                }
                kept.add(batch.features());
                kept.add(batch.targets());
                replaced.addAll(forwardWork);
                Tensor.release(replaced, kept);
                stepsDone++;
                if (!listeners.isEmpty()) {
                    float batchLoss = l.value().item();
                    for (TrainingListener listener : listeners) {
                        listener.onBatch(epoch, stepsDone, batchLoss);
                    }
                }
            }
            double lossSum = 0;
            Map<String, Double> metricSums = new LinkedHashMap<>();
            int perBatch = 1 + metrics.size();
            for (int b = 0; b < batchSizes.size(); b++) {
                int size = batchSizes.get(b);
                lossSum += batchValues.get(b * perBatch).item() * size;
                for (int i = 0; i < metrics.size(); i++) {
                    metricSums.merge(metrics.get(i).name(), (double) batchValues.get(b * perBatch + 1 + i).item() * size,
                            Double::sum);
                }
            }
            Tensor.release(batchValues, List.of());
            if (shuffle || augmentation != null) {
                release(batches);   // this epoch's own copies of the data
            }
            Map<String, Float> trainMetrics = new LinkedHashMap<>();
            metricSums.forEach((k, v) -> trainMetrics.put(k, (float) (v / data.size())));

            float valLoss = Float.NaN;
            Map<String, Float> valMetrics = Map.of();
            if (validation != null) {
                Evaluation ev = evaluate(validation);
                valLoss = ev.loss();
                valMetrics = ev.metrics();
            }
            EpochResult result = new EpochResult(epoch, (float) (lossSum / data.size()),
                    Collections.unmodifiableMap(trainMetrics), valLoss, valMetrics, optimizer.learningRate());
            history.add(result);
            for (TrainingListener listener : listeners) {
                listener.onEpoch(result);
            }

            float score = validation != null ? valLoss : result.loss();
            if (score < bestScore) {
                bestScore = score;
                bestEpoch = history.size();
                if (patience > 0) {
                    bestWeights = params.stream().map(Var::value).toList();
                }
            } else if (patience > 0 && history.size() - bestEpoch >= patience) {
                stoppedEarly = true;
                break;
            }
        }
        if (bestWeights != null && bestEpoch != history.size()) {
            for (int i = 0; i < params.size(); i++) {
                params.get(i).assign(bestWeights.get(i));
            }
        } else {
            bestEpoch = history.size();
        }
        return new TrainingResult(List.copyOf(history), bestEpoch, stoppedEarly);
    }

    /** Releases the values computed on the way to a batch's metrics. */
    private static void releaseWork(List<Tensor> work, List<Tensor> results, Batch batch, Tensor targets) {
        List<Tensor> kept = new ArrayList<>(results);
        kept.add(batch.features());
        kept.add(batch.targets());
        kept.add(targets);
        Tensor.release(work, kept);
    }

    private static void release(List<Batch> batches) {
        List<Tensor> values = new ArrayList<>(2 * batches.size());
        for (Batch b : batches) {
            values.add(b.features());
            values.add(b.targets());
        }
        Tensor.release(values, List.of());
    }

    /**
     * Computes the loss and the configured metrics on a dataset without training.
     *
     * @param data the data
     * @return the evaluation
     */
    public Evaluation evaluate(Dataset data) {
        checkSizes(data);
        Device device = model.device();
        List<Tensor> values = new ArrayList<>();
        List<Integer> sizes = new ArrayList<>();
        List<Batch> batches = data.batches(Math.max(batchSize, 256), device);
        for (Batch batch : batches) {
            Var out = model.forward(Var.of(batch.features()));
            Tensor targets = shaped(batch.targets(), out);
            int first = values.size();
            Var l = loss.compute(out, targets);
            values.add(l.value());
            List<Tensor> metricWork = Tensor.created(() -> {
                for (Metric m : metrics) {
                    values.add(m.value(out.value(), targets));
                }
            });
            Tensor.evaluate(values.subList(first, values.size()));   // no waiting between batches
            List<Tensor> batchResults = values.subList(first, values.size());
            l.release(batchResults.toArray(Tensor[]::new));   // the values on the way
            releaseWork(metricWork, batchResults, batch, targets);
            sizes.add(batch.size());
        }
        double lossSum = 0;
        Map<String, Double> sums = new LinkedHashMap<>();
        int perBatch = 1 + metrics.size();
        for (int b = 0; b < sizes.size(); b++) {
            lossSum += values.get(b * perBatch).item() * sizes.get(b);
            for (int i = 0; i < metrics.size(); i++) {
                sums.merge(metrics.get(i).name(), (double) values.get(b * perBatch + 1 + i).item() * sizes.get(b),
                        Double::sum);
            }
        }
        Tensor.release(values, List.of());
        release(batches);
        Map<String, Float> result = new LinkedHashMap<>();
        sums.forEach((k, v) -> result.put(k, (float) (v / data.size())));
        return new Evaluation((float) (lossSum / data.size()), Collections.unmodifiableMap(result));
    }

    /** Flat target rows take the model's output shape, such as one target per time step. */
    private static Tensor shaped(Tensor targets, Var out) {
        Shape s = out.shape();
        return s.equals(targets.shape()) || s.size() != targets.size() ? targets : targets.reshape(s);
    }

    private static Var l2Penalty(List<Var> params) {
        Var sum = null;
        for (Var p : params) {
            if (p.shape().rank() >= 2) {
                Var s = p.square().sum();
                sum = sum == null ? s : sum.add(s);
            }
        }
        return sum;
    }

    private void checkSizes(Dataset data) {
        int classes = model.outputShape().dim(-1);
        boolean ids = classes > 1 && (long) data.targetCount() * classes == model.outputSize();
        if (data.featureCount() != model.inputSize() || (data.targetCount() != model.outputSize() && !ids)) {
            throw new IllegalArgumentException("model maps " + model.inputSize() + " inputs to " + model.outputSize()
                    + " outputs, but the data has " + data.featureCount() + " features and "
                    + data.targetCount() + " targets");
        }
    }
}
