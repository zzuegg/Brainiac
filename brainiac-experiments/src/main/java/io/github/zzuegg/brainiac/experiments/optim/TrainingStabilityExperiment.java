package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.AdamC;
import io.github.zzuegg.brainiac.experimental.train.CautiousWeightDecay;
import io.github.zzuegg.brainiac.experimental.train.CheckpointAveraging;
import io.github.zzuegg.brainiac.experimental.train.SpikeGuard;
import io.github.zzuegg.brainiac.experimental.train.TrainingSchedules;
import io.github.zzuegg.brainiac.experimental.train.WarmupStableMerge;
import io.github.zzuegg.brainiac.experimental.train.ZClip;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Weight decay, clipping and schedule techniques of 2025 against their usual baselines, on
 * the Markov next-token source of {@link ModernOptimizerExperiment} (or a regression with
 * outlier batches for clipping):
 *
 * <ul>
 *   <li>{@link #cautiousDecay}: AdamW with {@link CautiousWeightDecay} against standard
 *       decoupled decay, final loss and weight norm.</li>
 *   <li>{@link #correctedDecay}: {@link AdamC} against AdamW under a cosine schedule on a
 *       network whose hidden layers are normalised, the gradient-norm curve (and its rise
 *       at the end) and the final loss.</li>
 *   <li>{@link #clipping}: {@link ZClip} against no clipping, fixed-norm clipping and
 *       {@link SpikeGuard} at a high learning rate with outlier batches: loss spikes and
 *       final error.</li>
 *   <li>{@link #schedule}: warmup-stable-merge ({@link WarmupStableMerge}) against WSD
 *       decay, cosine and {@link CheckpointAveraging} (LAWA) at equal steps.</li>
 * </ul>
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class TrainingStabilityExperiment {

    /** The task. */
    public static final MarkovSource SOURCE = MarkovSource.random(10, 1);

    private TrainingStabilityExperiment() {
    }

    private static float heldOut(SimpleMlp net, Tensor[] test) {
        return Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value().item();
    }

    private static Tensor[] testSet(long seed) {
        return SOURCE.batch(2048, new SplittableRandom(seed ^ 0x5DEECE66DL));
    }

    /**
     * The outcome of {@link #cautiousDecay}.
     *
     * @param loss the final held-out loss
     * @param weightNorm the Frobenius norm of all weight matrices at the end
     */
    public record DecayRun(float loss, double weightNorm) {
    }

    /**
     * Trains the Markov MLP with AdamW's update and a strong decoupled weight decay on the
     * matrices, plain or cautious.
     *
     * @param cautious whether the decay is cautious
     * @param learningRate Adam's rate
     * @param weightDecay the decay
     * @param steps training steps
     * @param seed the seed
     * @return the run
     */
    public static DecayRun cautiousDecay(boolean cautious, float learningRate, float weightDecay, int steps,
            long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, 20, 64, 64, 10);
        Optimizer matrices = cautious ? CautiousWeightDecay.adamW(learningRate, weightDecay)
                : Optimizer.adamW(learningRate, weightDecay);
        Optimizer opt = Optimizer.groups(Optimizer.adam(learningRate), Optimizer.Group.matrices(matrices));
        SplittableRandom r = new SplittableRandom(seed + 17);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = SOURCE.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(net.parameters());
        }
        double norm = 0;
        for (int l = 0; l < net.depth(); l++) {
            norm += net.weight(l).value().square().sum().item();
        }
        return new DecayRun(heldOut(net, testSet(seed)), Math.sqrt(norm));
    }

    /**
     * Layer normalisation without gains, then ReLU: makes the preceding weight matrix
     * scale-invariant.
     *
     * @param h the pre-activations
     * @return the normalised activations
     */
    static Var normRelu(Var h) {
        Var c = h.sub(h.mean(1, true));
        return c.div(c.square().mean(1, true).add(1e-5f).sqrt()).relu();
    }

    /**
     * The outcome of {@link #correctedDecay}.
     *
     * @param gradNorm the gradient norm of the normalised matrices, averaged over each tenth of training
     * @param loss the final held-out loss
     */
    public record GradNormRun(double[] gradNorm, float loss) {

        /** @return the gradient norm in the last tenth over its minimum over the earlier tenths after the first */
        public double endRise() {
            double min = Double.POSITIVE_INFINITY;
            for (int i = 1; i < gradNorm.length - 1; i++) {
                min = Math.min(min, gradNorm[i]);
            }
            return gradNorm[gradNorm.length - 1] / min;
        }
    }

    /**
     * Trains a network whose two hidden matrices are followed by layer norms with a cosine
     * learning-rate schedule (to 0) and a strong weight decay.
     *
     * @param corrected whether to use {@link AdamC} (else AdamW)
     * @param learningRate the peak rate
     * @param weightDecay the weight decay
     * @param steps training steps, a multiple of 10
     * @param seed the seed
     * @return the run
     */
    public static GradNormRun correctedDecay(boolean corrected, float learningRate, float weightDecay, int steps,
            long seed) {
        SimpleMlp net = SimpleMlp.of(seed, TrainingStabilityExperiment::normRelu, 20, 64, 64, 10);
        List<Var> normalized = List.of(net.weight(0), net.weight(1));
        Optimizer opt = corrected
                ? new AdamC(learningRate, 0.9f, 0.999f, 1e-8f, weightDecay, normalized::contains)
                : Optimizer.adamW(learningRate, weightDecay);
        LearningRateSchedule cosine = LearningRateSchedule.cosine(steps, 0f);
        SplittableRandom r = new SplittableRandom(seed + 17);
        double[] curve = new double[10];
        for (int s = 0; s < steps; s++) {
            opt.learningRate(Math.max(1e-7f, cosine.rate(s + 1, learningRate)));
            Tensor[] b = SOURCE.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            curve[s * 10 / steps] += ZClip.globalNorm(normalized) / (steps / 10.0);
            opt.step(net.parameters());
        }
        return new GradNormRun(curve, heldOut(net, testSet(seed)));
    }

    /** Gradient clipping methods compared in {@link #clipping}. */
    public enum Clipping {
        /** AdamW without clipping. */
        NONE,
        /** Global-norm clipping at 1. */
        FIXED_NORM,
        /** {@link SpikeGuard}: skip spike batches, clip the rest at norm 1. */
        SPIKE_GUARD,
        /** {@link ZClip}: z-score adaptive clipping. */
        ZCLIP
    }

    /**
     * The outcome of {@link #clipping}.
     *
     * @param finalMse the clean held-out error at the end
     * @param spikes evaluations whose held-out error jumped above twice its running minimum
     */
    public record ClipRun(float finalMse, int spikes) {
    }

    /**
     * A tanh MLP regression trained with AdamW at a high learning rate, where every
     * {@code outlierEvery}-th batch has inputs scaled by 30 and targets by 10 (a corrupt
     * batch), with held-out error measured every 5 steps.
     *
     * @param clipping the method
     * @param learningRate AdamW's rate
     * @param steps training steps
     * @param outlierEvery steps between outlier batches
     * @param seed the seed
     * @return the run
     */
    public static ClipRun clipping(Clipping clipping, float learningRate, int steps, int outlierEvery, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor teacher = Tensor.normal(Shape.of(8, 1), 0, 1, r);
        SimpleMlp net = SimpleMlp.tanh(seed, 8, 32, 32, 1);
        Optimizer adamW = Optimizer.adamW(learningRate, 0.01f);
        Optimizer opt = switch (clipping) {
            case NONE, FIXED_NORM -> adamW;
            case SPIKE_GUARD -> SpikeGuard.of(adamW, 1f);
            case ZCLIP -> ZClip.wrap(adamW);
        };
        Tensor xt = Tensor.normal(Shape.of(256, 8), 0, 1, new SplittableRandom(seed + 1));
        Tensor yt = xt.matmul(teacher).tanh();
        float best = Float.POSITIVE_INFINITY;
        int spikes = 0;
        float mse = Float.NaN;
        for (int s = 1; s <= steps; s++) {
            Tensor x = Tensor.normal(Shape.of(32, 8), 0, 1, r);
            Tensor y = x.matmul(teacher).tanh();
            if (s % outlierEvery == 0) {
                x = x.mul(30);
                y = y.mul(10);
            }
            net.forward(Var.of(x)).sub(y).square().mean().backward();
            if (clipping == Clipping.FIXED_NORM) {
                SpikeGuard.clipGlobalNorm(net.parameters(), 1f);
            }
            opt.step(net.parameters());
            if (s % 5 == 0) {
                mse = net.predict(xt).sub(yt).square().mean().item();
                if (!Float.isFinite(mse)) {
                    return new ClipRun(Float.POSITIVE_INFINITY, spikes + 1);
                }
                if (s > steps / 5 && mse > 2 * best) {
                    spikes++;
                }
                best = Math.min(best, mse);
            }
        }
        return new ClipRun(mse, spikes);
    }

    /** Learning-rate schedules compared in {@link #schedule}. */
    public enum Schedule {
        /** Warm-up, constant, linear decay over the last 20% (WSD). */
        WSD,
        /** Warm-up then cosine decay to 0. */
        COSINE,
        /** Constant rate after warm-up, raw weights. */
        CONSTANT,
        /** Constant rate, uniform average of the last checkpoints (LAWA). */
        LAWA,
        /** Constant rate, checkpoints merged with "1 - sqrt" decay weights (WSM). */
        WSM
    }

    /**
     * Trains the Markov MLP with Adam under a schedule.
     *
     * @param schedule the schedule
     * @param learningRate the peak rate
     * @param steps training steps (a multiple of 50)
     * @param seed the seed
     * @return the final held-out loss
     */
    public static float schedule(Schedule schedule, float learningRate, int steps, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, 20, 64, 64, 10);
        Optimizer opt = Optimizer.adam(learningRate);
        int warmup = steps / 20;
        int decay = steps / 5;
        LearningRateSchedule lr = switch (schedule) {
            case WSD -> TrainingSchedules.warmupStableDecay(warmup, steps - warmup - decay, decay, 0f, false);
            case COSINE -> (step, base) -> step <= warmup ? base * step / warmup
                    : LearningRateSchedule.cosine(steps - warmup, 0f).rate(step - warmup, base);
            default -> WarmupStableMerge.schedule(warmup);
        };
        int every = decay / 10;   // ten checkpoints over the span WSD decays in
        CheckpointAveraging lawa = new CheckpointAveraging(net.parameters(), 10, every);
        WarmupStableMerge wsm = new WarmupStableMerge(net.parameters(), 10, every,
                WarmupStableMerge.Shape.ONE_MINUS_SQRT);
        SplittableRandom r = new SplittableRandom(seed + 17);
        for (int s = 1; s <= steps; s++) {
            opt.learningRate(Math.max(1e-7f, lr.rate(s, learningRate)));
            Tensor[] b = SOURCE.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(net.parameters());
            lawa.onBatch(1, s, 0);
            wsm.onBatch(1, s, 0);
        }
        if (schedule == Schedule.LAWA) {
            lawa.swap();
        } else if (schedule == Schedule.WSM) {
            wsm.swap();
        }
        return heldOut(net, testSet(seed));
    }
}
