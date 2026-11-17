package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.BatchSizeSchedule;
import io.github.zzuegg.brainiac.experimental.train.CheckpointAveraging;
import io.github.zzuegg.brainiac.experimental.train.SpikeGuard;
import io.github.zzuegg.brainiac.experimental.train.StochasticRounding;
import io.github.zzuegg.brainiac.experimental.train.StochasticWeightAveraging;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Ema;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Training-dynamics techniques against their usual baselines, each on a small task where
 * its effect shows:
 *
 * <ul>
 *   <li>{@link #spikes}: regression with injected outlier batches (targets 100x too large),
 *       {@link SpikeGuard} around AdamW against AdamW with global norm clipping.</li>
 *   <li>{@link #averaging}: held-out loss along a high-learning-rate run of the raw
 *       weights, {@link CheckpointAveraging} (LAWA), the core {@code Ema} and
 *       {@link StochasticWeightAveraging}.</li>
 *   <li>{@link #batchRamp}: at an equal example budget, a growing batch with a constant
 *       learning rate against a fixed batch with a cosine schedule.</li>
 *   <li>{@link #bf16}: training with weights kept in bfloat16, rounded to nearest or
 *       stochastically ({@link StochasticRounding}), against float32 weights.</li>
 * </ul>
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class TrainingDynamicsExperiment {

    private TrainingDynamicsExperiment() {
    }

    /**
     * The outcome of {@link #spikes}.
     *
     * @param finalMse the clean held-out mean squared error at the end
     * @param skipped batches the guard skipped (0 for the baseline)
     * @param outliers outlier batches injected
     */
    public record SpikeRun(float finalMse, int skipped, int outliers) {
    }

    /**
     * Trains a tanh MLP regression where every {@code outlierEvery}-th batch has targets
     * scaled by 100.
     *
     * @param guarded whether to use {@link SpikeGuard} with the same norm clipping (else
     *     AdamW with norm clipping at 1)
     * @param steps training steps
     * @param outlierEvery steps between outlier batches; 0 for none
     * @param seed the seed
     * @return the run
     */
    public static SpikeRun spikes(boolean guarded, int steps, int outlierEvery, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor teacher = Tensor.normal(Shape.of(8, 1), 0, 1, r);
        SimpleMlp net = SimpleMlp.tanh(seed, 8, 32, 1);
        Optimizer adamW = Optimizer.adamW(3e-3f, 0.01f);
        SpikeGuard guard = guarded ? SpikeGuard.of(adamW, 1f) : null;
        int outliers = 0;
        for (int s = 1; s <= steps; s++) {
            Tensor x = Tensor.normal(Shape.of(32, 8), 0, 1, r);
            Tensor y = x.matmul(teacher).tanh();
            if (outlierEvery > 0 && s % outlierEvery == 0) {
                y = y.mul(100);
                outliers++;
            }
            net.forward(Var.of(x)).sub(y).square().mean().backward();
            if (guarded) {
                guard.step(net.parameters());
            } else {
                SpikeGuard.clipGlobalNorm(net.parameters(), 1f);
                adamW.step(net.parameters());
            }
        }
        Tensor xt = Tensor.normal(Shape.of(1024, 8), 0, 1, new SplittableRandom(seed + 1));
        float mse = net.predict(xt).sub(xt.matmul(teacher).tanh()).square().mean().item();
        return new SpikeRun(mse, guarded ? guard.skipped() : 0, outliers);
    }

    /**
     * Held-out losses along one run, for the raw weights and each average.
     *
     * @param steps the step of each checkpoint
     * @param raw the raw weights' loss
     * @param lawa the loss of the last-k checkpoint average
     * @param ema the core {@code Ema}'s loss
     * @param swa the loss of the uniform average since the start
     */
    public record AveragingCurves(int[] steps, float[] raw, float[] lawa, float[] ema, float[] swa) {
    }

    /**
     * Trains the Markov-source MLP with Adam at a high constant learning rate and
     * evaluates every {@code evalEvery} steps.
     *
     * @param steps training steps
     * @param evalEvery steps between evaluations
     * @param seed the seed
     * @return the curves
     */
    public static AveragingCurves averaging(int steps, int evalEvery, long seed) {
        MarkovSource source = MarkovSource.random(10, 1);
        SimpleMlp net = SimpleMlp.relu(seed, 20, 64, 64, 10);
        List<Var> params = net.parameters();
        Optimizer opt = Optimizer.adam(2e-2f);
        CheckpointAveraging lawa = new CheckpointAveraging(params, 6, 10);
        Ema ema = new Ema(params, 0.98f);
        StochasticWeightAveraging swa = new StochasticWeightAveraging(params, 10, 10);
        Tensor[] test = source.batch(2048, new SplittableRandom(seed + 5));
        SplittableRandom r = new SplittableRandom(seed);
        int points = steps / evalEvery;
        AveragingCurves c = new AveragingCurves(new int[points], new float[points], new float[points],
                new float[points], new float[points]);
        for (int s = 1; s <= steps; s++) {
            Tensor[] b = source.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(params);
            lawa.onBatch(1, s, 0);
            ema.update();
            swa.onBatch(1, s, 0);
            if (s % evalEvery == 0) {
                int i = s / evalEvery - 1;
                c.steps()[i] = s;
                c.raw()[i] = loss(net, test);
                lawa.swap();
                c.lawa()[i] = loss(net, test);
                lawa.swap();
                ema.swap();
                c.ema()[i] = loss(net, test);
                ema.swap();
                swa.swap();
                c.swa()[i] = loss(net, test);
                swa.swap();
            }
        }
        return c;
    }

    private static float loss(SimpleMlp net, Tensor[] test) {
        return Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value().item();
    }

    /**
     * The outcome of {@link #batchRamp}.
     *
     * @param finalLoss the held-out loss at the end
     * @param steps optimizer steps taken
     */
    public record BudgetRun(float finalLoss, int steps) {
    }

    /**
     * Trains the Markov-source MLP on exactly {@code budget} examples.
     *
     * @param schedule the batch sizes
     * @param cosine whether to decay the learning rate with a cosine (else constant)
     * @param learningRate Adam's (peak) learning rate
     * @param budget the examples to train on
     * @param seed the seed
     * @return the run
     */
    public static BudgetRun batchRamp(BatchSizeSchedule schedule, boolean cosine, float learningRate, long budget,
            long seed) {
        MarkovSource source = MarkovSource.random(10, 1);
        SimpleMlp net = SimpleMlp.relu(seed, 20, 64, 64, 10);
        int[] plan = schedule.plan(budget);
        Optimizer opt = Optimizer.adam(learningRate);
        LearningRateSchedule decay = LearningRateSchedule.cosine(plan.length, 0f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < plan.length; s++) {
            if (cosine) {
                opt.learningRate(Math.max(1e-6f, decay.rate(s + 1, learningRate)));
            }
            Tensor[] b = source.batch(plan[s], r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(net.parameters());
        }
        return new BudgetRun(loss(net, source.batch(4096, new SplittableRandom(seed + 5))), plan.length);
    }

    /** How the weights are stored in {@link #bf16}. */
    public enum Precision {
        /** float32 weights, the reference */
        FLOAT32,
        /** bfloat16 weights, round to nearest after each step */
        BF16_NEAREST,
        /** bfloat16 weights, stochastic rounding after each step */
        BF16_STOCHASTIC
    }

    /**
     * Fits a linear regression {@code y = x W*} (16 inputs, 4 outputs, {@code |W*| ~ 1})
     * with Adam at a small learning rate, so late updates are smaller than half of
     * bfloat16's spacing near the weights.
     *
     * @param precision how weights are stored
     * @param learningRate Adam's learning rate, for example {@code 1e-3f}
     * @param steps training steps
     * @param seed the seed
     * @return the final held-out mean squared error
     */
    public static float bf16(Precision precision, float learningRate, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor teacher = Tensor.normal(Shape.of(16, 4), 1, 0.5f, r);
        Var w = Var.param(StochasticRounding.nearestBf16(Tensor.normal(Shape.of(16, 4), 1, 0.5f, r)));
        Optimizer adam = Optimizer.adam(learningRate);
        Optimizer opt = switch (precision) {
            case FLOAT32 -> adam;
            case BF16_NEAREST -> StochasticRounding.bf16Weights(adam, false, seed);
            case BF16_STOCHASTIC -> StochasticRounding.bf16Weights(adam, true, seed);
        };
        for (int s = 0; s < steps; s++) {
            Tensor x = Tensor.normal(Shape.of(32, 16), 0, 1, r);
            Var.of(x).matmul(w).sub(x.matmul(teacher)).square().mean().backward();
            opt.step(List.of(w));
        }
        Tensor xt = Tensor.normal(Shape.of(512, 16), 0, 1, new SplittableRandom(seed + 1));
        return xt.matmul(w.value()).sub(xt.matmul(teacher)).square().mean().item();
    }
}
