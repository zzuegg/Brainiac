package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.LearningMath;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Grokking (Power et al. 2022, <a href="https://arxiv.org/abs/2201.02177">arXiv:2201.02177</a>):
 * on modular addition {@code (a + b) mod p} with only part of the table for training, a
 * network first memorises the training pairs (train accuracy 100%, test at chance) and,
 * with strong weight decay, much later suddenly generalises. A cheap, reproducible probe
 * of training dynamics: memorisation versus the slow, decay-driven formation of a
 * general (Fourier) circuit. Inputs are the two one-hot operands side by side.
 *
 * <p>{@link #run(int, double, int, int, Optimizer, int, long)} takes any optimizer, so the
 * same curve measures Grokfast (amplified slow gradients) against AdamW with weight decay:
 * the step at which test accuracy first passes 0.99.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class GrokkingExperiment {

    private GrokkingExperiment() {
    }

    /**
     * The modular-addition table split into train and test pairs.
     *
     * @param p the modulus
     * @param trainFraction the share of the {@code p^2} pairs used for training
     * @param seed the seed of the split
     * @return {@code [trainX, trainIds, testX, testIds]}
     */
    public static Tensor[] modularAddition(int p, double trainFraction, long seed) {
        int n = p * p;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int train = (int) Math.round(trainFraction * n);
        return new Tensor[] {inputs(order, 0, train, p), ids(order, 0, train, p), inputs(order, train, n, p),
            ids(order, train, n, p)};
    }

    private static Tensor inputs(int[] order, int from, int to, int p) {
        float[] x = new float[(to - from) * 2 * p];
        for (int k = from; k < to; k++) {
            int a = order[k] / p;
            int b = order[k] % p;
            x[(k - from) * 2 * p + a] = 1;
            x[(k - from) * 2 * p + p + b] = 1;
        }
        return Tensor.of(Shape.of(to - from, 2 * p), x);
    }

    private static Tensor ids(int[] order, int from, int to, int p) {
        float[] y = new float[to - from];
        for (int k = from; k < to; k++) {
            y[k - from] = (order[k] / p + order[k] % p) % p;
        }
        return Tensor.of(Shape.of(to - from), y);
    }

    /**
     * The learning curve.
     *
     * @param steps the step of each point
     * @param trainAccuracy train accuracy at each point
     * @param testAccuracy test accuracy at each point
     * @param weightNorm the parameter norm at each point
     */
    public record Curve(List<Integer> steps, List<Double> trainAccuracy, List<Double> testAccuracy,
            List<Double> weightNorm) {

        /** @return the first step with 100% train accuracy, or -1 */
        public int memorisedAt() {
            for (int i = 0; i < steps.size(); i++) {
                if (trainAccuracy.get(i) >= 1) {
                    return steps.get(i);
                }
            }
            return -1;
        }

        /**
         * @param level a test accuracy, for example 0.95
         * @return the first step at which test accuracy reached it, or -1
         */
        public int generalisedAt(double level) {
            for (int i = 0; i < steps.size(); i++) {
                if (testAccuracy.get(i) >= level) {
                    return steps.get(i);
                }
            }
            return -1;
        }
    }

    /**
     * Trains full-batch with AdamW and records the curve.
     *
     * @param p the modulus, for example 23
     * @param trainFraction the training share, for example 0.5
     * @param hidden the hidden width
     * @param steps the training steps
     * @param learningRate AdamW's learning rate, for example {@code 3e-3f}
     * @param weightDecay AdamW's decay, large (for example 1) for grokking
     * @param logEvery steps between curve points
     * @param seed the seed
     * @return the learning curve
     */
    public static Curve run(int p, double trainFraction, int hidden, int steps, float learningRate,
            float weightDecay, int logEvery, long seed) {
        return run(p, trainFraction, hidden, steps,
                Optimizer.adamW(learningRate, 0.9f, 0.98f, 1e-8f, weightDecay), logEvery, seed);
    }

    /**
     * Trains full-batch with any optimizer, for example AdamW behind
     * {@link io.github.zzuegg.brainiac.experimental.train.Grokfast}, and records the curve.
     *
     * @param p the modulus
     * @param trainFraction the training share
     * @param hidden the hidden width
     * @param steps the training steps
     * @param opt the optimizer, new
     * @param logEvery steps between curve points
     * @param seed the seed
     * @return the learning curve
     */
    public static Curve run(int p, double trainFraction, int hidden, int steps, Optimizer opt, int logEvery,
            long seed) {
        Tensor[] d = modularAddition(p, trainFraction, seed);
        SimpleMlp net = SimpleMlp.relu(seed, 2 * p, hidden, p);
        Curve curve = new Curve(new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        for (int s = 1; s <= steps; s++) {
            Functional.crossEntropy(net.forward(Var.of(d[0])), d[1]).backward();
            opt.step(net.parameters());
            if (s % logEvery == 0 || s == steps) {
                curve.steps().add(s);
                curve.trainAccuracy().add(LearningMath.accuracy(net.predict(d[0]), d[1]));
                curve.testAccuracy().add(LearningMath.accuracy(net.predict(d[2]), d[3]));
                double norm = 0;
                for (Var v : net.parameters()) {
                    norm += v.value().square().sum().item();
                }
                curve.weightNorm().add(Math.sqrt(norm));
            }
        }
        return curve;
    }
}
