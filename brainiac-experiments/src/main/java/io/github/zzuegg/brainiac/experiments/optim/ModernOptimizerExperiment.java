package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.Adopt;
import io.github.zzuegg.brainiac.experimental.train.AdamMini;
import io.github.zzuegg.brainiac.experimental.train.Cautious;
import io.github.zzuegg.brainiac.experimental.train.Mars;
import io.github.zzuegg.brainiac.experimental.train.PsgdKron;
import io.github.zzuegg.brainiac.experimental.train.StableAdamW;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Compares the 2024-2025 optimizers against AdamW, Lion and Muon on a small next-token
 * task: a random second-order Markov source over a small vocabulary, learned by an MLP
 * from the two previous tokens (one-hot), trained on fresh mini-batches so gradient noise
 * matters. Held-out cross-entropy is measured against the source's own entropy, the best
 * any model can do.
 *
 * <ul>
 *   <li>{@link Cautious} (C-AdamW, C-Lion): loss at a fixed step budget against plain
 *       AdamW and Lion.</li>
 *   <li>{@link Mars}: steps to a target loss against AdamW.</li>
 *   <li>{@link Adopt}: final loss over a {@code beta2} sweep against Adam.</li>
 *   <li>{@link AdamMini}: loss and optimizer-state bytes against AdamW and the core
 *       8-bit Adam.</li>
 *   <li>{@link StableAdamW} and Adam-atan2: loss and divergence across seeds after a sudden
 *       jump in gradient size, at a moderate and a too-high learning rate ({@link #stress}).</li>
 *   <li>{@link Adopt} also on the online counterexample where Adam fails
 *       ({@link #counterexample}).</li>
 *   <li>{@link PsgdKron}: loss per step and per second against the Shampoo, SOAP and Muon
 *       baselines.</li>
 * </ul>
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ModernOptimizerExperiment {

    private ModernOptimizerExperiment() {
    }

    /**
     * A random second-order Markov source: each pair of previous tokens has a peaked
     * next-token distribution (two or three likely tokens).
     *
     * @param vocab the vocabulary size
     * @param table the next-token probabilities, {@code [vocab * vocab][vocab]}
     */
    public record MarkovSource(int vocab, double[][] table) {

        /**
         * @param vocab the vocabulary size, for example 10
         * @param seed the seed
         * @return a random source
         */
        public static MarkovSource random(int vocab, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            double[][] t = new double[vocab * vocab][vocab];
            for (double[] row : t) {
                double sum = 0;
                for (int k = 0; k < vocab; k++) {
                    row[k] = 0.02;
                    sum += row[k];
                }
                int likely = 2 + r.nextInt(2);
                for (int k = 0; k < likely; k++) {
                    double w = 1 + 3 * r.nextDouble();
                    row[r.nextInt(vocab)] += w;
                    sum += w;
                }
                for (int k = 0; k < vocab; k++) {
                    row[k] /= sum;
                }
            }
            return new MarkovSource(vocab, t);
        }

        /** @return the mean entropy of the next token in nats: the lowest reachable loss */
        public double entropy() {
            double h = 0;
            for (double[] row : table) {
                for (double p : row) {
                    h -= p > 0 ? p * Math.log(p) : 0;
                }
            }
            return h / table.length;
        }

        /**
         * Samples contexts uniformly and their next tokens.
         *
         * @param n examples
         * @param r the random source
         * @return {@code {one-hot contexts [n, 2 * vocab], next ids [n]}}
         */
        public Tensor[] batch(int n, SplittableRandom r) {
            float[] x = new float[n * 2 * vocab];
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                int a = r.nextInt(vocab);
                int b = r.nextInt(vocab);
                x[i * 2 * vocab + a] = 1;
                x[i * 2 * vocab + vocab + b] = 1;
                double u = r.nextDouble();
                double[] row = table[a * vocab + b];
                int k = 0;
                while (k < vocab - 1 && (u -= row[k]) > 0) {
                    k++;
                }
                y[i] = k;
            }
            return new Tensor[] {Tensor.of(Shape.of(n, 2 * vocab), x), Tensor.of(Shape.of(n), y)};
        }
    }

    /**
     * The held-out loss of a training run.
     *
     * @param lossCurve held-out cross-entropy after every {@code evalEvery} steps
     * @param seconds the wall-clock training time
     * @param stateBytes the optimizer's state at the end, in bytes
     */
    public record Run(float[] lossCurve, double seconds, long stateBytes) {

        /** @return the last held-out loss */
        public float finalLoss() {
            return lossCurve[lossCurve.length - 1];
        }

        /** @return the lowest held-out loss */
        public float bestLoss() {
            float best = Float.POSITIVE_INFINITY;
            for (float l : lossCurve) {
                best = Math.min(best, l);
            }
            return best;
        }

        /**
         * @param target a loss
         * @param evalEvery the steps between curve points
         * @return the first step whose loss is at most {@code target}, or -1
         */
        public int stepsTo(float target, int evalEvery) {
            for (int i = 0; i < lossCurve.length; i++) {
                if (lossCurve[i] <= target) {
                    return (i + 1) * evalEvery;
                }
            }
            return -1;
        }

        /** @return whether the run blew up (a non-finite loss) */
        public boolean diverged() {
            for (float l : lossCurve) {
                if (!Float.isFinite(l)) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Trains a ReLU MLP {@code [2 * vocab, hidden, hidden, vocab]} on the source.
     *
     * @param optimizer makes the optimizer
     * @param source the source
     * @param hidden the hidden width
     * @param steps training steps
     * @param batch the mini-batch size
     * @param evalEvery steps between held-out evaluations
     * @param seed the seed of the weights and batches
     * @return the run
     */
    public static Run train(Supplier<Optimizer> optimizer, MarkovSource source, int hidden, int steps, int batch,
            int evalEvery, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, 2 * source.vocab(), hidden, hidden, source.vocab());
        Optimizer opt = optimizer.get();
        Tensor[] test = source.batch(2048, new SplittableRandom(seed ^ 0x5DEECE66DL));
        SplittableRandom r = new SplittableRandom(seed + 17);
        float[] curve = new float[steps / evalEvery];
        long start = System.nanoTime();
        double evalNanos = 0;
        for (int s = 1; s <= steps; s++) {
            Tensor[] b = source.batch(batch, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(net.parameters());
            if (s % evalEvery == 0) {
                long e0 = System.nanoTime();
                curve[s / evalEvery - 1] = Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value()
                        .item();
                evalNanos += System.nanoTime() - e0;
            }
        }
        double seconds = (System.nanoTime() - start - evalNanos) / 1e9;
        return new Run(curve, seconds, AdamMini.stateBytes(opt, net.parameters()));
    }

    /**
     * The high-learning-rate stress test with a stale second moment: a deep, narrow tanh
     * MLP on the source whose inputs are scaled down by {@code jump} for the first
     * {@code quiet} steps and then jump to full size, so gradients suddenly grow while
     * Adam's {@code v} (decay 0.999) still remembers the small ones, the situation
     * Wortsman et al. trace loss spikes to. A run diverges when its held-out loss becomes
     * non-finite or ends above the loss of predicting uniformly.
     *
     * @param optimizer makes the optimizer for a learning rate
     * @param learningRate the learning rate
     * @param seeds the seeds to try
     * @param quiet steps with small inputs
     * @param jump how much smaller the inputs are at first
     * @param steps steps per run
     * @return what happened over the seeds
     */
    public static Stress stress(Function<Float, Optimizer> optimizer, float learningRate, List<Long> seeds,
            int quiet, float jump, int steps) {
        MarkovSource source = MarkovSource.random(8, 3);
        double uniform = Math.log(source.vocab());
        int bad = 0;
        double peak = 0;
        double fin = 0;
        for (long seed : seeds) {
            SimpleMlp net = SimpleMlp.tanh(seed, 2 * source.vocab(), 32, 32, 32, 32, source.vocab());
            Optimizer opt = optimizer.apply(learningRate);
            SplittableRandom r = new SplittableRandom(seed);
            Tensor[] test = source.batch(1024, new SplittableRandom(seed + 99));
            float worst = 0;
            for (int s = 0; s < steps; s++) {
                Tensor[] b = source.batch(32, r);
                Tensor x = s < quiet ? b[0].div(jump) : b[0];
                Var loss = Functional.crossEntropy(net.forward(Var.of(x)), b[1]);
                if (s >= quiet + 10) {
                    float l = loss.value().item();
                    worst = Float.isFinite(l) ? Math.max(worst, l) : Float.POSITIVE_INFINITY;
                }
                loss.backward();
                opt.step(net.parameters());
            }
            float loss = Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value().item();
            if (!Float.isFinite(loss) || loss > uniform) {
                bad++;
            }
            peak += worst / seeds.size();
            fin += (Float.isFinite(loss) ? loss : 100) / seeds.size();
        }
        return new Stress(bad, peak, fin);
    }

    /**
     * The outcome of {@link #stress}.
     *
     * @param diverged runs whose final loss was non-finite or above the uniform guess
     * @param meanPeakLoss the mean over runs of the highest batch loss after the jump settles (10 steps on)
     * @param meanFinalLoss the mean held-out loss at the end (100 for a non-finite one)
     */
    public record Stress(int diverged, double meanPeakLoss, double meanFinalLoss) {
    }

    /**
     * The online convex counterexample on which Adam provably fails for small
     * {@code beta2} (Reddi et al. 2018; used by the ADOPT paper): {@code f_t(θ) = k θ} with
     * probability {@code 1/k}, else {@code -θ}, on {@code θ ∈ [-1, 1]}. The expected gradient
     * {@code 1/k} is positive, so the optimum is {@code θ = -1}; an optimizer whose step
     * size reacts to the rare large gradient drifts to {@code +1} instead.
     *
     * @param optimizer the optimizer
     * @param k the size of the rare gradient, for example 50
     * @param steps the steps
     * @param seed the seed
     * @return the mean of {@code θ} over the last tenth of the steps
     */
    public static double counterexample(Optimizer optimizer, int k, int steps, long seed) {
        Var theta = Var.param(Tensor.of(Shape.of(1), 0f));
        SplittableRandom r = new SplittableRandom(seed);
        double tail = 0;
        int counted = 0;
        for (int s = 0; s < steps; s++) {
            theta.setGrad(Tensor.of(Shape.of(1), r.nextInt(k) == 0 ? k : -1f));
            optimizer.step(List.of(theta));
            float v = Math.max(-1, Math.min(1, theta.value().item()));
            theta.assign(Tensor.of(Shape.of(1), v));
            if (s >= steps - steps / 10) {
                tail += v;
                counted++;
            }
        }
        return tail / counted;
    }
}
