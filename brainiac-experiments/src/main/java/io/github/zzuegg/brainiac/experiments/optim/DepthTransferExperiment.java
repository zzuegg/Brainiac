package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CompleteP;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning-rate transfer across depth, the depth twin of {@link ScionTransferExperiment}:
 * a pre-norm residual MLP {@code h <- h + m * W2 relu(W1 norm(h))} with {@code L} blocks on
 * the Markov next-token source, swept over learning rates at depths 2 to 16 under three
 * {@link CompleteP} parametrisations: the standard one ({@code alpha = 0}), depth-μP
 * ({@code alpha = 1/2}) and CompleteP ({@code alpha = 1}). Depth 2 is the base. A
 * parametrisation transfers when the best rate stays put as depth grows; the regret is the
 * deepest model's loss at the base model's best rate minus its best loss.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class DepthTransferExperiment {

    private static final MarkovSource SOURCE = MarkovSource.random(10, 1);

    private DepthTransferExperiment() {
    }

    /**
     * The sweep's results.
     *
     * @param depths the depths, the base first
     * @param rates the learning rates swept
     * @param loss {@code [depth][rate]} final held-out losses
     */
    public record Sweep(int[] depths, float[] rates, float[][] loss) {

        /**
         * @param d a depth index
         * @return the index of its best rate
         */
        public int best(int d) {
            int b = 0;
            for (int i = 1; i < rates.length; i++) {
                if (loss[d][i] < loss[d][b]) {
                    b = i;
                }
            }
            return b;
        }

        /** @return the deepest model's loss at the base model's best rate minus its best loss */
        public float regret() {
            int last = depths.length - 1;
            return loss[last][best(0)] - loss[last][best(last)];
        }

        /** @return the largest move of the best rate, in grid steps, from the base depth */
        public int maxShift() {
            int shift = 0;
            for (int d = 1; d < depths.length; d++) {
                shift = Math.max(shift, Math.abs(best(d) - best(0)));
            }
            return shift;
        }
    }

    /**
     * Runs the sweep.
     *
     * @param alpha the depth exponent of the parametrisation
     * @param depths the depths, base first
     * @param rates the learning rates
     * @param width the residual width
     * @param steps steps per run
     * @param seed the seed
     * @return the sweep
     */
    public static Sweep sweep(double alpha, int[] depths, float[] rates, int width, int steps, long seed) {
        Tensor[] test = SOURCE.batch(2048, new SplittableRandom(seed + 5));
        float[][] loss = new float[depths.length][rates.length];
        for (int d = 0; d < depths.length; d++) {
            for (int i = 0; i < rates.length; i++) {
                loss[d][i] = train(new CompleteP(depths[0], depths[d], alpha), depths[d], width, rates[i], steps,
                        test, seed);
            }
        }
        return new Sweep(depths.clone(), rates.clone(), loss);
    }

    private static Var norm(Var h) {
        Var c = h.sub(h.mean(1, true));
        return c.div(c.square().mean(1, true).add(1e-5f).sqrt());
    }

    private static Var weight(SplittableRandom r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(2.0 / in), r));
    }

    private static float train(CompleteP p, int depth, int width, float rate, int steps, Tensor[] test, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int vocab = SOURCE.vocab();
        Var embed = weight(r, 2 * vocab, width);
        Var out = Var.param(Tensor.normal(Shape.of(width, vocab), 0, (float) (1 / Math.sqrt(width)), r));
        List<Var> hidden = new ArrayList<>();
        for (int l = 0; l < 2 * depth; l++) {
            hidden.add(weight(r, width, width));
        }
        float m = p.branchMultiplier();
        Optimizer opt = Optimizer.groups(Optimizer.adam(rate),
                Optimizer.Group.of(hidden, Optimizer.adam(p.hiddenLearningRate(rate), 0.9f, 0.999f,
                        p.hiddenEpsilon(1e-8f))));
        List<Var> params = new ArrayList<>(hidden);
        params.add(embed);
        params.add(out);
        SplittableRandom data = new SplittableRandom(seed + 11);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = SOURCE.batch(32, data);
            Functional.crossEntropy(forward(Var.of(b[0]), embed, hidden, out, m), b[1]).backward();
            opt.step(params);
        }
        float l = Functional.crossEntropy(forward(Var.of(test[0]), embed, hidden, out, m), test[1]).value().item();
        return Float.isFinite(l) ? l : 100f;
    }

    private static Var forward(Var x, Var embed, List<Var> hidden, Var out, float m) {
        Var h = x.matmul(embed);
        for (int l = 0; l < hidden.size(); l += 2) {
            h = h.add(norm(h).matmul(hidden.get(l)).relu().matmul(hidden.get(l + 1)).mul(m));
        }
        return norm(h).matmul(out);
    }
}
