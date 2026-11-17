package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.AdaptiveMuon;
import io.github.zzuegg.brainiac.experimental.train.Apollo;
import io.github.zzuegg.brainiac.experimental.train.NorMuon;
import io.github.zzuegg.brainiac.experimental.train.OrthogonalMuon;
import io.github.zzuegg.brainiac.experimental.train.PolarExpress;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.Run;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * The 2025-2026 Muon family and memory-light Adam variants on the Markov next-token task of
 * {@link ModernOptimizerExperiment}, each at a learning rate tuned over a small grid:
 *
 * <ul>
 *   <li>{@link PolarExpress} inside Muon ({@link OrthogonalMuon}) against Muon's
 *       Newton-Schulz at 3 and 5 iterations;</li>
 *   <li>{@link NorMuon}: loss and the spread of per-neuron update norms against Muon and
 *       AdamW ({@link #neuronSpread});</li>
 *   <li>{@link AdaptiveMuon} (NAMO, NAMO-D, MALT): loss and learning-rate sensitivity
 *       ({@link #sweep}) against Muon and NorMuon;</li>
 *   <li>{@link Apollo}: loss and optimizer-state bytes against GaLore, Adam8bit and
 *       Adam-mini, through {@link ModernOptimizerExperiment#train}.</li>
 * </ul>
 * Muon-family optimizers take the matrices; biases go to Adam at {@code 3e-3}.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class MuonFamilyExperiment {

    /** The task: a 10-token second-order Markov source. */
    public static final MarkovSource SOURCE = MarkovSource.random(10, 1);

    private MuonFamilyExperiment() {
    }

    /**
     * A Muon-family optimizer for the matrices and Adam for the rest.
     *
     * @param matrices the optimizer for the weight matrices
     * @return the grouped optimizer
     */
    public static Optimizer withAdamForVectors(Optimizer matrices) {
        return Optimizer.groups(Optimizer.adam(3e-3f), Optimizer.Group.matrices(matrices));
    }

    /**
     * Final held-out losses over a learning-rate grid, averaged over seeds.
     *
     * @param make the optimizer for a rate
     * @param rates the grid
     * @param steps steps per run
     * @param seeds the seeds per rate
     * @return the mean final loss at each rate
     */
    public static double[] sweep(Function<Float, Optimizer> make, float[] rates, int steps, int seeds) {
        double[] out = new double[rates.length];
        for (int i = 0; i < rates.length; i++) {
            float lr = rates[i];
            for (long seed = 1; seed <= seeds; seed++) {
                Run run = ModernOptimizerExperiment.train(() -> make.apply(lr), SOURCE, 64, steps, 32, steps, seed);
                out[i] += (run.diverged() ? 10 : run.finalLoss()) / seeds;
            }
        }
        return out;
    }

    /**
     * @param losses a sweep's losses
     * @return the best one
     */
    public static double best(double[] losses) {
        double b = Double.POSITIVE_INFINITY;
        for (double l : losses) {
            b = Math.min(b, l);
        }
        return b;
    }

    /**
     * Learning-rate sensitivity: the mean excess loss over the grid, {@code mean(l) - min(l)}.
     *
     * @param losses a sweep's losses
     * @return the mean regret of picking a rate from the grid at random
     */
    public static double sensitivity(double[] losses) {
        double mean = 0;
        for (double l : losses) {
            mean += l / losses.length;
        }
        return mean - best(losses);
    }

    /**
     * Trains with an optimizer and returns the spread of per-neuron update norms of the
     * input matrix ({@code [20, 64]}: wide, so orthogonalisation alone does not even out its
     * 64 neurons) over the last {@code last} steps: each neuron's total update size
     * {@code sqrt(sum_t ||dw_t[:, j]||^2)}, the largest over the median one. Single steps
     * are dominated by batch noise; NorMuon evens out what accumulates.
     *
     * @param optimizer the optimizer, matrices and vectors (see {@link #withAdamForVectors})
     * @param steps training steps
     * @param last steps at the end whose updates are measured
     * @param seed the seed
     * @return {@code {final held-out loss, neuron spread}}
     */
    public static double[] neuronSpread(Optimizer optimizer, int steps, int last, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, 2 * SOURCE.vocab(), 64, 64, SOURCE.vocab());
        SplittableRandom r = new SplittableRandom(seed + 17);
        Tensor energy = null;
        for (int s = 1; s <= steps; s++) {
            Tensor[] b = SOURCE.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            Tensor before = net.weight(0).value();
            optimizer.step(net.parameters());
            if (s > steps - last) {
                Tensor e = net.weight(0).value().sub(before).square().sum(0, true);
                energy = energy == null ? e : energy.add(e);
            }
        }
        Tensor[] test = SOURCE.batch(2048, new SplittableRandom(seed ^ 0x5DEECE66DL));
        float loss = Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value().item();
        return new double[] {loss, NorMuon.neuronSpread(energy.sqrt())};
    }
}
