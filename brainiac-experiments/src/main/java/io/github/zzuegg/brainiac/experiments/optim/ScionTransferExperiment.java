package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.MaximalUpdate;
import io.github.zzuegg.brainiac.experimental.train.Scion;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning-rate transfer across widths: the promise of {@link Scion} (and of μP) is that
 * the learning rate tuned on a narrow network is still the best one on a wide network.
 * A three-layer ReLU MLP on the Markov next-token source is trained at several widths and
 * learning rates with:
 *
 * <ul>
 *   <li><b>Scion</b>: row-normalised input layer, spectral (orthogonalised) hidden layer,
 *       sign output layer scaled by {@code 1 / fanIn}, RMS-normalised biases;</li>
 *   <li><b>Muon + μP</b> (the baseline): μP initialisation from {@link MaximalUpdate}, Muon
 *       for the hidden matrix, Adam for the input layer and biases and Adam at
 *       {@code lr / r} for the output layer, all from one swept rate (Adam at a tenth of
 *       Muon's).</li>
 * </ul>
 * The result is each width's loss for every rate, the best rate per width, and the
 * <em>transfer regret</em>: the loss at the widest width using the narrowest width's best
 * rate, minus the widest width's best loss.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ScionTransferExperiment {

    /** The optimizer families compared. */
    public enum Method {
        /** Scion with per-layer norms. */
        SCION,
        /** Muon for hidden matrices, μP Adam for the rest. */
        MUON_MUP,
        /** Plain Adam at a tenth of the swept rate, standard parametrisation: no transfer expected. */
        ADAM
    }

    private ScionTransferExperiment() {
    }

    /**
     * The sweep's results.
     *
     * @param widths the widths, narrowest first
     * @param rates the learning rates swept
     * @param loss {@code [width][rate]} final held-out losses
     */
    public record Sweep(int[] widths, float[] rates, float[][] loss) {

        /**
         * @param w a width index
         * @return the index of its best rate
         */
        public int best(int w) {
            int b = 0;
            for (int i = 1; i < rates.length; i++) {
                if (loss[w][i] < loss[w][b]) {
                    b = i;
                }
            }
            return b;
        }

        /** @return the widest width's loss at the narrowest width's best rate, minus its best loss */
        public float regret() {
            int last = widths.length - 1;
            return loss[last][best(0)] - loss[last][best(last)];
        }

        /** @return how many grid points the best rate moved from the narrowest to the widest width */
        public int shift() {
            return Math.abs(best(widths.length - 1) - best(0));
        }
    }

    /**
     * Runs the sweep.
     *
     * @param method the optimizer family
     * @param widths the widths, narrowest first (the base width of μP)
     * @param rates the learning rates
     * @param steps training steps per run
     * @param seed the seed
     * @return the sweep
     */
    public static Sweep sweep(Method method, int[] widths, float[] rates, int steps, long seed) {
        MarkovSource source = MarkovSource.random(10, 1);
        Tensor[] test = source.batch(2048, new SplittableRandom(seed + 5));
        float[][] loss = new float[widths.length][rates.length];
        for (int w = 0; w < widths.length; w++) {
            for (int i = 0; i < rates.length; i++) {
                loss[w][i] = train(method, source, test, widths[0], widths[w], rates[i], steps, seed);
            }
        }
        return new Sweep(widths.clone(), rates.clone(), loss);
    }

    private static float train(Method method, MarkovSource source, Tensor[] test, int base, int width, float rate,
            int steps, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, 2 * source.vocab(), width, width, source.vocab());
        Optimizer opt;
        if (method == Method.SCION) {
            Scion scion = new Scion(rate, 0.1f, false);
            scion.norm(net.weight(0), Scion.Norm.ROW, 1f);
            scion.norm(net.weight(1), Scion.Norm.SPECTRAL, 1f);
            scion.norm(net.weight(2), Scion.Norm.SIGN, 10f);
            opt = scion;
        } else if (method == Method.ADAM) {
            opt = Optimizer.adam(rate / 10);
        } else {
            MaximalUpdate mup = new MaximalUpdate(base, width);
            SplittableRandom r = new SplittableRandom(seed);
            MaximalUpdate.Role[] roles = {MaximalUpdate.Role.INPUT, MaximalUpdate.Role.HIDDEN,
                MaximalUpdate.Role.OUTPUT};
            for (int l = 0; l < 3; l++) {
                Var wl = net.weight(l);
                wl.assign(mup.init(roles[l]).create(wl.shape(), wl.shape().dim(0), wl.shape().dim(1), r));
            }
            float adam = rate / 10;
            opt = Optimizer.groups(Optimizer.adam(adam),
                    Optimizer.Group.of(List.of(net.weight(1)), Optimizer.muon(rate, 0, 0.95f, false)),
                    Optimizer.Group.of(List.of(net.weight(2)),
                            Optimizer.adam(mup.adamLearningRate(MaximalUpdate.Role.OUTPUT, adam))));
        }
        SplittableRandom r = new SplittableRandom(seed + 11);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = source.batch(32, r);
            Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
            opt.step(net.parameters());
        }
        float l = Functional.crossEntropy(net.forward(Var.of(test[0])), test[1]).value().item();
        return Float.isFinite(l) ? l : 100f;
    }
}
