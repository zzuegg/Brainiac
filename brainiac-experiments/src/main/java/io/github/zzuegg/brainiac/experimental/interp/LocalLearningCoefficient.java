package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.WeightPosterior;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;

/**
 * The local learning coefficient (LLC) of singular learning theory as a developmental probe
 * (Lau et al., AISTATS 2025; Hoogland et al., TMLR 2025). The learning coefficient
 * {@code lambda} measures the effective dimensionality of a minimum: degenerate, simpler
 * solutions have a lower one. At trained weights {@code w*} it is estimated as
 *
 * <pre>
 * lambda_hat = n beta (E[L_n(w)] - L_n(w*)),   beta = 1 / log n,
 * </pre>
 *
 * <p>with the expectation under the localised tempered posterior
 * {@code exp(-n beta L_n(w) - gamma/2 |w - w*|^2)}, sampled by stochastic-gradient Langevin
 * dynamics. The sampler is {@link WeightPosterior.Sgld} unchanged, run at temperature
 * {@code 1 / beta} without a prior, on the loss plus {@code gamma / (2 n beta) |w - w*|^2},
 * which gives exactly that density. Sampling only some parameters (the others held at
 * {@code w*}) gives the per-layer "refined" LLC.
 *
 * <p>Tracked over training, plateaus and drops of {@code lambda_hat} mark developmental
 * stages; a chain whose loss explodes is reported as diverged, since the estimate is then
 * meaningless (the step size is too large or {@code gamma} too small).
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2308.12108">arXiv:2308.12108</a>,
 * <a href="https://arxiv.org/abs/2402.02364">arXiv:2402.02364</a>.
 */
public final class LocalLearningCoefficient {

    private LocalLearningCoefficient() {
    }

    /**
     * Sampler settings.
     *
     * @param stepSize the SGLD step size on the mean loss
     * @param localization {@code gamma}, the pull towards {@code w*}
     * @param chains independent chains
     * @param steps SGLD steps per chain
     * @param burnIn steps per chain not counted
     */
    public record Settings(float stepSize, float localization, int chains, int steps, int burnIn) {

        /**
         * Checks the settings.
         *
         * @param stepSize step size
         * @param localization gamma
         * @param chains chains
         * @param steps steps
         * @param burnIn burn-in
         */
        public Settings {
            if (!(stepSize > 0) || localization < 0 || chains < 1 || steps <= burnIn || burnIn < 0) {
                throw new IllegalArgumentException("bad LLC settings");
            }
        }
    }

    /**
     * An estimate.
     *
     * @param llc the mean over chains of {@code lambda_hat}
     * @param std the standard deviation over chains
     * @param chains the per-chain estimates
     * @param diverged whether any chain's loss became non-finite or exploded
     */
    public record Estimate(double llc, double std, double[] chains, boolean diverged) {
    }

    /**
     * Estimates the LLC at the parameters' current values, which are restored afterwards.
     *
     * @param sampled the parameters to sample (all of them, or one layer's for the refined LLC)
     * @param batchLoss the mean loss of minibatch {@code step}, as a function of the parameters
     * @param fullLoss {@code L_n(w*)}, the mean loss over the whole training set at the start
     * @param n the training-set size
     * @param settings the sampler settings
     * @param seed the seed
     * @return the estimate
     */
    public static Estimate estimate(List<Var> sampled, IntFunction<Var> batchLoss, double fullLoss, int n,
            Settings settings, long seed) {
        double beta = 1 / Math.log(n);
        double nBeta = n * beta;
        List<Tensor> star = new ArrayList<>();
        for (Var p : sampled) {
            star.add(p.value().add(0f));
        }
        double[] perChain = new double[settings.chains()];
        boolean diverged = false;
        int step = 0;
        for (int c = 0; c < settings.chains(); c++) {
            WeightPosterior.Sgld sgld = new WeightPosterior.Sgld(settings.stepSize(), n, 0f, (float) (1 / beta),
                    seed + 7919L * c);
            double sum = 0;
            int counted = 0;
            for (int s = 0; s < settings.steps(); s++) {
                Var loss = batchLoss.apply(step++);
                float value = loss.value().item();
                Var objective = loss;
                if (settings.localization() > 0) {
                    Var pull = null;
                    for (int i = 0; i < sampled.size(); i++) {
                        Var d = sampled.get(i).sub(star.get(i)).square().sum();
                        pull = pull == null ? d : pull.add(d);
                    }
                    objective = loss.add(pull.mul((float) (settings.localization() / (2 * nBeta))));
                }
                objective.backward();
                sgld.step(sampled);
                if (!Float.isFinite(value) || value > 1e6) {
                    diverged = true;
                    break;
                }
                if (s >= settings.burnIn()) {
                    sum += value;
                    counted++;
                }
            }
            perChain[c] = counted == 0 ? Double.NaN : nBeta * (sum / counted - fullLoss);
            for (int i = 0; i < sampled.size(); i++) {
                sampled.get(i).assign(star.get(i).add(0f));
                sampled.get(i).zeroGrad();
            }
        }
        double mean = 0;
        for (double v : perChain) {
            mean += v / perChain.length;
        }
        double var = 0;
        for (double v : perChain) {
            var += (v - mean) * (v - mean) / perChain.length;
        }
        return new Estimate(mean, Math.sqrt(var), perChain, diverged);
    }
}
