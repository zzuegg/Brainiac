package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;
import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * Full-parameter fine-tuning with a plain evolution strategy, as in "Evolution Strategies at
 * Scale: LLM Fine-Tuning Beyond Reinforcement Learning": every iteration perturbs all weights
 * of the model with seeded Gaussian noise ({@link ParameterVector#noise}), loads each
 * perturbed copy into the model's own parameters, scores it with a sequence-level reward
 * (typically greedy decoding on a shared batch of prompts, so all perturbations see the same
 * prompts), and moves the weights along the z-scored reward-weighted noise
 * ({@link EvolutionStrategy}, antithetic pairs, no Adam by default). Only seeds and scalar
 * rewards would need to be communicated between workers. It needs no gradients, no
 * log-probabilities and no critic, and it optimises a reward smoothed over parameter space
 * rather than over sampled tokens.
 *
 * <p>Qiu et al. 2025, <a href="https://arxiv.org/abs/2509.24372">arXiv:2509.24372</a>.
 */
public final class EvolutionStrategyFineTuner {

    private final List<Var> parameters;
    private final EvolutionStrategy es;
    private long rewardEvaluations;

    private EvolutionStrategyFineTuner(List<Var> parameters, double sigma, double learningRate, int pairs, long seed) {
        this.parameters = parameters;
        this.es = EvolutionStrategy.of(ParameterVector.flatten(parameters)).sigma(sigma).learningRate(learningRate)
                .pairs(pairs).adam(false).rankShaping(false).seed(seed);
    }

    /**
     * @param parameters the model's parameters, fine-tuned in place
     * @param sigma the perturbation scale
     * @param learningRate the step size on the z-scored gradient estimate
     * @param pairs antithetic perturbation pairs per iteration
     * @param seed the noise seed
     * @return the fine-tuner
     */
    public static EvolutionStrategyFineTuner of(List<Var> parameters, double sigma, double learningRate, int pairs,
            long seed) {
        return new EvolutionStrategyFineTuner(Objects.requireNonNull(parameters), sigma, learningRate, pairs, seed);
    }

    /** @return reward evaluations (perturbed models scored) so far */
    public long rewardEvaluations() {
        return rewardEvaluations;
    }

    /**
     * One iteration: score every perturbation with {@code reward} (evaluated on the model as
     * currently loaded), update, and load the new centre into the model.
     *
     * @param reward the reward of the currently loaded weights (higher is better)
     * @return the iteration's statistics
     */
    public EvolutionStrategy.Step step(DoubleSupplier reward) {
        EvolutionStrategy.Step s = es.step(x -> {
            ParameterVector.assign(parameters, x);
            rewardEvaluations++;
            return reward.getAsDouble();
        });
        ParameterVector.assign(parameters, es.center());
        return s;
    }
}
