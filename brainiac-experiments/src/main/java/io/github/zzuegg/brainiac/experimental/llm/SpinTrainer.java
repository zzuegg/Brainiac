package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-Play fINe-tuning (SPIN, <a href="https://arxiv.org/abs/2401.01335">2401.01335</a>):
 * improve a supervised model without new labels. Each iteration freezes the current model
 * as the "opponent", lets it answer the training prompts, and trains the model with a DPO
 * loss to prefer the human (ground-truth) answers over the opponent's own generations, with
 * the opponent as the reference. The fixed point is reached when the model's generations are
 * indistinguishable from the data.
 */
public final class SpinTrainer {

    private final WindowLm policy;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float beta = 0.1f;
    private float nll;
    private int stepsPerIteration = 20;
    private int maxNewTokens = 4;
    private int endToken;
    private double temperature = 1.0;

    private SpinTrainer(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
    }

    /**
     * @param policy a model, usually already fine-tuned on the data
     * @return the trainer
     */
    public static SpinTrainer of(WindowLm policy) {
        return new SpinTrainer(policy);
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public SpinTrainer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param weight the weight of a likelihood term on the chosen answers, see
     *     {@link DpoTrainer#chosenNll(float)}
     * @return this trainer
     */
    public SpinTrainer chosenNll(float weight) {
        this.nll = weight;
        return this;
    }

    /**
     * @param beta the DPO inverse temperature
     * @return this trainer
     */
    public SpinTrainer beta(float beta) {
        this.beta = beta;
        return this;
    }

    /**
     * @param n DPO steps per self-play iteration
     * @return this trainer
     */
    public SpinTrainer stepsPerIteration(int n) {
        this.stepsPerIteration = n;
        return this;
    }

    /**
     * @param maxTokens the most tokens per generation
     * @param end the end token
     * @param temperature the opponent's sampling temperature
     * @return this trainer
     */
    public SpinTrainer generation(int maxTokens, int end, double temperature) {
        this.maxNewTokens = maxTokens;
        this.endToken = end;
        this.temperature = temperature;
        return this;
    }

    /**
     * Runs one self-play iteration.
     *
     * @param prompts the training prompts
     * @param answers their ground-truth completions
     * @param random the random source
     * @return the mean DPO loss of the iteration
     */
    public double iterate(List<int[]> prompts, List<int[]> answers, SplittableRandom random) {
        WindowLm opponent = policy.copy();
        int[][] generated = opponent.sample(prompts.toArray(int[][]::new), maxNewTokens, endToken, temperature, random);
        List<DpoTrainer.Preference> pairs = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            if (!Arrays.equals(generated[i], answers.get(i)) && generated[i].length > 0) {
                pairs.add(new DpoTrainer.Preference(prompts.get(i), answers.get(i), generated[i]));
            }
        }
        if (pairs.isEmpty()) {
            return 0;
        }
        DpoTrainer dpo = DpoTrainer.of(policy).reference(opponent).beta(beta).optimizer(optimizer).chosenNll(nll);
        double total = 0;
        for (int s = 0; s < stepsPerIteration; s++) {
            total += dpo.step(pairs);
        }
        return total / stepsPerIteration;
    }
}
