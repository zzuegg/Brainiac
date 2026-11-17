package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGenerator;

/**
 * Rejection-sampling fine-tuning (best-of-n distillation): sample {@code n} answers per
 * prompt, score them with a reward (a {@link RewardModel} or a verifier), keep the best of
 * each prompt if it beats a threshold, and fine-tune on the kept ones with plain
 * next-token loss. Repeated, it climbs the reward without RL machinery; it is the
 * "rejection sampling" stage of Llama 2 (Touvron et al. 2023,
 * <a href="https://arxiv.org/abs/2307.09288">arXiv:2307.09288</a>), RAFT (Dong et al. 2023,
 * <a href="https://arxiv.org/abs/2304.06767">arXiv:2304.06767</a>) and ReST / RFT.
 */
public final class RejectionSamplingFineTuning {

    private final HookedDecoder policy;
    private final ToDoubleFunction<int[]> reward;
    private final Optimizer optimizer;
    private final List<Var> parameters;
    private final float temperature;

    /**
     * @param policy the model to improve
     * @param reward scores a full sequence (prompt and answer)
     * @param optimizer the fine-tuning optimizer
     * @param parameters the values to train
     * @param temperature the sampling temperature
     */
    public RejectionSamplingFineTuning(HookedDecoder policy, ToDoubleFunction<int[]> reward, Optimizer optimizer,
            List<Var> parameters, float temperature) {
        this.policy = policy;
        this.reward = reward;
        this.optimizer = optimizer;
        this.parameters = parameters;
        this.temperature = temperature;
    }

    /**
     * @param meanReward the mean reward of every sample this round
     * @param keptReward the mean reward of the kept samples
     * @param kept the number of samples kept
     */
    public record Round(double meanReward, double keptReward, int kept) {
    }

    /**
     * One round: sample, select, fine-tune.
     *
     * @param prompts the prompts, each at least one token
     * @param samples answers per prompt
     * @param answerLength tokens per answer
     * @param threshold the reward a best answer must reach to be kept
     * @param epochs passes over the kept answers
     * @param random the source of randomness
     * @return what happened
     */
    public Round round(int[][] prompts, int samples, int answerLength, double threshold, int epochs,
            RandomGenerator random) {
        List<int[]> kept = new ArrayList<>();
        double sum = 0;
        double keptSum = 0;
        int count = 0;
        DecodingPipeline sampler = DecodingPipeline.of(policy).processor(LogitsProcessors.temperature(temperature))
                .sampler(TokenSampler.ancestral());
        for (int[] prompt : prompts) {
            int[] best = null;
            double bestReward = Double.NEGATIVE_INFINITY;
            for (int s = 0; s < samples; s++) {
                int[] answer = sampler.generate(prompt, answerLength, random);
                int[] full = Arrays.copyOf(prompt, prompt.length + answer.length);
                System.arraycopy(answer, 0, full, prompt.length, answer.length);
                double r = reward.applyAsDouble(full);
                sum += r;
                count++;
                if (r > bestReward) {
                    bestReward = r;
                    best = full;
                }
            }
            if (best != null && bestReward >= threshold) {
                kept.add(best);
                keptSum += bestReward;
            }
        }
        if (!kept.isEmpty()) {
            int[][] data = kept.toArray(new int[0][]);
            for (int e = 0; e < epochs; e++) {
                Var loss = policy.loss(data, true);
                loss.backward();
                optimizer.step(parameters);
                policy.zeroGrad();
            }
        }
        return new Round(sum / count, kept.isEmpty() ? 0 : keptSum / kept.size(), kept.size());
    }
}
