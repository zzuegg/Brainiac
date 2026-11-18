package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Step-DPO's pair construction (Lai et al. 2024, <a href="https://arxiv.org/abs/2406.18629">
 * arXiv:2406.18629</a>). DPO on whole solutions gives one coarse signal for a long chain;
 * Step-DPO localizes it to the first mistake. Given a wrong solution whose steps are checked
 * automatically, the shared prefix is the correct steps before the first wrong one, the
 * rejected continuation is that wrong step, and the chosen continuation is a correct next step
 * sampled from the model at the same prefix. The DPO loss ({@code PreferenceLosses.dpo}, here
 * through {@link DpoTrainer}) is then applied to the single step conditioned on the prompt and
 * the prefix: {@link #build} returns a {@link DpoTrainer.Preference} whose prompt is
 * {@code prompt ++ prefix}.
 */
public final class StepPreferenceBuilder {

    private final WindowLm policy;
    private final int maxStepTokens;
    private final int[] stops;
    private int samples = 8;
    private double temperature = 1.0;
    private long tokens;

    private StepPreferenceBuilder(WindowLm policy, int maxStepTokens, int[] stops) {
        this.policy = Objects.requireNonNull(policy);
        this.maxStepTokens = maxStepTokens;
        this.stops = stops.clone();
    }

    /**
     * @param policy the model that proposes correct next steps
     * @param maxStepTokens the most tokens per step
     * @param stops the tokens that end a step (the end token included)
     * @return the builder
     */
    public static StepPreferenceBuilder of(WindowLm policy, int maxStepTokens, int... stops) {
        return new StepPreferenceBuilder(policy, maxStepTokens, stops);
    }

    /**
     * @param n candidate next steps sampled per pair
     * @return this builder
     */
    public StepPreferenceBuilder samples(int n) {
        this.samples = n;
        return this;
    }

    /**
     * @param t the sampling temperature of candidate steps (1 by default)
     * @return this builder
     */
    public StepPreferenceBuilder temperature(double t) {
        this.temperature = t;
        return this;
    }

    /** @return tokens sampled for chosen steps so far */
    public long tokens() {
        return tokens;
    }

    /**
     * Builds the pair of a wrong solution.
     *
     * @param prompt the prompt
     * @param completion a solution with at least one wrong step
     * @param stepLabels the checker: per step of a (partial) solution, 1 while every step so
     *     far is right and 0 from the first mistake on
     * @param random the random source
     * @return the step-level preference, or empty if the solution has no wrong step or no
     *     correct alternative was sampled
     */
    public Optional<DpoTrainer.Preference> build(int[] prompt, int[] completion,
            BiFunction<int[], int[], float[]> stepLabels, SplittableRandom random) {
        float[] labels = stepLabels.apply(prompt, completion);
        int first = -1;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0.5f) {
                first = i;
                break;
            }
        }
        if (first < 0) {
            return Optional.empty();
        }
        int[] ends = TokenPolicyGradient.stepEnds(completion, stops);
        int from = first == 0 ? 0 : ends[first - 1];
        int[] prefix = Arrays.copyOf(completion, from);
        int[] rejected = Arrays.copyOfRange(completion, from, ends[first]);
        int[] context = Arrays.copyOf(prompt, prompt.length + prefix.length);
        System.arraycopy(prefix, 0, context, prompt.length, prefix.length);
        int[][] rep = new int[samples][];
        Arrays.fill(rep, context);
        int[][] next = policy.sampleUntil(rep, maxStepTokens, stops, temperature, random);
        for (int[] step : next) {
            tokens += step.length;
            if (step.length == 0 || Arrays.equals(step, rejected)) {
                continue;
            }
            int[] candidate = Arrays.copyOf(prefix, prefix.length + step.length);
            System.arraycopy(step, 0, candidate, prefix.length, step.length);
            float[] l = stepLabels.apply(prompt, candidate);
            if (l.length > first && l[first] >= 0.5f) {
                return Optional.of(new DpoTrainer.Preference(context, step, rejected));
            }
        }
        return Optional.empty();
    }
}
