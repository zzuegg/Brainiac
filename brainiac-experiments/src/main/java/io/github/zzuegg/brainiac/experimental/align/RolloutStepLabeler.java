package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiPredicate;

/**
 * Math-Shepherd's automatic process labels (Wang et al., ACL 2024,
 * <a href="https://arxiv.org/abs/2312.08935">arXiv:2312.08935</a>). Human step labels are
 * expensive; instead each step is labelled by its potential to reach the right answer: from the
 * prefix that ends at step {@code s}, a completer samples {@code N} continuations, the
 * <em>hard</em> label is 1 if any of them reaches the gold answer and the <em>soft</em> label is
 * the share that do. The last step of a finished solution is labelled by the solution's own
 * correctness. Steps are cut as {@code ProcessRewardModel.stepPrefixes} cuts them, so the labels
 * feed {@code ProcessRewardModel.fit} unchanged.
 */
public final class RolloutStepLabeler {

    /**
     * The labels of one solution.
     *
     * @param hard 1 if any rollout from the step succeeded
     * @param soft the share of rollouts that succeeded
     * @param tokens tokens sampled for the labels
     */
    public record Labels(float[] hard, float[] soft, long tokens) {
    }

    private final WindowLm completer;
    private final int maxTokens;
    private final int end;
    private final int[] stepTokens;
    private int rollouts = 8;

    private RolloutStepLabeler(WindowLm completer, int maxTokens, int end, int[] stepTokens) {
        this.completer = Objects.requireNonNull(completer);
        this.maxTokens = maxTokens;
        this.end = end;
        this.stepTokens = stepTokens.clone();
    }

    /**
     * @param completer the model that completes partial solutions
     * @param maxTokens the most tokens per solution
     * @param end the end token
     * @param stepTokens the tokens that end a step (besides the end token)
     * @return the labeller
     */
    public static RolloutStepLabeler of(WindowLm completer, int maxTokens, int end, int... stepTokens) {
        return new RolloutStepLabeler(completer, maxTokens, end, stepTokens);
    }

    /**
     * @param n continuations sampled per step
     * @return this labeller
     */
    public RolloutStepLabeler rollouts(int n) {
        this.rollouts = n;
        return this;
    }

    /**
     * Labels every step of a solution.
     *
     * @param prompt the prompt
     * @param completion the solution
     * @param correct whether a complete solution reaches the gold answer
     * @param random the random source
     * @return hard and soft labels, one per step
     */
    public Labels label(int[] prompt, int[] completion, BiPredicate<int[], int[]> correct, SplittableRandom random) {
        int[] stops = Arrays.copyOf(stepTokens, stepTokens.length + 1);
        stops[stepTokens.length] = end;
        int[] ends = TokenPolicyGradient.stepEnds(completion, stops);
        float[] hard = new float[ends.length];
        float[] soft = new float[ends.length];
        long tokens = 0;
        for (int s = 0; s < ends.length; s++) {
            int[] prefix = Arrays.copyOf(completion, ends[s]);
            if (prefix[prefix.length - 1] == end || ends[s] >= maxTokens) {
                float ok = correct.test(prompt, prefix) ? 1 : 0;
                hard[s] = ok;
                soft[s] = ok;
                continue;
            }
            int[] context = Arrays.copyOf(prompt, prompt.length + prefix.length);
            System.arraycopy(prefix, 0, context, prompt.length, prefix.length);
            int[][] rep = new int[rollouts][];
            Arrays.fill(rep, context);
            int[][] outs = completer.sample(rep, maxTokens - prefix.length, end, 1.0, random);
            int hits = 0;
            for (int[] o : outs) {
                tokens += o.length;
                int[] full = Arrays.copyOf(prefix, prefix.length + o.length);
                System.arraycopy(o, 0, full, prefix.length, o.length);
                hits += correct.test(prompt, full) ? 1 : 0;
            }
            hard[s] = hits > 0 ? 1 : 0;
            soft[s] = (float) hits / rollouts;
        }
        return new Labels(hard, soft, tokens);
    }

    /**
     * The share of steps whose label, thresholded at {@code 1/2}, matches the gold label.
     *
     * @param labels the predicted labels
     * @param gold the gold labels
     * @return the agreement, over {@code min(lengths)} steps
     */
    public static double agreement(float[] labels, float[] gold) {
        int n = Math.min(labels.length, gold.length);
        int ok = 0;
        for (int i = 0; i < n; i++) {
            ok += (labels[i] >= 0.5f) == (gold[i] >= 0.5f) ? 1 : 0;
        }
        return n == 0 ? 1 : (double) ok / n;
    }
}
