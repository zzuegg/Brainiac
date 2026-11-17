package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A process reward model: scores every reasoning step instead of only the final answer
 * (Lightman et al., "Let's Verify Step by Step",
 * <a href="https://arxiv.org/abs/2305.20050">2305.20050</a>). A step ends at one of the
 * step tokens; the model reads the prompt and the solution up to the end of each step and
 * predicts whether every step so far is right. A solution's score aggregates the step scores
 * ({@link Aggregate#MIN} is the usual choice: a chain is only as good as its weakest step).
 * Because it scores partial solutions it is also the value function of step-level tree
 * search ({@link TestTimeCompute#stepBeamSearch}).
 */
public final class ProcessRewardModel {

    /** How step scores combine into a solution score. */
    public enum Aggregate {
        /** The weakest step. */
        MIN,
        /** The product of the step probabilities. */
        PRODUCT,
        /** The score of the last step only. */
        LAST
    }

    private final SequenceScorer scorer;
    private final int[] stepTokens;

    private ProcessRewardModel(SequenceScorer scorer, int[] stepTokens) {
        this.scorer = Objects.requireNonNull(scorer);
        this.stepTokens = stepTokens.clone();
    }

    /**
     * @param scorer the network that scores prefixes
     * @param stepTokens the tokens that end a step (the end-of-sequence token is usually one)
     * @return the model
     */
    public static ProcessRewardModel of(SequenceScorer scorer, int... stepTokens) {
        return new ProcessRewardModel(scorer, stepTokens);
    }

    /** @return the scorer */
    public SequenceScorer scorer() {
        return scorer;
    }

    private boolean isStepEnd(int token) {
        for (int t : stepTokens) {
            if (t == token) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param prompt the prompt
     * @param completion a (possibly partial) solution
     * @return {@code prompt ++ completion} cut after every step, in order; a trailing
     *     unfinished step is included as the last prefix
     */
    public List<int[]> stepPrefixes(int[] prompt, int[] completion) {
        List<int[]> out = new ArrayList<>();
        for (int t = 0; t < completion.length; t++) {
            if (isStepEnd(completion[t]) || t == completion.length - 1) {
                int[] prefix = Arrays.copyOf(prompt, prompt.length + t + 1);
                System.arraycopy(completion, 0, prefix, prompt.length, t + 1);
                out.add(prefix);
            }
        }
        return out;
    }

    /**
     * @param prompt the prompt
     * @param completion a solution
     * @return the probability that the solution is right up to each step
     */
    public double[] stepScores(int[] prompt, int[] completion) {
        List<int[]> prefixes = stepPrefixes(prompt, completion);
        double[] s = new double[prefixes.size()];
        for (int i = 0; i < s.length; i++) {
            s[i] = scorer.score(prefixes.get(i));
        }
        return s;
    }

    /**
     * @param prompt the prompt
     * @param completion a solution
     * @param aggregate how to combine the steps
     * @return the solution's score
     */
    public double score(int[] prompt, int[] completion, Aggregate aggregate) {
        double[] s = stepScores(prompt, completion);
        if (s.length == 0) {
            return 0;
        }
        return switch (aggregate) {
            case MIN -> Arrays.stream(s).min().orElse(0);
            case PRODUCT -> Arrays.stream(s).reduce(1, (a, b) -> a * b);
            case LAST -> s[s.length - 1];
        };
    }

    /**
     * Trains on step-labelled solutions.
     *
     * @param prompts the prompts
     * @param completions the solutions
     * @param stepLabels per solution, 1 or 0 for each step (as {@link #stepPrefixes} cuts it)
     * @param steps optimizer steps
     * @return the final loss
     */
    public float fit(List<int[]> prompts, List<int[]> completions, List<float[]> stepLabels, int steps) {
        List<int[]> seqs = new ArrayList<>();
        List<Float> labels = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            List<int[]> prefixes = stepPrefixes(prompts.get(i), completions.get(i));
            float[] l = stepLabels.get(i);
            for (int k = 0; k < prefixes.size() && k < l.length; k++) {
                seqs.add(prefixes.get(k));
                labels.add(l[k]);
            }
        }
        float[] y = new float[labels.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = labels.get(i);
        }
        return scorer.fit(seqs, y, steps);
    }
}
