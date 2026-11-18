package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * A conservative combination of several reward models against reward over-optimization
 * (Coste et al., "Reward model ensembles help mitigate overoptimization", ICLR 2024,
 * <a href="https://arxiv.org/abs/2310.02743">arXiv:2310.02743</a>). Optimizing one learned
 * proxy first raises and then lowers the true reward as the policy drifts into regions where
 * the proxy is wrong (Gao et al., <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a>);
 * members trained from different seeds disagree exactly there, so scoring by
 * <ul>
 *   <li>{@link Mode#MEAN}: the members' mean,</li>
 *   <li>{@link Mode#WORST}: the lowest member (worst-case optimization), or</li>
 *   <li>{@link Mode#UNCERTAINTY}: the mean minus {@code lambda} times their standard deviation
 *       (uncertainty-weighted optimization)</li>
 * </ul>
 * removes most of the reward a policy can win by exploiting one member's errors. Members are
 * {@link BatchScorer}s, so any reward source fits; {@link #of(RewardModel, BiFunction)} adapts
 * a core-style {@link RewardModel}. Each member's raw scores are standardized by its own mean
 * and spread on a calibration set ({@link #calibrate}), as the paper does, so that members on
 * different scales combine sensibly.
 */
public final class RewardEnsemble implements GrpoTrainer.Reward {

    /** How member scores combine. */
    public enum Mode {
        /** The mean of the members. */
        MEAN,
        /** The lowest member. */
        WORST,
        /** The mean minus {@code lambda} standard deviations. */
        UNCERTAINTY
    }

    /** Scores many responses at once: one reward-model member. */
    @FunctionalInterface
    public interface BatchScorer {
        /**
         * @param prompts the prompts
         * @param responses one response per prompt
         * @return one score per response
         */
        float[] score(List<int[]> prompts, List<int[]> responses);
    }

    private final List<BatchScorer> members;
    private final Mode mode;
    private final float lambda;
    private final float[] offset;
    private final float[] scale;
    private final Map<String, Float> cache = new HashMap<>();

    private RewardEnsemble(List<BatchScorer> members, Mode mode, float lambda) {
        if (members.isEmpty()) {
            throw new IllegalArgumentException("an ensemble needs members");
        }
        this.members = List.copyOf(members);
        this.mode = Objects.requireNonNull(mode);
        this.lambda = lambda;
        this.offset = new float[members.size()];
        this.scale = new float[members.size()];
        Arrays.fill(scale, 1);
    }

    /**
     * @param members the members
     * @param mode how to combine them
     * @param lambda the uncertainty penalty of {@link Mode#UNCERTAINTY} (ignored otherwise)
     * @return the ensemble
     */
    public static RewardEnsemble of(List<BatchScorer> members, Mode mode, float lambda) {
        return new RewardEnsemble(members, mode, lambda);
    }

    /**
     * Adapts a reward model that reads whole fixed-length sequences.
     *
     * @param model the reward model
     * @param layout lays a prompt and a response out as the model's input sequence
     * @return the member
     */
    public static BatchScorer of(RewardModel model, BiFunction<int[], int[], int[]> layout) {
        return (prompts, responses) -> {
            int[][] seqs = new int[prompts.size()][];
            for (int i = 0; i < seqs.length; i++) {
                seqs[i] = layout.apply(prompts.get(i), responses.get(i));
            }
            return model.score(seqs).value().toArray();
        };
    }

    /** @return the number of members */
    public int size() {
        return members.size();
    }

    /**
     * Standardizes every member on a calibration set of responses (for example samples of the
     * reference policy), so each has mean 0 and spread 1 there.
     *
     * @param prompts the prompts
     * @param responses the responses
     * @return this ensemble
     */
    public RewardEnsemble calibrate(List<int[]> prompts, List<int[]> responses) {
        for (int m = 0; m < members.size(); m++) {
            float[] s = members.get(m).score(prompts, responses);
            double mean = 0;
            for (float x : s) {
                mean += x;
            }
            mean /= s.length;
            double var = 0;
            for (float x : s) {
                var += (x - mean) * (x - mean);
            }
            offset[m] = (float) mean;
            scale[m] = (float) Math.sqrt(var / Math.max(1, s.length - 1) + 1e-8);
        }
        cache.clear();
        return this;
    }

    /**
     * @param prompts the prompts
     * @param responses the responses
     * @return the standardized member scores {@code [member][response]}
     */
    public float[][] memberScores(List<int[]> prompts, List<int[]> responses) {
        float[][] out = new float[members.size()][];
        for (int m = 0; m < out.length; m++) {
            float[] s = members.get(m).score(prompts, responses);
            for (int i = 0; i < s.length; i++) {
                s[i] = (s[i] - offset[m]) / scale[m];
            }
            out[m] = s;
        }
        return out;
    }

    /**
     * Combines standardized member scores.
     *
     * @param column one score per member
     * @param mode the combination
     * @param lambda the uncertainty penalty
     * @return the combined score
     */
    public static float combine(float[] column, Mode mode, float lambda) {
        double mean = 0;
        float min = Float.POSITIVE_INFINITY;
        for (float x : column) {
            mean += x;
            min = Math.min(min, x);
        }
        mean /= column.length;
        if (mode == Mode.WORST) {
            return min;
        }
        if (mode == Mode.MEAN || column.length < 2) {
            return (float) mean;
        }
        double var = 0;
        for (float x : column) {
            var += (x - mean) * (x - mean);
        }
        return (float) (mean - lambda * Math.sqrt(var / (column.length - 1)));
    }

    /**
     * @param prompts the prompts
     * @param responses the responses
     * @return the combined score of each response
     */
    public float[] scoreAll(List<int[]> prompts, List<int[]> responses) {
        float[][] s = memberScores(prompts, responses);
        float[] out = new float[responses.size()];
        float[] column = new float[members.size()];
        for (int i = 0; i < out.length; i++) {
            for (int m = 0; m < column.length; m++) {
                column[m] = s[m][i];
            }
            out[i] = combine(column, mode, lambda);
        }
        return out;
    }

    @Override
    public float score(int[] prompt, int[] completion) {
        String key = Arrays.toString(prompt) + Arrays.toString(completion);
        Float hit = cache.get(key);
        if (hit != null) {
            return hit;
        }
        float v = scoreAll(List.of(prompt), List.of(completion))[0];
        cache.put(key, v);
        return v;
    }

    /**
     * A copy with another combination rule over the same members and calibration.
     *
     * @param m the combination
     * @param lam the uncertainty penalty
     * @return the copy
     */
    public RewardEnsemble withMode(Mode m, float lam) {
        RewardEnsemble e = new RewardEnsemble(members, m, lam);
        System.arraycopy(offset, 0, e.offset, 0, offset.length);
        System.arraycopy(scale, 0, e.scale, 0, scale.length);
        return e;
    }

    /**
     * A single-member view, standardized like the ensemble.
     *
     * @param index the member
     * @return an ensemble of that member alone
     */
    public RewardEnsemble member(int index) {
        RewardEnsemble e = new RewardEnsemble(new ArrayList<>(List.of(members.get(index))), Mode.MEAN, 0);
        e.offset[0] = offset[index];
        e.scale[0] = scale[index];
        return e;
    }

    @Override
    public String toString() {
        return "RewardEnsemble(" + members.size() + " members, " + mode + ")";
    }
}
