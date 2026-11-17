package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Objects;
import java.util.function.ToIntFunction;

/**
 * Length-controlled policy optimization, LCPO (Aggarwal and Welleck, "L1: controlling how long
 * a reasoning model thinks with reinforcement learning",
 * <a href="https://arxiv.org/abs/2503.04697">2503.04697</a>). The prompt states a target
 * thinking length {@code n_gold} ("think for N tokens"), and the reward combines correctness
 * with how well the realized length {@code n_y} follows it, so one model learns a
 * controllable reasoning budget instead of having it forced at decode time (compare
 * {@link BudgetForcing}):
 * <ul>
 *   <li>{@link Mode#EXACT} (L1-Exact): {@code r = 1[correct] - alpha |n_gold - n_y|};</li>
 *   <li>{@link Mode#MAX} (L1-Max): {@code r = 1[correct] clip(alpha (n_gold - n_y) + delta, 0, 1)},
 *       which only penalizes going over the budget and still prefers a correct answer
 *       ({@code delta = 0.5}).</li>
 * </ul>
 * It wraps any correctness reward and plugs into {@link GrpoTrainer}.
 */
public final class LengthControlledReward implements GrpoTrainer.Reward {

    /** Which LCPO objective. */
    public enum Mode {
        /** Hit the target length exactly. */
        EXACT,
        /** Stay within the target length. */
        MAX
    }

    private final GrpoTrainer.Reward correct;
    private final ToIntFunction<int[]> target;
    private final ToIntFunction<int[]> length;
    private final Mode mode;
    private final float alpha;
    private final float delta;

    /**
     * @param correct 1 for a correct completion, 0 otherwise
     * @param target reads the target length {@code n_gold} off the prompt
     * @param length measures the realized thinking length {@code n_y} of a completion
     * @param mode exact or max
     * @param alpha the length penalty per unit of length
     * @param delta the L1-Max offset (ignored by {@code EXACT}; paper: 0.5)
     */
    public LengthControlledReward(GrpoTrainer.Reward correct, ToIntFunction<int[]> target,
            ToIntFunction<int[]> length, Mode mode, float alpha, float delta) {
        this.correct = Objects.requireNonNull(correct);
        this.target = Objects.requireNonNull(target);
        this.length = Objects.requireNonNull(length);
        this.mode = Objects.requireNonNull(mode);
        this.alpha = alpha;
        this.delta = delta;
    }

    @Override
    public float score(int[] prompt, int[] completion) {
        float c = correct.score(prompt, completion);
        int gold = target.applyAsInt(prompt);
        int n = length.applyAsInt(completion);
        return switch (mode) {
            case EXACT -> c - alpha * Math.abs(gold - n);
            case MAX -> c * Math.clamp(alpha * (gold - n) + delta, 0f, 1f);
        };
    }
}
