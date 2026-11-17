package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Adds an {@link IntrinsicReward} to any environment, so every learner of the RL module
 * (tabular, DQN, PPO, ...) explores with curiosity without changing the learner:
 * {@code reward = extrinsic + coefficient * bonus}. The wrapper also counts the distinct
 * observations seen and the extrinsic reward separately, to measure exploration.
 *
 * <pre>{@code
 * Environment env = CuriosityEnvironment.of(GridWorld.of(...),
 *         RandomNetworkDistillation.of(cells, 16, 32, 1), 0.1f);
 * Tabular.qLearning(4).fit(env, 5_000);
 * }</pre>
 */
public final class CuriosityEnvironment implements Environment {

    private final Environment inner;
    private final IntrinsicReward bonus;
    private final float coefficient;
    private final Set<String> seen = new HashSet<>();
    private float[] last;
    private double extrinsic;
    private double intrinsic;
    private int goals;
    private boolean centered;
    private final AgentKit.Running bonuses = new AgentKit.Running();

    private CuriosityEnvironment(Environment inner, IntrinsicReward bonus, float coefficient) {
        this.inner = Objects.requireNonNull(inner);
        this.bonus = bonus;
        this.coefficient = coefficient;
    }

    /**
     * @param inner the environment
     * @param bonus the intrinsic reward, or {@code null} for none (only counting)
     * @param coefficient the weight of the bonus
     * @return the wrapped environment
     */
    public static CuriosityEnvironment of(Environment inner, IntrinsicReward bonus, float coefficient) {
        return new CuriosityEnvironment(inner, bonus, coefficient);
    }

    /**
     * Subtracts the running mean of the bonus, so that transitions more familiar than
     * average are penalized. Learners whose unvisited states start at value 0 (tabular,
     * without generalization) otherwise prefer revisiting states that still pay a small
     * positive bonus over trying unvisited ones.
     *
     * @param on whether to center the bonus
     * @return this environment
     */
    public CuriosityEnvironment centered(boolean on) {
        this.centered = on;
        return this;
    }

    @Override
    public Space observationSpace() {
        return inner.observationSpace();
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        last = inner.reset(random);
        seen.add(Arrays.toString(last));
        return last;
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        extrinsic += s.reward();
        if (s.reward() > 0) {
            goals++;
        }
        seen.add(Arrays.toString(s.observation()));
        float b = bonus == null ? 0 : bonus.observe(last, action, s.observation());
        intrinsic += b;
        if (centered && bonus != null) {
            bonuses.add(b);
            b -= (float) bonuses.mean();
        }
        last = s.observation();
        return new Step(s.observation(), s.reward() + coefficient * b, s.terminated(), s.truncated());
    }

    @Override
    public String render() {
        return inner.render();
    }

    /** @return the number of distinct observations seen so far */
    public int distinctObservations() {
        return seen.size();
    }

    /** @return the summed extrinsic reward */
    public double extrinsicReward() {
        return extrinsic;
    }

    /** @return the summed (unweighted) intrinsic reward */
    public double intrinsicReward() {
        return intrinsic;
    }

    /** @return how many steps had a positive extrinsic reward */
    public int positiveRewards() {
        return goals;
    }
}
