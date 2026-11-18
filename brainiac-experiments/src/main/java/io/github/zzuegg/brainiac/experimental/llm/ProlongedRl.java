package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * ProRL, prolonged reinforcement learning (Liu et al., "ProRL: prolonged reinforcement learning
 * expands reasoning boundaries in large language models",
 * <a href="https://arxiv.org/abs/2505.24864">arXiv:2505.24864</a>): GRPO run for far longer than
 * usual, kept stable by a KL penalty to a reference model and by a <em>reference reset</em> every
 * so often: the reference becomes a copy of the current policy and the optimizer state is
 * cleared, so the KL term stops pulling the policy back to where it started (which would stall
 * long training) while still bounding how far each phase can move. The paper adds DAPO's
 * decoupled clipping and a diverse task mix; both are set on the wrapped {@link GrpoTrainer} and
 * in the prompt source.
 *
 * <p>This is a schedule around an existing {@link GrpoTrainer}: it owns the trainer's reference
 * model and swaps the optimizer, and otherwise every step is the trainer's own.
 */
public final class ProlongedRl {
    private final GrpoTrainer grpo;
    private final WindowLm policy;
    private final WindowLm reference;
    private final Supplier<Optimizer> optimizers;
    private final int resetEvery;
    private int steps;
    private int resets;

    private ProlongedRl(GrpoTrainer grpo, WindowLm policy, int resetEvery, Supplier<Optimizer> optimizers) {
        this.grpo = grpo;
        this.policy = policy;
        this.reference = policy.copy();
        this.resetEvery = resetEvery;
        this.optimizers = optimizers;
    }

    /**
     * Wraps a trainer: its reference becomes a copy of {@code policy} that this schedule resets,
     * and its optimizer is replaced from {@code optimizers}.
     *
     * @param grpo the trainer of {@code policy}
     * @param policy the model the trainer updates
     * @param resetEvery steps between reference resets; 0 never resets
     * @param optimizers makes a fresh optimizer at the start and after every reset
     * @return the schedule
     */
    public static ProlongedRl of(GrpoTrainer grpo, WindowLm policy, int resetEvery, Supplier<Optimizer> optimizers) {
        ProlongedRl p = new ProlongedRl(Objects.requireNonNull(grpo), Objects.requireNonNull(policy), resetEvery,
                Objects.requireNonNull(optimizers));
        grpo.reference(p.reference).optimizer(optimizers.get());
        return p;
    }

    /**
     * One GRPO step, preceded by a reference reset when one is due.
     *
     * @param promptSource draws a prompt
     * @param reward the verifiable reward
     * @param random the random source
     * @return the trainer's statistics
     */
    public GrpoTrainer.Stats step(Function<SplittableRandom, int[]> promptSource, GrpoTrainer.Reward reward,
            SplittableRandom random) {
        if (resetEvery > 0 && steps > 0 && steps % resetEvery == 0) {
            reset();
        }
        steps++;
        return grpo.step(promptSource, reward, random);
    }

    /** Resets now: the reference becomes the current policy and the optimizer starts afresh. */
    public void reset() {
        reference.load(policy);
        grpo.optimizer(optimizers.get());
        resets++;
    }

    /** @return the resets so far */
    public int resets() {
        return resets;
    }

    /** @return the current reference model */
    public WindowLm reference() {
        return reference;
    }

    @Override
    public String toString() {
        return "ProlongedRl(resetEvery=" + resetEvery + ", steps=" + steps + ", resets=" + resets + ")";
    }
}
