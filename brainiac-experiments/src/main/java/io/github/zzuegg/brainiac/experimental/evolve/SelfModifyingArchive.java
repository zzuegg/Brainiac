package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * The open-ended loop of the Darwin Gödel Machine (Zhang et al.,
 * <a href="https://arxiv.org/abs/2505.22954">2505.22954</a>): coding agents that rewrite
 * their own scaffold, kept in a growing archive. Each generation picks parents from the
 * archive, with probability proportional to
 * {@code sigmoid(lambda (score - alpha0)) / (1 + children)}, where {@code alpha0} is the mean
 * of the three best scores, so good agents and agents that have not been explored much are
 * both chosen; each parent produces a child by <em>modifying itself</em> (the agent decides
 * the edit, as DGM's agents edit their own code); the child is validated on the benchmark,
 * and every valid child joins the archive, even one that scores worse than its parent. Those
 * weaker agents are the stepping stones that greedy hill-climbing throws away
 * ({@link #hillClimb}).
 *
 * @param <A> the agent type
 */
public final class SelfModifyingArchive<A> {

    /**
     * An archived agent.
     *
     * @param id its index
     * @param agent the agent
     * @param score its benchmark score
     * @param parent the parent's id, -1 for the root
     * @param <A> the agent type
     */
    public record Member<A>(int id, A agent, double score, int parent) {
    }

    private final Function<A, OptionalDouble> evaluate;
    private final BiFunction<A, SplittableRandom, A> selfModify;
    private final List<Member<A>> archive = new ArrayList<>();
    private final List<Integer> children = new ArrayList<>();
    private double lambda = 10;
    private double keepAbove = Double.NEGATIVE_INFINITY;
    private int evaluations;

    private SelfModifyingArchive(A initial, Function<A, OptionalDouble> evaluate,
            BiFunction<A, SplittableRandom, A> selfModify) {
        this.evaluate = Objects.requireNonNull(evaluate);
        this.selfModify = Objects.requireNonNull(selfModify);
        OptionalDouble s = evaluate.apply(initial);
        evaluations++;
        if (s.isEmpty()) {
            throw new IllegalArgumentException("the initial agent is not valid");
        }
        archive.add(new Member<>(0, initial, s.getAsDouble(), -1));
        children.add(0);
    }

    /**
     * @param initial the first agent
     * @param evaluate its benchmark score, or empty if the agent is broken
     * @param selfModify the agent's own edit of itself
     * @param <A> the agent type
     * @return an archive holding the initial agent
     */
    public static <A> SelfModifyingArchive<A> of(A initial, Function<A, OptionalDouble> evaluate,
            BiFunction<A, SplittableRandom, A> selfModify) {
        return new SelfModifyingArchive<>(initial, evaluate, selfModify);
    }

    /**
     * @param sharpness the sigmoid's slope in parent selection (10 in DGM)
     * @return this archive
     */
    public SelfModifyingArchive<A> sharpness(double sharpness) {
        this.lambda = sharpness;
        return this;
    }

    /**
     * @param minimum the score a child needs to be archived (DGM keeps children that still
     *     compile and edit code; by default every valid child is kept)
     * @return this archive
     */
    public SelfModifyingArchive<A> keepAbove(double minimum) {
        this.keepAbove = minimum;
        return this;
    }

    /**
     * Runs generations of self-modification.
     *
     * @param generations how many
     * @param childrenPerGeneration children per generation
     * @param random the random source
     * @return the best score after each generation
     */
    public double[] run(int generations, int childrenPerGeneration, SplittableRandom random) {
        double[] best = new double[generations];
        for (int g = 0; g < generations; g++) {
            double[] weights = selectionWeights();
            List<Member<A>> born = new ArrayList<>();
            for (int c = 0; c < childrenPerGeneration; c++) {
                int p = sample(weights, random);
                A child = selfModify.apply(archive.get(p).agent(), random);
                OptionalDouble s = evaluate.apply(child);
                evaluations++;
                children.set(p, children.get(p) + 1);
                if (s.isPresent() && s.getAsDouble() > keepAbove) {
                    born.add(new Member<>(archive.size() + born.size(), child, s.getAsDouble(), p));
                }
            }
            for (Member<A> m : born) {
                archive.add(m);
                children.add(0);
            }
            best[g] = best().score();
        }
        return best;
    }

    private double[] selectionWeights() {
        List<Double> scores = new ArrayList<>();
        for (Member<A> m : archive) {
            scores.add(m.score());
        }
        scores.sort(Comparator.reverseOrder());
        double alpha0 = 0;
        int top = Math.min(3, scores.size());
        for (int i = 0; i < top; i++) {
            alpha0 += scores.get(i) / top;
        }
        double[] w = new double[archive.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = 1 / (1 + Math.exp(-lambda * (archive.get(i).score() - alpha0))) / (1 + children.get(i));
        }
        return w;
    }

    private static int sample(double[] w, SplittableRandom random) {
        double total = 0;
        for (double x : w) {
            total += x;
        }
        double u = random.nextDouble() * total;
        for (int i = 0; i < w.length; i++) {
            u -= w[i];
            if (u <= 0) {
                return i;
            }
        }
        return w.length - 1;
    }

    /** @return the best agent so far */
    public Member<A> best() {
        return archive.stream().max(Comparator.comparingDouble(Member::score)).orElseThrow();
    }

    /** @return the archive, in the order agents were added */
    public List<Member<A>> archive() {
        return List.copyOf(archive);
    }

    /** @return how many agents were evaluated */
    public int evaluations() {
        return evaluations;
    }

    /**
     * The baseline without an archive: keep only the current best agent and replace it by
     * a child only when the child scores strictly higher.
     *
     * @param initial the first agent
     * @param evaluate its score, or empty if broken
     * @param selfModify the agent's edit of itself
     * @param generations how many generations
     * @param childrenPerGeneration children per generation
     * @param random the random source
     * @param <A> the agent type
     * @return where the climb ended, and the best score after each generation
     */
    public static <A> Climb<A> hillClimb(A initial, Function<A, OptionalDouble> evaluate,
            BiFunction<A, SplittableRandom, A> selfModify, int generations, int childrenPerGeneration,
            SplittableRandom random) {
        A current = initial;
        double score = evaluate.apply(initial).orElseThrow();
        double[] best = new double[generations];
        for (int g = 0; g < generations; g++) {
            A bestChild = null;
            double bestChildScore = score;
            for (int c = 0; c < childrenPerGeneration; c++) {
                A child = selfModify.apply(current, random);
                OptionalDouble s = evaluate.apply(child);
                if (s.isPresent() && s.getAsDouble() > bestChildScore) {
                    bestChild = child;
                    bestChildScore = s.getAsDouble();
                }
            }
            if (bestChild != null) {
                current = bestChild;
                score = bestChildScore;
            }
            best[g] = score;
        }
        return new Climb<>(current, best);
    }

    /**
     * The outcome of {@link #hillClimb}.
     *
     * @param agent the agent the climb ended at
     * @param best the best score after each generation
     * @param <A> the agent type
     */
    public record Climb<A>(A agent, double[] best) {
    }
}
