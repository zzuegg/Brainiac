package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Go-Explore, phase 1 (Ecoffet et al., <a href="https://arxiv.org/abs/1901.10995">1901.10995</a>,
 * "First return, then explore", Nature 2021): keep an archive of <em>cells</em> (coarse
 * descriptions of states) with the best action sequence that reached each one; repeatedly
 * pick a promising cell (rarely chosen ones are preferred), <em>return</em> to it by
 * replaying its actions from the same deterministic start, then <em>explore</em> from there
 * with random, sticky actions, adding every new cell or better path to the archive. Returning
 * before exploring avoids the detachment and derailment that make novelty bonuses fail on
 * hard-exploration problems.
 *
 * <p>The environment must be deterministic given its reset random source; the archive resets
 * it with the same seed every time. A path is better when it scores a higher return, or the
 * same return in fewer steps.
 */
public final class GoExplore {

    private final Function<float[], Object> cellOf;
    private final Map<Object, Cell> archive = new LinkedHashMap<>();
    private int explorationSteps = 50;
    private double repeat = 0.9;
    private final long resetSeed;
    private final SplittableRandom random;
    private Cell best;
    private long environmentSteps;

    /** An archive entry. */
    private static final class Cell {
        int[] actions;
        double score;
        boolean finished;
        int chosen;
        int seen;

        Cell(int[] actions, double score, boolean finished) {
            this.actions = actions;
            this.score = score;
            this.finished = finished;
        }

        boolean worseThan(double otherScore, int otherLength) {
            return otherScore > score + 1e-9 || (Math.abs(otherScore - score) <= 1e-9 && otherLength < actions.length);
        }
    }

    private GoExplore(Function<float[], Object> cellOf, long seed) {
        this.cellOf = Objects.requireNonNull(cellOf);
        this.random = new SplittableRandom(seed);
        this.resetSeed = seed;
    }

    /**
     * @param cellOf maps an observation to its cell (a value with equals and hashCode)
     * @param seed the seed of the exploration and of the fixed reset
     * @return an empty archive
     */
    public static GoExplore of(Function<float[], Object> cellOf, long seed) {
        return new GoExplore(cellOf, seed);
    }

    /**
     * A cell function that divides every observation dimension into equal bins.
     *
     * @param low the lower bounds
     * @param high the upper bounds
     * @param bins bins per dimension
     * @return the cell function
     */
    public static Function<float[], Object> grid(float[] low, float[] high, int bins) {
        return x -> {
            int[] c = new int[x.length];
            for (int i = 0; i < x.length; i++) {
                double f = (x[i] - low[i]) / (high[i] - low[i]);
                c[i] = (int) Math.max(0, Math.min(bins - 1, Math.floor(f * bins)));
            }
            return Arrays.toString(c);
        };
    }

    /**
     * @param steps random steps after each return
     * @return this archive
     */
    public GoExplore explorationSteps(int steps) {
        this.explorationSteps = steps;
        return this;
    }

    /**
     * @param probability the chance to repeat the previous random action
     * @return this archive
     */
    public GoExplore stickyActions(double probability) {
        this.repeat = probability;
        return this;
    }

    /**
     * Runs return-then-explore iterations.
     *
     * @param env a deterministic environment with discrete actions
     * @param iterations how many cells to return to and explore from
     * @return this archive
     */
    public GoExplore explore(Environment env, int iterations) {
        if (!(env.actionSpace() instanceof Space.Discrete d)) {
            throw new IllegalArgumentException("Go-Explore needs discrete actions");
        }
        int actions = d.n();
        if (archive.isEmpty()) {
            float[] start = env.reset(new SplittableRandom(resetSeed));
            archive.put(cellOf.apply(start), new Cell(new int[0], 0, false));
        }
        for (int it = 0; it < iterations; it++) {
            Cell from = select();
            from.chosen++;
            float[] obs = env.reset(new SplittableRandom(resetSeed));
            double score = 0;
            List<Integer> path = new ArrayList<>(from.actions.length + explorationSteps);
            boolean over = false;
            for (int a : from.actions) {
                Environment.Step s = env.step(new float[] {a});
                environmentSteps++;
                score += s.reward();
                path.add(a);
                obs = s.observation();
                over = s.done();
            }
            int a = random.nextInt(actions);
            for (int k = 0; k < explorationSteps && !over; k++) {
                if (random.nextDouble() >= repeat) {
                    a = random.nextInt(actions);
                }
                Environment.Step s = env.step(new float[] {a});
                environmentSteps++;
                score += s.reward();
                path.add(a);
                obs = s.observation();
                over = s.done();
                record(cellOf.apply(obs), path, score, s.terminated());
            }
        }
        return this;
    }

    private void record(Object key, List<Integer> path, double score, boolean finished) {
        Cell c = archive.get(key);
        if (c == null || c.worseThan(score, path.size())) {
            int[] actions = path.stream().mapToInt(Integer::intValue).toArray();
            if (c == null) {
                c = new Cell(actions, score, finished);
                archive.put(key, c);
            } else {
                c.actions = actions;
                c.score = score;
                c.finished = finished;
                c.chosen = 0;
            }
        }
        c.seen++;
        if (finished && (best == null || best.worseThan(score, path.size()))) {
            best = new Cell(path.stream().mapToInt(Integer::intValue).toArray(), score, true);
        }
    }

    private Cell select() {
        List<Cell> open = new ArrayList<>();
        double total = 0;
        for (Cell c : archive.values()) {
            if (!c.finished) {
                open.add(c);
                total += weight(c);
            }
        }
        double u = random.nextDouble() * total;
        for (Cell c : open) {
            u -= weight(c);
            if (u <= 0) {
                return c;
            }
        }
        return open.getLast();
    }

    private static double weight(Cell c) {
        return 1 / Math.sqrt(c.chosen + 1) + 1 / Math.sqrt(c.seen + 1);
    }

    /** @return the number of cells found */
    public int cells() {
        return archive.size();
    }

    /** @return environment steps taken, returning included */
    public long environmentSteps() {
        return environmentSteps;
    }

    /** @return whether an episode was ever finished (terminated, not cut off) */
    public boolean solved() {
        return best != null;
    }

    /** @return the return of the best finished episode, or negative infinity */
    public double bestReturn() {
        return best == null ? Double.NEGATIVE_INFINITY : best.score;
    }

    /** @return the actions of the best finished episode, or an empty array */
    public int[] bestActions() {
        return best == null ? new int[0] : best.actions.clone();
    }

    /**
     * Replays the best finished episode from the fixed start.
     *
     * @param env the environment
     * @return the return obtained
     */
    public double replayBest(Environment env) {
        env.reset(new SplittableRandom(resetSeed));
        double score = 0;
        for (int a : bestActions()) {
            Environment.Step s = env.step(new float[] {a});
            score += s.reward();
            if (s.done()) {
                break;
            }
        }
        return score;
    }
}
