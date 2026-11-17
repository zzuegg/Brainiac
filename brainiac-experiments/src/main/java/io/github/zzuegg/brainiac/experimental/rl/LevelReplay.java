package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Prioritized Level Replay (Jiang, Grefenstette and Rocktäschel,
 * <a href="https://arxiv.org/abs/2010.03934">2010.03934</a>), the replay half of unsupervised
 * environment design (Robust PLR, <a href="https://arxiv.org/abs/2110.02439">2110.02439</a>):
 * a curriculum over procedurally generated levels. After every episode the learner reports a
 * score of how much it still has to learn on that level (the paper uses the mean absolute TD
 * or GAE error). The next level is either a new one from the generator, or a seen one drawn
 * by the mixture {@code (1 - rho) P_score + rho P_staleness}, where {@code P_score} ranks
 * levels by score with temperature {@code beta} ({@code (1 / rank)^(1 / beta)}) and
 * {@code P_staleness} favours levels not visited for long, so stale scores get refreshed.
 * Levels already mastered and levels that are hopeless both score low and are skipped.
 */
public final class LevelReplay {

    private final int levels;
    private final SplittableRandom random;
    private final List<Integer> seen = new ArrayList<>();
    private final double[] score;
    private final long[] lastPlayed;
    private double replayProbability = 0.5;
    private double temperature = 0.1;
    private double staleness = 0.1;
    private long episodes;
    private int nextNew;

    private LevelReplay(int levels, long seed) {
        this.levels = levels;
        this.random = new SplittableRandom(seed);
        this.score = new double[levels];
        this.lastPlayed = new long[levels];
    }

    /**
     * @param levels the number of levels the generator can make (ids {@code 0..levels-1})
     * @param seed the seed
     * @return the sampler
     */
    public static LevelReplay of(int levels, long seed) {
        return new LevelReplay(levels, seed);
    }

    /**
     * @param p the chance to replay a seen level while unseen ones remain
     * @return this sampler
     */
    public LevelReplay replayProbability(double p) {
        this.replayProbability = p;
        return this;
    }

    /**
     * @param beta the rank temperature (lower: greedier)
     * @return this sampler
     */
    public LevelReplay temperature(double beta) {
        this.temperature = beta;
        return this;
    }

    /**
     * @param rho the weight of the staleness distribution
     * @return this sampler
     */
    public LevelReplay staleness(double rho) {
        this.staleness = rho;
        return this;
    }

    /** @return the levels seen so far */
    public int seenLevels() {
        return seen.size();
    }

    /**
     * @param level a level
     * @return its current score
     */
    public double score(int level) {
        return score[level];
    }

    /** @return the next level to play */
    public int next() {
        boolean unseenLeft = nextNew < levels;
        if (seen.isEmpty() || (unseenLeft && random.nextDouble() >= replayProbability)) {
            return nextNew++;
        }
        return replay();
    }

    private int replay() {
        List<Integer> order = new ArrayList<>(seen);
        order.sort(Comparator.comparingDouble((Integer l) -> -score[l]));
        double[] p = new double[order.size()];
        double sumScore = 0;
        double sumStale = 0;
        for (int r = 0; r < p.length; r++) {
            sumScore += Math.pow(1.0 / (r + 1), 1 / temperature);
            sumStale += episodes - lastPlayed[order.get(r)];
        }
        double u = random.nextDouble();
        for (int r = 0; r < p.length; r++) {
            double ps = Math.pow(1.0 / (r + 1), 1 / temperature) / sumScore;
            double pc = sumStale > 0 ? (episodes - lastPlayed[order.get(r)]) / sumStale : 1.0 / p.length;
            u -= (1 - staleness) * ps + staleness * pc;
            if (u <= 0) {
                return order.get(r);
            }
        }
        return order.getLast();
    }

    /**
     * Reports a finished episode.
     *
     * @param level the level played
     * @param learningPotential its score, such as the mean absolute TD error
     */
    public void update(int level, double learningPotential) {
        episodes++;
        if (!seen.contains(level)) {
            seen.add(level);
        }
        score[level] = learningPotential;
        lastPlayed[level] = episodes;
    }
}
