package io.github.zzuegg.brainiac.experimental.games;

import java.util.SplittableRandom;

/**
 * A small dice race for expectimax: two players race to {@code target}; on a turn the player
 * either rolls one die (gain its value) or two dice (gain their sum, but nothing if either die
 * shows a one), so the risky roll gains more on average ({@code 5.56} against {@code 3.5}) and
 * the right choice depends on both scores.
 */
public final class DiceRace implements ChanceGame {

    /** Gains of the two-dice roll with their probabilities (index 0: bust). */
    private static final double[] TWO_DICE = new double[13];
    private static final double[] ONE_DIE = new double[7];

    static {
        for (int a = 1; a <= 6; a++) {
            ONE_DIE[a] = 1 / 6.0;
            for (int b = 1; b <= 6; b++) {
                TWO_DICE[a == 1 || b == 1 ? 0 : a + b] += 1 / 36.0;
            }
        }
    }

    private final int target;
    private final int first;
    private final int second;
    private final int toMove;
    private final int pending;

    private DiceRace(int target, int first, int second, int toMove, int pending) {
        this.target = target;
        this.first = first;
        this.second = second;
        this.toMove = toMove;
        this.pending = pending;
    }

    /**
     * @param target the score to reach
     * @return the start position
     */
    public static DiceRace start(int target) {
        return new DiceRace(target, 0, 0, 1, -1);
    }

    /** @return the first player's score */
    public int firstScore() {
        return first;
    }

    /** @return the second player's score */
    public int secondScore() {
        return second;
    }

    /** @return the target */
    public int target() {
        return target;
    }

    @Override
    public int actions() {
        return 2;
    }

    @Override
    public boolean[] legal() {
        return new boolean[] {true, true};
    }

    @Override
    public DiceRace play(int action) {
        return new DiceRace(target, first, second, toMove, action);
    }

    @Override
    public boolean chanceNode() {
        return pending >= 0;
    }

    @Override
    public double[] outcomeProbabilities() {
        return (pending == 0 ? ONE_DIE : TWO_DICE).clone();
    }

    @Override
    public DiceRace outcome(int gain) {
        int f = first + (toMove == 1 ? gain : 0);
        int s = second + (toMove == -1 ? gain : 0);
        return new DiceRace(target, f, s, -toMove, -1);
    }

    /**
     * @param random the dice
     * @return the position after nature rolls
     */
    public DiceRace roll(SplittableRandom random) {
        double[] p = outcomeProbabilities();
        double u = random.nextDouble();
        for (int k = 0; k < p.length; k++) {
            u -= p[k];
            if (u <= 0) {
                return outcome(k);
            }
        }
        return outcome(p.length - 1);
    }

    @Override
    public boolean over() {
        return first >= target || second >= target;
    }

    @Override
    public int winner() {
        return first >= target ? 1 : second >= target ? -1 : 0;
    }

    @Override
    public int toMove() {
        return toMove;
    }

    /**
     * The score difference from the side to move, scaled by the target.
     *
     * @param g a dice race
     * @return the evaluation
     */
    public static double scoreDifference(ChanceGame g) {
        DiceRace d = (DiceRace) g;
        double diff = d.toMove == 1 ? d.first - d.second : d.second - d.first;
        return Math.max(-1, Math.min(1, diff / d.target));
    }

    @Override
    public String toString() {
        return "DiceRace(" + first + ":" + second + ", to move " + toMove + ")";
    }
}
