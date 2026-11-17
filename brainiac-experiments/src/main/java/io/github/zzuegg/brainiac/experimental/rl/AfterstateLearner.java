package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A self-play learner for {@link BoardGame}s that learns the value of afterstates (the
 * position right after a move) for the player who made the move, by temporal differences,
 * as in Sutton and Barto's tic-tac-toe player and TD-Gammon. One table plays both sides, so
 * every game teaches it from both perspectives; {@link #snapshot} freezes a copy as a league
 * opponent.
 */
public final class AfterstateLearner implements GamePlayer {

    private final Map<String, Double> values;
    private double alpha = 0.2;
    private double epsilon = 0.1;
    private boolean learning = true;

    private AfterstateLearner(Map<String, Double> values) {
        this.values = values;
    }

    /** @return a fresh learner (unseen afterstates are worth 0.5) */
    public static AfterstateLearner create() {
        return new AfterstateLearner(new HashMap<>());
    }

    /**
     * @param rate the TD step size
     * @return this learner
     */
    public AfterstateLearner learningRate(double rate) {
        this.alpha = rate;
        return this;
    }

    /**
     * @param e the exploration rate while learning
     * @return this learner
     */
    public AfterstateLearner epsilon(double e) {
        this.epsilon = e;
        return this;
    }

    /** @return a frozen, greedy copy */
    public AfterstateLearner snapshot() {
        AfterstateLearner s = new AfterstateLearner(new HashMap<>(values));
        s.learning = false;
        s.epsilon = 0;
        return s;
    }

    /** @return the number of afterstates valued */
    public int size() {
        return values.size();
    }

    private double value(BoardGame afterstate, int mover) {
        int w = afterstate.winner();
        if (w == mover) {
            return 1;
        }
        if (w == -1) {
            return 0.5;
        }
        if (w != 0) {
            return 0;
        }
        return values.getOrDefault(afterstate.key(), 0.5);
    }

    private int greedy(BoardGame position, SplittableRandom random) {
        int mover = position.toMove();
        List<Integer> best = new ArrayList<>();
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int m : position.legalMoves()) {
            double v = value(position.play(m), mover);
            if (v > bestValue + 1e-12) {
                bestValue = v;
                best.clear();
                best.add(m);
            } else if (Math.abs(v - bestValue) <= 1e-12) {
                best.add(m);
            }
        }
        return best.get(random.nextInt(best.size()));
    }

    @Override
    public int move(BoardGame position, SplittableRandom random) {
        if (learning && random.nextDouble() < epsilon) {
            List<Integer> moves = position.legalMoves();
            return moves.get(random.nextInt(moves.size()));
        }
        return greedy(position, random);
    }

    /**
     * Plays one game against itself and learns from it: each player's previous afterstate
     * moves towards the value of its next afterstate (or the final result).
     *
     * @param start the first position
     * @param random the random source
     * @return the winner
     */
    public int selfPlay(BoardGame start, SplittableRandom random) {
        return train(start, this, this, random);
    }

    /**
     * Plays one game in which this learner takes one or both sides (an opponent can be any
     * player, such as a league snapshot) and learns for its own sides.
     *
     * @param start the first position
     * @param first player 1
     * @param second player 2
     * @param random the random source
     * @return the winner
     */
    public int train(BoardGame start, GamePlayer first, GamePlayer second, SplittableRandom random) {
        Map<Integer, BoardGame> previous = new HashMap<>();
        BoardGame g = start;
        while (!g.isOver()) {
            int mover = g.toMove();
            GamePlayer p = mover == 1 ? first : second;
            BoardGame after = g.play(p.move(g, random));
            if (p == this) {
                BoardGame prev = previous.get(mover);
                if (prev != null) {
                    update(prev, mover, value(after, mover));
                }
                previous.put(mover, after);
            }
            g = after;
        }
        int w = g.winner();
        for (Map.Entry<Integer, BoardGame> e : previous.entrySet()) {
            int player = e.getKey();
            double result = w == player ? 1 : w == -1 ? 0.5 : 0;
            update(e.getValue(), player, result);
        }
        return w;
    }

    private void update(BoardGame afterstate, int mover, double target) {
        if (!learning || afterstate.isOver()) {
            return;
        }
        double v = value(afterstate, mover);
        values.put(afterstate.key(), v + alpha * (target - v));
    }
}
