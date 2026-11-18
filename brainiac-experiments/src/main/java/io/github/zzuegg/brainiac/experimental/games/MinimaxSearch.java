package io.github.zzuegg.brainiac.experimental.games;

import io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Depth-limited minimax search for two-player zero-sum games ({@link TwoPlayerGame}):
 *
 * <ul>
 *   <li>{@link Method#NEGAMAX}: full-width negamax, the node-count baseline;</li>
 *   <li>{@link Method#ALPHA_BETA}: alpha-beta pruning (Knuth &amp; Moore, AI 1975) with iterative
 *       deepening, a transposition table that stores bounds and best moves, and history-heuristic
 *       move ordering (the previous iteration's best move first, then moves that caused cut-offs),
 *       which brings the effective branching factor towards {@code sqrt(b)};</li>
 *   <li>{@link Method#ALPHA_BETA_PLAIN}: alpha-beta with the natural move order and no table,
 *       to separate pruning from ordering.</li>
 * </ul>
 *
 * <p>The evaluator scores positions from the side to move in {@code [-1, 1]}; finished games
 * score {@code +-(1 + remaining depth / 1000)} so that faster wins are preferred. Every search
 * also returns the principal variation's leaf, which {@link TdLeaf} trains on. Chance games use
 * {@link Expectimax}.
 */
public final class MinimaxSearch {

    /** The search variant. */
    public enum Method {
        /** full-width negamax */
        NEGAMAX,
        /** alpha-beta without ordering or table */
        ALPHA_BETA_PLAIN,
        /** alpha-beta with iterative deepening, a transposition table and history ordering */
        ALPHA_BETA
    }

    /** Scores a position for the side to move. */
    @FunctionalInterface
    public interface Evaluator {
        /**
         * @param position an unfinished position
         * @return its value for the side to move, in {@code [-1, 1]}
         */
        double evaluate(TwoPlayerGame position);
    }

    /**
     * A search result.
     *
     * @param move the best move at the root
     * @param value its value for the side to move
     * @param nodes positions visited (all iterations)
     * @param leaf the leaf of the principal variation
     */
    public record Result(int move, double value, long nodes, TwoPlayerGame leaf) {
    }

    private record Scored(double value, TwoPlayerGame leaf) {
    }

    private record Entry(int depth, double value, int flag, int move, TwoPlayerGame leaf) {
    }

    private static final int EXACT = 0;
    private static final int LOWER = 1;
    private static final int UPPER = 2;

    private final Method method;
    private final Evaluator evaluator;
    private int depth = 4;
    private long nodes;
    private final Map<TwoPlayerGame, Entry> table = new HashMap<>();
    private int[] history = new int[0];

    private MinimaxSearch(Method method, Evaluator evaluator) {
        this.method = Objects.requireNonNull(method, "method");
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
    }

    /**
     * @param method the variant
     * @param evaluator the leaf evaluation
     * @return the search
     */
    public static MinimaxSearch of(Method method, Evaluator evaluator) {
        return new MinimaxSearch(method, evaluator);
    }

    /** @param d the search depth in plies @return this search */
    public MinimaxSearch depth(int d) {
        this.depth = Math.max(1, d);
        return this;
    }

    /**
     * Searches a position.
     *
     * @param root an unfinished position
     * @return the best move, its value, the nodes visited and the principal leaf
     */
    public Result search(TwoPlayerGame root) {
        nodes = 0;
        table.clear();
        history = new int[root.actions()];
        if (method != Method.ALPHA_BETA) {
            return root(root, depth);
        }
        Result last = null;
        for (int d = 1; d <= depth; d++) {
            last = root(root, d);
        }
        return new Result(last.move(), last.value(), nodes, last.leaf());
    }

    /**
     * @param plies the search depth
     * @return a player that moves by this search
     */
    public TwoPlayerGame.Player player(int plies) {
        return g -> depth(plies).search(g).move();
    }

    private Result root(TwoPlayerGame root, int d) {
        int[] moves = order(root, d);
        double best = Double.NEGATIVE_INFINITY;
        int bestMove = moves[0];
        TwoPlayerGame bestLeaf = root;
        double alpha = Double.NEGATIVE_INFINITY;
        nodes++;
        for (int m : moves) {
            Scored s = method == Method.NEGAMAX ? negamax(root.play(m), d - 1)
                    : alphaBeta(root.play(m), d - 1, Double.NEGATIVE_INFINITY, -alpha);
            double v = -s.value();
            if (v > best) {
                best = v;
                bestMove = m;
                bestLeaf = s.leaf();
            }
            alpha = Math.max(alpha, v);
        }
        if (method == Method.ALPHA_BETA) {
            table.put(root, new Entry(d, best, EXACT, bestMove, bestLeaf));
        }
        return new Result(bestMove, best, nodes, bestLeaf);
    }

    private double terminal(TwoPlayerGame g, int remaining) {
        int w = g.winner();
        if (w == 0) {
            return 0;
        }
        double magnitude = 1 + remaining / 1000.0;
        return w == g.toMove() ? magnitude : -magnitude;
    }

    private Scored negamax(TwoPlayerGame g, int d) {
        nodes++;
        if (g.over()) {
            return new Scored(terminal(g, d), g);
        }
        if (d == 0) {
            return new Scored(evaluator.evaluate(g), g);
        }
        boolean[] legal = g.legal();
        double best = Double.NEGATIVE_INFINITY;
        TwoPlayerGame leaf = g;
        for (int m = 0; m < legal.length; m++) {
            if (legal[m]) {
                Scored s = negamax(g.play(m), d - 1);
                if (-s.value() > best) {
                    best = -s.value();
                    leaf = s.leaf();
                }
            }
        }
        return new Scored(best, leaf);
    }

    private int[] order(TwoPlayerGame g, int d) {
        boolean[] legal = g.legal();
        int count = 0;
        for (boolean b : legal) {
            count += b ? 1 : 0;
        }
        int[] moves = new int[count];
        int k = 0;
        for (int m = 0; m < legal.length; m++) {
            if (legal[m]) {
                moves[k++] = m;
            }
        }
        if (method != Method.ALPHA_BETA) {
            return moves;
        }
        Entry e = table.get(g);
        int hint = e == null ? -1 : e.move();
        Integer[] boxed = new Integer[count];
        for (int i = 0; i < count; i++) {
            boxed[i] = moves[i];
        }
        java.util.Arrays.sort(boxed, (x, y) -> {
            if (x == hint) {
                return -1;
            }
            if (y == hint) {
                return 1;
            }
            return Integer.compare(history[y], history[x]);
        });
        for (int i = 0; i < count; i++) {
            moves[i] = boxed[i];
        }
        return moves;
    }

    private Scored alphaBeta(TwoPlayerGame g, int d, double alpha, double beta) {
        nodes++;
        if (g.over()) {
            return new Scored(terminal(g, d), g);
        }
        if (d == 0) {
            return new Scored(evaluator.evaluate(g), g);
        }
        double alpha0 = alpha;
        if (method == Method.ALPHA_BETA) {
            Entry e = table.get(g);
            if (e != null && e.depth() >= d) {
                if (e.flag() == EXACT) {
                    return new Scored(e.value(), e.leaf());
                }
                if (e.flag() == LOWER) {
                    alpha = Math.max(alpha, e.value());
                } else {
                    beta = Math.min(beta, e.value());
                }
                if (alpha >= beta) {
                    return new Scored(e.value(), e.leaf());
                }
            }
        }
        double best = Double.NEGATIVE_INFINITY;
        int bestMove = -1;
        TwoPlayerGame leaf = g;
        for (int m : order(g, d)) {
            Scored s = alphaBeta(g.play(m), d - 1, -beta, -alpha);
            double v = -s.value();
            if (v > best) {
                best = v;
                bestMove = m;
                leaf = s.leaf();
            }
            alpha = Math.max(alpha, v);
            if (alpha >= beta) {
                if (method == Method.ALPHA_BETA) {
                    history[m] += d * d;
                }
                break;
            }
        }
        if (method == Method.ALPHA_BETA) {
            int flag = best <= alpha0 ? UPPER : best >= beta ? LOWER : EXACT;
            table.put(g, new Entry(d, best, flag, bestMove, leaf));
        }
        return new Scored(best, leaf);
    }

    /**
     * Depth-limited expectimax for games with chance nodes (Michie, 1966): max nodes for the
     * player to move (negamax form) and probability-weighted averages at chance nodes.
     */
    public static final class Expectimax {
        private final ChanceGame.Evaluator evaluator;
        private long nodes;

        private Expectimax(ChanceGame.Evaluator evaluator) {
            this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        }

        /**
         * @param evaluator the leaf evaluation for the side to move
         * @return the search
         */
        public static Expectimax of(ChanceGame.Evaluator evaluator) {
            return new Expectimax(evaluator);
        }

        /** @return nodes visited by the last search */
        public long nodes() {
            return nodes;
        }

        /**
         * @param g a decision position (not a chance node, not over)
         * @param depth decision plies
         * @return the best action
         */
        public int bestMove(ChanceGame g, int depth) {
            nodes = 0;
            boolean[] legal = g.legal();
            double best = Double.NEGATIVE_INFINITY;
            int move = -1;
            for (int a = 0; a < legal.length; a++) {
                if (legal[a]) {
                    double v = value(g.play(a), depth - 1, g.toMove());
                    if (v > best) {
                        best = v;
                        move = a;
                    }
                }
            }
            return move;
        }

        /** The value of {@code g} for player {@code me}. */
        private double value(ChanceGame g, int depth, int me) {
            nodes++;
            if (g.over()) {
                return g.winner() == 0 ? 0 : g.winner() == me ? 1 : -1;
            }
            if (g.chanceNode()) {
                double[] p = g.outcomeProbabilities();
                double v = 0;
                for (int k = 0; k < p.length; k++) {
                    if (p[k] > 0) {
                        v += p[k] * value(g.outcome(k), depth, me);
                    }
                }
                return v;
            }
            if (depth <= 0) {
                double e = evaluator.evaluate(g);
                return g.toMove() == me ? e : -e;
            }
            boolean[] legal = g.legal();
            double best = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < legal.length; a++) {
                if (legal[a]) {
                    double v = value(g.play(a), depth - 1, me);
                    best = Math.max(best, g.toMove() == me ? v : -v);
                }
            }
            return g.toMove() == me ? best : -best;
        }
    }
}
