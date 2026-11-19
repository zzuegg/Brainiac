package io.github.zzuegg.brainiac.experimental.explain;

import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * A cooperative game over feature coalitions, the object every Shapley-style explanation is
 * computed from. Evaluation is batched so that a model is called once for many coalitions.
 */
@FunctionalInterface
public interface CoalitionGame {

    /**
     * Values a batch of coalitions.
     *
     * @param coalitions one membership vector per coalition
     * @return one value per coalition
     */
    double[] values(boolean[][] coalitions);

    /**
     * Values one coalition.
     *
     * @param coalition the membership vector
     * @return its value
     */
    default double value(boolean[] coalition) {
        return values(new boolean[][] {coalition})[0];
    }

    /**
     * The interventional (marginal) game of a model: {@code v(S) = mean_z f(x_S, z_{-S})}.
     *
     * @param model the batch model
     * @param x the row explained
     * @param background the reference rows
     * @return the game
     */
    static CoalitionGame interventional(Function<double[][], double[]> model, double[] x, double[][] background) {
        return coalitions -> {
            int b = background.length;
            double[][] rows = new double[coalitions.length * b][];
            for (int c = 0; c < coalitions.length; c++) {
                for (int k = 0; k < b; k++) {
                    double[] row = background[k].clone();
                    for (int j = 0; j < x.length; j++) {
                        if (coalitions[c][j]) {
                            row[j] = x[j];
                        }
                    }
                    rows[c * b + k] = row;
                }
            }
            double[] f = model.apply(rows);
            double[] out = new double[coalitions.length];
            for (int c = 0; c < coalitions.length; c++) {
                double s = 0;
                for (int k = 0; k < b; k++) {
                    s += f[c * b + k];
                }
                out[c] = s / b;
            }
            return out;
        };
    }

    /**
     * Wraps a game so that every coalition evaluation is counted.
     *
     * @param game the game
     * @param counter incremented by the number of coalitions valued
     * @return the counting game
     */
    static CoalitionGame counting(CoalitionGame game, AtomicLong counter) {
        return coalitions -> {
            counter.addAndGet(coalitions.length);
            return game.values(coalitions);
        };
    }

    /**
     * Converts a bit mask to a membership vector.
     *
     * @param mask the bits
     * @param p the number of players
     * @return the vector
     */
    static boolean[] members(long mask, int p) {
        boolean[] s = new boolean[p];
        for (int j = 0; j < p; j++) {
            s[j] = (mask & (1L << j)) != 0;
        }
        return s;
    }

    /**
     * Values all {@code 2^p} coalitions, indexed by bit mask.
     *
     * @param game the game
     * @param p the number of players, at most about 20
     * @return the table
     */
    static double[] table(CoalitionGame game, int p) {
        int full = 1 << p;
        boolean[][] all = new boolean[full][];
        for (int s = 0; s < full; s++) {
            all[s] = members(s, p);
        }
        return game.values(all);
    }
}
