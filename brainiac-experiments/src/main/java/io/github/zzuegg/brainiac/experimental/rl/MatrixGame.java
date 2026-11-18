package io.github.zzuegg.brainiac.experimental.rl;

/**
 * A two-player zero-sum matrix (normal-form) game written as an {@link ExtensiveFormGame}:
 * player 0 picks a row, then player 1 picks a column without seeing it (one information set
 * each), and player 0 receives the entry. This lets every extensive-form solver, best response
 * and equilibrium learner run on matrix games unchanged; with one information set per player,
 * a {@link GameTree} policy table is just the two mixed strategies.
 */
public final class MatrixGame implements ExtensiveFormGame {

    private final String name;
    private final double[][] payoff;
    private final int rows;
    private final int columns;

    private MatrixGame(String name, double[][] payoff) {
        this.name = name;
        this.rows = payoff.length;
        this.columns = payoff[0].length;
        this.payoff = new double[rows][];
        for (int i = 0; i < rows; i++) {
            this.payoff[i] = payoff[i].clone();
        }
    }

    /**
     * @param name a label
     * @param payoff the payoff to the row player {@code [rows][columns]}
     * @return the game
     */
    public static MatrixGame of(String name, double[][] payoff) {
        return new MatrixGame(name, payoff);
    }

    /** @return matching pennies; the unique equilibrium is uniform, value 0 */
    public static MatrixGame matchingPennies() {
        return of("matching pennies", new double[][] {{1, -1}, {-1, 1}});
    }

    /** @return rock-paper-scissors; the unique equilibrium is uniform, value 0 */
    public static MatrixGame rockPaperScissors() {
        return of("rock-paper-scissors", new double[][] {{0, -1, 1}, {1, 0, -1}, {-1, 1, 0}});
    }

    /**
     * Biased rock-paper-scissors (winning with rock pays 2, with scissors 3), whose unique
     * equilibrium is not uniform, so learners cannot sit at the centre by symmetry.
     *
     * @return the game
     */
    public static MatrixGame biasedRockPaperScissors() {
        return of("biased RPS", new double[][] {{0, -1, 2}, {1, 0, -1}, {-3, 1, 0}});
    }

    /**
     * Rock-paper-scissors with skill: {@code 3 * levels} strategies, each a type (rock, paper or
     * scissors) at a skill level. The type decides who wins across types; the skill
     * difference adds a transitive bonus, and within one type the more skilled wins. The
     * game is non-transitive (naive self-play cycles between the types) but has a transitive
     * component (only the top skill of each type is in the equilibrium).
     *
     * @param levels skill levels per type
     * @return the game
     */
    public static MatrixGame rockPaperScissorsWithSkill(int levels) {
        int n = 3 * levels;
        double[][] a = new double[n][n];
        double[][] rps = {{0, -1, 1}, {1, 0, -1}, {-1, 1, 0}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double skill = (i % levels - j % levels) / (double) Math.max(1, levels - 1);
                a[i][j] = rps[i / levels][j / levels] + 0.4 * skill;
            }
        }
        return of("RPS with skill (" + levels + " levels)", a);
    }

    /** @return the payoff matrix to the row player (a copy) */
    public double[][] payoff() {
        double[][] out = new double[rows][];
        for (int i = 0; i < rows; i++) {
            out[i] = payoff[i].clone();
        }
        return out;
    }

    /** @return the number of rows */
    public int rows() {
        return rows;
    }

    /** @return the number of columns */
    public int columns() {
        return columns;
    }

    /**
     * Converts a {@link GameTree} policy table of this game into the two mixed strategies.
     *
     * @param tree the compiled game
     * @param policy the table
     * @return {@code {row strategy, column strategy}}
     */
    public double[][] strategies(GameTree tree, double[][] policy) {
        double[] row = new double[rows];
        double[] column = new double[columns];
        for (int s = 0; s < tree.infoSets(); s++) {
            if (tree.infoSetPlayer(s) == 0) {
                System.arraycopy(policy[s], 0, row, 0, rows);
            } else {
                System.arraycopy(policy[s], 0, column, 0, columns);
            }
        }
        return new double[][] {row, column};
    }

    @Override
    public State root() {
        return new Cell(-1, -1);
    }

    @Override
    public int maxActions() {
        return Math.max(rows, columns);
    }

    @Override
    public int featureSize() {
        return 2;
    }

    @Override
    public String toString() {
        return name;
    }

    private final class Cell implements State {
        private final int row;
        private final int column;

        Cell(int row, int column) {
            this.row = row;
            this.column = column;
        }

        @Override
        public boolean terminal() {
            return column >= 0;
        }

        @Override
        public int player() {
            return row < 0 ? 0 : 1;
        }

        @Override
        public int actionCount() {
            return maxActions();
        }

        @Override
        public boolean legal(int action) {
            return action >= 0 && action < (row < 0 ? rows : columns);
        }

        @Override
        public double chanceProbability(int action) {
            return 0;
        }

        @Override
        public State play(int action) {
            return row < 0 ? new Cell(action, -1) : new Cell(row, action);
        }

        @Override
        public double payoff() {
            return payoff[row][column];
        }

        @Override
        public String infoSet() {
            return row < 0 ? "row" : "column";
        }

        @Override
        public float[] features() {
            return row < 0 ? new float[] {1, 0} : new float[] {0, 1};
        }
    }
}
