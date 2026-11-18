package io.github.zzuegg.brainiac.experimental.games;

import io.github.zzuegg.brainiac.experimental.ops.TwoPhaseSimplex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Commitment and correlation in general-sum two-player matrix games:
 *
 * <ul>
 *   <li>{@link #stackelberg}: the optimal mixed strategy to commit to, by one linear program per
 *       follower response (Conitzer &amp; Sandholm, EC 2006), with ties broken for the leader
 *       (strong Stackelberg equilibrium);</li>
 *   <li>{@link #securityCoverage}: Stackelberg security games (the ERASER/DOBSS line, Paruchuri et
 *       al., AAMAS 2008): coverage probabilities over targets by one LP per attacked target;</li>
 *   <li>{@link #maxWelfareCe}: the welfare-maximising correlated equilibrium, an LP over joint
 *       distributions under the incentive constraints;</li>
 *   <li>{@link #nashEquilibria}: all Nash equilibria of small non-degenerate games by support
 *       enumeration, for the comparisons;</li>
 *   <li>{@link #dynamics}: no-regret dynamics whose empirical joint play approaches the coarse
 *       correlated (Hedge, optimistic Hedge, regret matching) or correlated (internal regret
 *       matching, Hart &amp; Mas-Colell, Econometrica 2000) equilibria, and fictitious play;
 *       optimistic Hedge converges faster (Syrgkanis et al., NeurIPS 2015,
 *       <a href="https://arxiv.org/abs/1507.00407">arXiv:1507.00407</a>).</li>
 * </ul>
 *
 * <p>The LPs need equalities and right-hand sides of any sign, so they use
 * {@link TwoPhaseSimplex} rather than the wave 6 {@code LinearProgram}.
 */
public final class CommitmentAndCorrelation {

    private CommitmentAndCorrelation() {
    }

    /**
     * A general-sum bimatrix game.
     *
     * @param name a label
     * @param row the row player's payoffs
     * @param column the column player's payoffs
     */
    public record Bimatrix(String name, double[][] row, double[][] column) {
        /** @return row actions */
        public int rows() {
            return row.length;
        }

        /** @return column actions */
        public int columns() {
            return row[0].length;
        }
    }

    /** @return chicken (swerve, straight) */
    public static Bimatrix chicken() {
        return new Bimatrix("chicken", new double[][] {{6, 2}, {7, 0}}, new double[][] {{6, 7}, {2, 0}});
    }

    /** @return the battle of the sexes */
    public static Bimatrix battleOfTheSexes() {
        return new Bimatrix("battle of the sexes", new double[][] {{3, 0}, {0, 2}}, new double[][] {{2, 0}, {0, 3}});
    }

    /** @return Shapley's 3 x 3 game, on which fictitious play cycles */
    public static Bimatrix shapley() {
        return new Bimatrix("Shapley", new double[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                new double[][] {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}});
    }

    /**
     * @param rows row actions
     * @param columns column actions
     * @param seed the seed
     * @return a game with independent uniform payoffs in {@code [0, 1]}
     */
    public static Bimatrix random(int rows, int columns, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = new double[rows][columns];
        double[][] b = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                a[i][j] = r.nextDouble();
                b[i][j] = r.nextDouble();
            }
        }
        return new Bimatrix("random " + rows + "x" + columns, a, b);
    }

    // ------------------------------------------------------------------ Stackelberg

    /**
     * An optimal commitment.
     *
     * @param leader the row player's mixed strategy
     * @param follower the column player's best response
     * @param leaderValue the leader's expected payoff
     * @param followerValue the follower's expected payoff
     */
    public record Commitment(double[] leader, int follower, double leaderValue, double followerValue) {
    }

    /**
     * The row player's optimal commitment (multiple LPs).
     *
     * @param g the game
     * @return the commitment
     */
    public static Commitment stackelberg(Bimatrix g) {
        int n = g.rows();
        int m = g.columns();
        Commitment best = null;
        for (int j = 0; j < m; j++) {
            double[] c = new double[n];
            for (int i = 0; i < n; i++) {
                c[i] = g.row()[i][j];
            }
            double[][] ub = new double[m - 1][n];
            int k = 0;
            for (int jj = 0; jj < m; jj++) {
                if (jj == j) {
                    continue;
                }
                for (int i = 0; i < n; i++) {
                    ub[k][i] = g.column()[i][jj] - g.column()[i][j];
                }
                k++;
            }
            double[][] eq = new double[1][n];
            Arrays.fill(eq[0], 1);
            TwoPhaseSimplex.Result r = TwoPhaseSimplex.maximize(c, ub, new double[m - 1], eq, new double[] {1});
            if (!r.optimal()) {
                continue;
            }
            if (best == null || r.objective() > best.leaderValue() + 1e-12) {
                double fv = 0;
                for (int i = 0; i < n; i++) {
                    fv += r.x()[i] * g.column()[i][j];
                }
                best = new Commitment(r.x(), j, r.objective(), fv);
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ security games

    /**
     * A security game: the defender covers targets with {@code resources} units of coverage, the
     * attacker observes the coverage and attacks one target.
     *
     * @param defenderCovered defender payoff when the attacked target is covered
     * @param defenderUncovered defender payoff when it is not
     * @param attackerCovered attacker payoff when covered
     * @param attackerUncovered attacker payoff when not
     * @param resources the coverage budget
     */
    public record SecurityGame(double[] defenderCovered, double[] defenderUncovered, double[] attackerCovered,
            double[] attackerUncovered, double resources) {
        /** @return the number of targets */
        public int targets() {
            return defenderCovered.length;
        }

        /**
         * @param coverage coverage per target
         * @return the defender's payoff when the attacker best responds (ties for the defender)
         */
        public double defenderValue(double[] coverage) {
            double bestAtt = Double.NEGATIVE_INFINITY;
            double def = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < targets(); t++) {
                double ua = coverage[t] * attackerCovered[t] + (1 - coverage[t]) * attackerUncovered[t];
                double ud = coverage[t] * defenderCovered[t] + (1 - coverage[t]) * defenderUncovered[t];
                if (ua > bestAtt + 1e-9 || (Math.abs(ua - bestAtt) <= 1e-9 && ud > def)) {
                    bestAtt = ua;
                    def = ud;
                }
            }
            return def;
        }
    }

    /**
     * @param targets targets
     * @param resources coverage budget
     * @param seed the seed
     * @return a random security game (defender gains when covering, loses otherwise)
     */
    public static SecurityGame securityGame(int targets, double resources, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] dc = new double[targets];
        double[] du = new double[targets];
        double[] ac = new double[targets];
        double[] au = new double[targets];
        for (int t = 0; t < targets; t++) {
            dc[t] = 10 * r.nextDouble();
            du[t] = -10 * r.nextDouble();
            ac[t] = -10 * r.nextDouble();
            au[t] = 10 * r.nextDouble();
        }
        return new SecurityGame(dc, du, ac, au, resources);
    }

    /**
     * The defender's optimal coverage by one LP per attacked target.
     *
     * @param g the game
     * @return coverage probabilities per target
     */
    public static double[] securityCoverage(SecurityGame g) {
        int n = g.targets();
        double[] best = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int star = 0; star < n; star++) {
            double[] c = new double[n];
            c[star] = g.defenderCovered()[star] - g.defenderUncovered()[star];
            List<double[]> rows = new ArrayList<>();
            List<Double> rhs = new ArrayList<>();
            double[] budget = new double[n];
            Arrays.fill(budget, 1);
            rows.add(budget);
            rhs.add(g.resources());
            for (int t = 0; t < n; t++) {
                double[] cap = new double[n];
                cap[t] = 1;
                rows.add(cap);
                rhs.add(1.0);
                if (t == star) {
                    continue;
                }
                double[] row = new double[n];
                row[t] = g.attackerCovered()[t] - g.attackerUncovered()[t];
                row[star] -= g.attackerCovered()[star] - g.attackerUncovered()[star];
                rows.add(row);
                rhs.add(g.attackerUncovered()[star] - g.attackerUncovered()[t]);
            }
            TwoPhaseSimplex.Result r = TwoPhaseSimplex.maximize(c, rows.toArray(new double[0][]),
                    rhs.stream().mapToDouble(Double::doubleValue).toArray());
            if (r.optimal()) {
                double value = r.objective() + g.defenderUncovered()[star];
                if (value > bestValue) {
                    bestValue = value;
                    best = r.x();
                }
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ correlated equilibria

    /**
     * The correlated equilibrium with the highest total payoff.
     *
     * @param g the game
     * @return the joint distribution {@code [rows][columns]}
     */
    public static double[][] maxWelfareCe(Bimatrix g) {
        int n = g.rows();
        int m = g.columns();
        int vars = n * m;
        double[] c = new double[vars];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                c[i * m + j] = g.row()[i][j] + g.column()[i][j];
            }
        }
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double[] row = new double[vars];
                    for (int j = 0; j < m; j++) {
                        row[i * m + j] = g.row()[k][j] - g.row()[i][j];
                    }
                    rows.add(row);
                }
            }
        }
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                if (k != j) {
                    double[] row = new double[vars];
                    for (int i = 0; i < n; i++) {
                        row[i * m + j] = g.column()[i][k] - g.column()[i][j];
                    }
                    rows.add(row);
                }
            }
        }
        double[][] eq = new double[1][vars];
        Arrays.fill(eq[0], 1);
        TwoPhaseSimplex.Result r = TwoPhaseSimplex.maximize(c, rows.toArray(new double[0][]),
                new double[rows.size()], eq, new double[] {1});
        double[][] p = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                p[i][j] = r.x()[i * m + j];
            }
        }
        return p;
    }

    /**
     * @param g the game
     * @param p a joint distribution
     * @return the total expected payoff
     */
    public static double welfare(Bimatrix g, double[][] p) {
        double w = 0;
        for (int i = 0; i < g.rows(); i++) {
            for (int j = 0; j < g.columns(); j++) {
                w += p[i][j] * (g.row()[i][j] + g.column()[i][j]);
            }
        }
        return w;
    }

    /**
     * The largest gain from a recommendation-dependent deviation (0 for a correlated equilibrium).
     *
     * @param g the game
     * @param p a joint distribution
     * @return the CE gap
     */
    public static double ceGap(Bimatrix g, double[][] p) {
        double gap = 0;
        for (int i = 0; i < g.rows(); i++) {
            for (int k = 0; k < g.rows(); k++) {
                double s = 0;
                for (int j = 0; j < g.columns(); j++) {
                    s += p[i][j] * (g.row()[k][j] - g.row()[i][j]);
                }
                gap = Math.max(gap, s);
            }
        }
        for (int j = 0; j < g.columns(); j++) {
            for (int k = 0; k < g.columns(); k++) {
                double s = 0;
                for (int i = 0; i < g.rows(); i++) {
                    s += p[i][j] * (g.column()[i][k] - g.column()[i][j]);
                }
                gap = Math.max(gap, s);
            }
        }
        return gap;
    }

    /**
     * The largest gain from a fixed deviation (0 for a coarse correlated equilibrium).
     *
     * @param g the game
     * @param p a joint distribution
     * @return the CCE gap
     */
    public static double cceGap(Bimatrix g, double[][] p) {
        int n = g.rows();
        int m = g.columns();
        double ur = 0;
        double uc = 0;
        double[] colMarg = new double[m];
        double[] rowMarg = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ur += p[i][j] * g.row()[i][j];
                uc += p[i][j] * g.column()[i][j];
                colMarg[j] += p[i][j];
                rowMarg[i] += p[i][j];
            }
        }
        double gap = 0;
        for (int k = 0; k < n; k++) {
            double s = 0;
            for (int j = 0; j < m; j++) {
                s += colMarg[j] * g.row()[k][j];
            }
            gap = Math.max(gap, s - ur);
        }
        for (int k = 0; k < m; k++) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += rowMarg[i] * g.column()[i][k];
            }
            gap = Math.max(gap, s - uc);
        }
        return gap;
    }

    // ------------------------------------------------------------------ Nash by support enumeration

    /**
     * All Nash equilibria of a small non-degenerate game, by enumerating equal-size supports.
     *
     * @param g the game (up to about 5 x 5)
     * @return pairs {row strategy, column strategy}
     */
    public static List<double[][]> nashEquilibria(Bimatrix g) {
        int n = g.rows();
        int m = g.columns();
        List<double[][]> out = new ArrayList<>();
        for (int rs = 1; rs < (1 << n); rs++) {
            for (int cs = 1; cs < (1 << m); cs++) {
                if (Integer.bitCount(rs) != Integer.bitCount(cs)) {
                    continue;
                }
                double[] y = indifference(g.row(), rs, cs, false);
                double[] x = indifference(g.column(), cs, rs, true);
                if (x == null || y == null) {
                    continue;
                }
                if (bestResponse(g.row(), x, y, false) && bestResponse(g.column(), y, x, true)) {
                    boolean dup = out.stream().anyMatch(e -> close(e[0], x) && close(e[1], y));
                    if (!dup) {
                        out.add(new double[][] {x, y});
                    }
                }
            }
        }
        return out;
    }

    private static boolean close(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 1e-7) {
                return false;
            }
        }
        return true;
    }

    /**
     * The opponent strategy (on support {@code oppSupport}) making the player indifferent over
     * {@code support}; {@code transpose} reads the payoffs as {@code [opponent][player]}.
     */
    private static double[] indifference(double[][] payoff, int support, int oppSupport, boolean transpose) {
        int[] s = bits(support);
        int[] o = bits(oppSupport);
        int k = o.length;
        int size = k + 1;
        double[][] a = new double[size][size + 1];
        for (int r = 0; r < s.length; r++) {
            for (int c = 0; c < k; c++) {
                a[r][c] = transpose ? payoff[o[c]][s[r]] : payoff[s[r]][o[c]];
            }
            a[r][k] = -1;
        }
        for (int c = 0; c < k; c++) {
            a[size - 1][c] = 1;
        }
        a[size - 1][size] = 1;
        double[] sol = solve(a);
        if (sol == null) {
            return null;
        }
        int len = transpose ? payoff.length : payoff[0].length;
        double[] strat = new double[len];
        for (int c = 0; c < k; c++) {
            if (sol[c] < -1e-9) {
                return null;
            }
            strat[o[c]] = Math.max(0, sol[c]);
        }
        return strat;
    }

    private static boolean bestResponse(double[][] payoff, double[] mine, double[] other, boolean transpose) {
        int len = mine.length;
        double[] u = new double[len];
        for (int a = 0; a < len; a++) {
            for (int b = 0; b < other.length; b++) {
                u[a] += other[b] * (transpose ? payoff[b][a] : payoff[a][b]);
            }
        }
        double max = Arrays.stream(u).max().orElse(0);
        for (int a = 0; a < len; a++) {
            if (mine[a] > 1e-9 && u[a] < max - 1e-7) {
                return false;
            }
        }
        return true;
    }

    private static int[] bits(int mask) {
        int[] out = new int[Integer.bitCount(mask)];
        int k = 0;
        for (int i = 0; i < 31; i++) {
            if ((mask >> i & 1) == 1) {
                out[k++] = i;
            }
        }
        return out;
    }

    /** Gaussian elimination with partial pivoting on an augmented square system. */
    private static double[] solve(double[][] a) {
        int n = a.length;
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[p][c])) {
                    p = r;
                }
            }
            if (Math.abs(a[p][c]) < 1e-12) {
                return null;
            }
            double[] t = a[p];
            a[p] = a[c];
            a[c] = t;
            for (int r = 0; r < n; r++) {
                if (r != c) {
                    double f = a[r][c] / a[c][c];
                    for (int k = c; k <= n; k++) {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = a[i][n] / a[i][i];
        }
        return x;
    }

    // ------------------------------------------------------------------ dynamics

    /** Learning dynamics. */
    public enum Dynamics {
        /** Hedge (multiplicative weights) on expected payoffs */
        HEDGE,
        /** optimistic Hedge, which counts the last payoff twice */
        OPTIMISTIC_HEDGE,
        /** unconditional regret matching (to the coarse correlated equilibria) */
        REGRET_MATCHING,
        /** internal (conditional) regret matching (to the correlated equilibria) */
        INTERNAL_REGRET_MATCHING,
        /** fictitious play (best response to the empirical frequencies) */
        FICTITIOUS_PLAY
    }

    /**
     * Runs a dynamics with full-information expected payoffs and reports the gap of the empirical
     * joint distribution.
     *
     * @param g the game
     * @param dynamics the dynamics
     * @param iterations rounds
     * @param eta the Hedge step size
     * @param checkpoints rounds at which to measure
     * @param correlated whether to measure the CE gap (else the CCE gap)
     * @return the gap at every checkpoint
     */
    public static double[] dynamics(Bimatrix g, Dynamics dynamics, int iterations, double eta, int[] checkpoints,
            boolean correlated) {
        Objects.requireNonNull(dynamics, "dynamics");
        int n = g.rows();
        int m = g.columns();
        double[] x = uniformish(n, 0);
        double[] y = uniformish(m, 1);
        double[] sx = new double[n];
        double[] sy = new double[m];
        double[] lastX = new double[n];
        double[] lastY = new double[m];
        double[][] rx = new double[n][n];
        double[][] ry = new double[m][m];
        double[] countX = new double[n];
        double[] countY = new double[m];
        double[][] joint = new double[n][m];
        double[] out = new double[checkpoints.length];
        int next = 0;
        for (int t = 1; t <= iterations; t++) {
            if (dynamics == Dynamics.FICTITIOUS_PLAY) {
                x = oneHot(n, argmax(t == 1 ? new double[n] : sx));
                y = oneHot(m, argmax(t == 1 ? new double[m] : sy));
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    joint[i][j] += x[i] * y[j];
                }
            }
            double[] ux = new double[n];
            double[] uy = new double[m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    ux[i] += g.row()[i][j] * y[j];
                    uy[j] += g.column()[i][j] * x[i];
                }
            }
            double vx = dot(x, ux);
            double vy = dot(y, uy);
            switch (dynamics) {
                case HEDGE, OPTIMISTIC_HEDGE, FICTITIOUS_PLAY -> {
                    add(sx, ux);
                    add(sy, uy);
                    lastX = ux;
                    lastY = uy;
                }
                case REGRET_MATCHING -> {
                    for (int i = 0; i < n; i++) {
                        sx[i] += ux[i] - vx;
                    }
                    for (int j = 0; j < m; j++) {
                        sy[j] += uy[j] - vy;
                    }
                }
                case INTERNAL_REGRET_MATCHING -> {
                    for (int i = 0; i < n; i++) {
                        for (int k = 0; k < n; k++) {
                            rx[i][k] += x[i] * (ux[k] - ux[i]);
                        }
                    }
                    for (int j = 0; j < m; j++) {
                        for (int k = 0; k < m; k++) {
                            ry[j][k] += y[j] * (uy[k] - uy[j]);
                        }
                    }
                }
            }
            if (dynamics == Dynamics.FICTITIOUS_PLAY) {
                countX[argmax(x)]++;
                countY[argmax(y)]++;
            }
            switch (dynamics) {
                case HEDGE -> {
                    x = softmax(sx, eta, null);
                    y = softmax(sy, eta, null);
                }
                case OPTIMISTIC_HEDGE -> {
                    x = softmax(sx, eta, lastX);
                    y = softmax(sy, eta, lastY);
                }
                case REGRET_MATCHING -> {
                    x = positivePart(sx, n);
                    y = positivePart(sy, m);
                }
                case INTERNAL_REGRET_MATCHING -> {
                    x = stationary(rx, t);
                    y = stationary(ry, t);
                }
                case FICTITIOUS_PLAY -> {
                    // fictitious play best-responds to the opponent's empirical frequencies
                    double[] fx = new double[n];
                    double[] fy = new double[m];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < m; j++) {
                            fx[i] += g.row()[i][j] * countY[j];
                            fy[j] += g.column()[i][j] * countX[i];
                        }
                    }
                    sx = fx;
                    sy = fy;
                }
            }
            if (next < checkpoints.length && t == checkpoints[next]) {
                double[][] p = new double[n][m];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        p[i][j] = joint[i][j] / t;
                    }
                }
                out[next++] = correlated ? ceGap(g, p) : cceGap(g, p);
            }
        }
        return out;
    }

    private static double[] uniformish(int n, int bias) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = (1 + (i == bias % n ? 0.5 : 0)) / (n + 0.5);
        }
        return x;
    }

    private static double[] oneHot(int n, int k) {
        double[] x = new double[n];
        x[k] = 1;
        return x;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b] + 1e-12) {
                b = i;
            }
        }
        return b;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static void add(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    private static double[] softmax(double[] s, double eta, double[] optimism) {
        double[] z = new double[s.length];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < s.length; i++) {
            z[i] = eta * (s[i] + (optimism == null ? 0 : optimism[i]));
            max = Math.max(max, z[i]);
        }
        double sum = 0;
        for (int i = 0; i < z.length; i++) {
            z[i] = Math.exp(z[i] - max);
            sum += z[i];
        }
        for (int i = 0; i < z.length; i++) {
            z[i] /= sum;
        }
        return z;
    }

    private static double[] positivePart(double[] r, int n) {
        double[] x = new double[n];
        double s = 0;
        for (int i = 0; i < n; i++) {
            x[i] = Math.max(0, r[i]);
            s += x[i];
        }
        if (s <= 0) {
            Arrays.fill(x, 1.0 / n);
            return x;
        }
        for (int i = 0; i < n; i++) {
            x[i] /= s;
        }
        return x;
    }

    /** The stationary distribution of the regret-matching switching chain. */
    private static double[] stationary(double[][] regret, int t) {
        int n = regret.length;
        double mu = 0;
        double[][] q = new double[n][n];
        for (int i = 0; i < n; i++) {
            double row = 0;
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    q[i][k] = Math.max(0, regret[i][k]) / t;
                    row += q[i][k];
                }
            }
            mu = Math.max(mu, row);
        }
        mu = Math.max(mu, 1e-12) * 1.5;
        for (int i = 0; i < n; i++) {
            double row = 0;
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    q[i][k] /= mu;
                    row += q[i][k];
                }
            }
            q[i][i] = 1 - row;
        }
        double[] p = new double[n];
        Arrays.fill(p, 1.0 / n);
        for (int it = 0; it < 200; it++) {
            double[] np = new double[n];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    np[k] += p[i] * q[i][k];
                }
            }
            p = np;
        }
        return p;
    }
}
