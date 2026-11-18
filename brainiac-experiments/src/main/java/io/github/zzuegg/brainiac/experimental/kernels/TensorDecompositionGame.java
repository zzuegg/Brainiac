package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.rl.Mcts;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An AlphaTensor-style game (Fawzi et al., Nature 2022) for low-rank decompositions of the
 * matrix-multiplication tensor {@code T} of {@code [n, m] x [m, p]}. Writing {@code T} as a sum
 * of {@code R} rank-one terms {@code u (x) v (x) w} gives an algorithm with {@code R} scalar
 * multiplications.
 *
 * <p>An action picks a pair {@code (u, v)} with at most two non-zeros in {@code {-1, 0, 1}}
 * each ({@code u} sign-normalised), which is a rank-one matrix {@code u v^T} over
 * (A entries) x (B entries). The {@code w} of all terms follows by least squares, since every
 * C-slice {@code T_c} must lie in the span of the chosen {@code u v^T}. The state therefore
 * keeps an orthonormal basis of the chosen matrices and the slices' residuals. The episode
 * succeeds when the residual vanishes and the rounded {@code w} reproduces {@code T} exactly.
 * This is a simplification of AlphaTensor, which samples whole {@code (u, v, w)} triples from
 * a network. Each step costs -1, and failing at the rank cap costs extra. Actions are only
 * taken in increasing index order, which removes permutations. The prior is a softmax over
 * how much each action would shrink the residual, restricted to the best few.
 *
 * <p>Next to the {@link Mcts} environment, the class has the comparison searches: random
 * sampling of pairs ({@link #randomSearch}), alternating least squares that is pulled towards
 * integers and then rounded ({@link #alsRounding}), and {@link CmaEs} over the factor entries
 * ({@link #cmaEs}). It also has exact and random-matrix verification of any decomposition.
 */
public final class TensorDecompositionGame implements Mcts.Model<TensorDecompositionGame.State> {

    private static final int TOP = 64;

    private final int n;
    private final int m;
    private final int p;
    private final int sizeA;
    private final int sizeB;
    private final int sizeC;
    private final int maxRank;
    private final double[][] slices;
    private final List<int[]> uList = new ArrayList<>();
    private final List<int[]> vList = new ArrayList<>();
    private final int[][] actionPairs;
    private final double[][] actionVectors;

    /**
     * A decomposition {@code T = sum_r u_r (x) v_r (x) w_r}.
     *
     * @param u {@code [R][n m]}
     * @param v {@code [R][m p]}
     * @param w {@code [R][n p]}
     */
    public record Decomposition(int[][] u, int[][] v, int[][] w) {
        /** @return the rank (multiplications) */
        public int rank() {
            return u.length;
        }
    }

    /**
     * The outcome of a search.
     *
     * @param found the decomposition at the target rank, or null
     * @param evaluations episodes, restarts or candidate evaluations used
     */
    public record Search(Decomposition found, int evaluations) {
    }

    /** A game state: the chosen actions, an orthonormal basis of them, the slice residuals. */
    public static final class State {
        final int[] actions;
        final double[][] basis;
        final double[][] residual;
        final double energy;

        State(int[] actions, double[][] basis, double[][] residual) {
            this.actions = actions;
            this.basis = basis;
            this.residual = residual;
            double e = 0;
            for (double[] r : residual) {
                for (double x : r) {
                    e += x * x;
                }
            }
            this.energy = e;
        }

        /** @return the chosen actions */
        public int[] actions() {
            return actions.clone();
        }

        /** @return the squared norm of what the chosen matrices cannot explain */
        public double energy() {
            return energy;
        }
    }

    /**
     * @param n rows of A
     * @param m columns of A, rows of B
     * @param p columns of B
     * @param maxRank the rank cap of an episode (the target rank)
     */
    public TensorDecompositionGame(int n, int m, int p, int maxRank) {
        this.n = n;
        this.m = m;
        this.p = p;
        this.sizeA = n * m;
        this.sizeB = m * p;
        this.sizeC = n * p;
        this.maxRank = maxRank;
        double[][] t = new double[sizeC][sizeA * sizeB];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < m; k++) {
                for (int j = 0; j < p; j++) {
                    t[i * p + j][(i * m + k) * sizeB + k * p + j] = 1;
                }
            }
        }
        this.slices = t;
        enumerate(sizeA, true, uList);
        enumerate(sizeB, false, vList);
        actionPairs = new int[uList.size() * vList.size()][];
        actionVectors = new double[actionPairs.length][];
        int a = 0;
        for (int[] u : uList) {
            for (int[] v : vList) {
                actionPairs[a] = new int[] {uList.indexOf(u), vList.indexOf(v)};
                double[] x = new double[sizeA * sizeB];
                for (int i = 0; i < sizeA; i++) {
                    for (int j = 0; j < sizeB; j++) {
                        x[i * sizeB + j] = u[i] * v[j];
                    }
                }
                actionVectors[a++] = x;
            }
        }
    }

    /** Vectors with one or two non-zeros in {-1, 1}; with {@code canonical} the first is +1. */
    private static void enumerate(int d, boolean canonical, List<int[]> out) {
        for (int i = 0; i < d; i++) {
            for (int s = 1; s >= (canonical ? 1 : -1); s -= 2) {
                int[] v = new int[d];
                v[i] = s;
                out.add(v);
                for (int j = i + 1; j < d; j++) {
                    for (int t = 1; t >= -1; t -= 2) {
                        int[] w = v.clone();
                        w[j] = t;
                        out.add(w);
                    }
                }
            }
        }
    }

    /** @return the starting state */
    public State initial() {
        double[][] r = new double[sizeC][];
        for (int c = 0; c < sizeC; c++) {
            r[c] = slices[c].clone();
        }
        return new State(new int[0], new double[0][], r);
    }

    @Override
    public int actions() {
        return actionPairs.length;
    }

    /** @return the naive rank {@code n m p} */
    public int naiveRank() {
        return n * m * p;
    }

    private double[] orthogonalize(double[] x, double[][] basis) {
        double[] y = x.clone();
        for (double[] q : basis) {
            double d = dot(y, q);
            for (int i = 0; i < y.length; i++) {
                y[i] -= d * q[i];
            }
        }
        double norm = Math.sqrt(dot(y, y));
        if (norm < 1e-9) {
            return null;
        }
        for (int i = 0; i < y.length; i++) {
            y[i] /= norm;
        }
        return y;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * @param s a state
     * @param action an action
     * @return how much the residual energy would shrink
     */
    public double reduction(State s, int action) {
        double[] q = orthogonalize(actionVectors[action], s.basis);
        if (q == null) {
            return 0;
        }
        double red = 0;
        for (double[] r : s.residual) {
            double d = dot(r, q);
            red += d * d;
        }
        return red;
    }

    private boolean solved(State s) {
        return s.energy < 1e-9 && decomposition(s.actions) != null;
    }

    @Override
    public Mcts.Evaluation evaluate(State s) {
        if (solved(s) || s.actions.length >= maxRank) {
            return new Mcts.Evaluation(new float[actions()], 0, true);
        }
        int last = s.actions.length == 0 ? -1 : s.actions[s.actions.length - 1];
        double[] red = new double[actions()];
        Arrays.fill(red, -1);
        for (int a = last + 1; a < actions(); a++) {
            red[a] = reduction(s, a);
        }
        double[] sorted = red.clone();
        Arrays.sort(sorted);
        double cut = sorted[Math.max(0, sorted.length - TOP)];
        float[] prior = new float[actions()];
        for (int a = last + 1; a < actions(); a++) {
            if (red[a] >= cut && red[a] > 1e-9) {
                prior[a] = (float) Math.exp(red[a] / 0.25);
            }
        }
        return new Mcts.Evaluation(prior, (float) -s.energy, false);
    }

    @Override
    public Mcts.Step<State> step(State s, int action) {
        double[] q = orthogonalize(actionVectors[action], s.basis);
        int[] acts = Arrays.copyOf(s.actions, s.actions.length + 1);
        acts[s.actions.length] = action;
        double[][] basis = s.basis;
        double[][] residual = s.residual;
        if (q != null) {
            basis = Arrays.copyOf(s.basis, s.basis.length + 1);
            basis[s.basis.length] = q;
            residual = new double[sizeC][];
            for (int c = 0; c < sizeC; c++) {
                double d = dot(s.residual[c], q);
                double[] r = s.residual[c].clone();
                for (int i = 0; i < r.length; i++) {
                    r[i] -= d * q[i];
                }
                residual[c] = r;
            }
        }
        State next = new State(acts, basis, residual);
        float reward = -1;
        if (acts.length >= maxRank && !solved(next)) {
            reward -= (float) (2 + next.energy);
        }
        return new Mcts.Step<>(next, reward);
    }

    /**
     * Solves {@code w} by least squares for chosen actions, rounds it and verifies.
     *
     * @param actions the chosen actions
     * @return the exact decomposition, or null
     */
    public Decomposition decomposition(int[] actions) {
        int r = actions.length;
        double[][] g = new double[r][r];
        for (int a = 0; a < r; a++) {
            for (int b = 0; b < r; b++) {
                g[a][b] = dot(actionVectors[actions[a]], actionVectors[actions[b]]) + (a == b ? 1e-9 : 0);
            }
        }
        int[][] u = new int[r][];
        int[][] v = new int[r][];
        int[][] w = new int[r][sizeC];
        for (int a = 0; a < r; a++) {
            u[a] = uList.get(actionPairs[actions[a]][0]);
            v[a] = vList.get(actionPairs[actions[a]][1]);
        }
        for (int c = 0; c < sizeC; c++) {
            double[] rhs = new double[r];
            for (int a = 0; a < r; a++) {
                rhs[a] = dot(actionVectors[actions[a]], slices[c]);
            }
            double[] coef = solve(g, rhs);
            for (int a = 0; a < r; a++) {
                w[a][c] = (int) Math.max(-1, Math.min(1, Math.round(coef[a])));
            }
        }
        Decomposition d = new Decomposition(u, v, w);
        return exact(d) ? d : null;
    }

    private static double[] solve(double[][] a0, double[] b0) {
        int r = b0.length;
        double[][] a = new double[r][];
        for (int i = 0; i < r; i++) {
            a[i] = Arrays.copyOf(a0[i], r + 1);
            a[i][r] = b0[i];
        }
        for (int col = 0; col < r; col++) {
            int piv = col;
            for (int i = col + 1; i < r; i++) {
                if (Math.abs(a[i][col]) > Math.abs(a[piv][col])) {
                    piv = i;
                }
            }
            double[] t = a[col];
            a[col] = a[piv];
            a[piv] = t;
            if (Math.abs(a[col][col]) < 1e-12) {
                continue;
            }
            for (int i = 0; i < r; i++) {
                if (i != col) {
                    double f = a[i][col] / a[col][col];
                    for (int j = col; j <= r; j++) {
                        a[i][j] -= f * a[col][j];
                    }
                }
            }
        }
        double[] x = new double[r];
        for (int i = 0; i < r; i++) {
            x[i] = Math.abs(a[i][i]) < 1e-12 ? 0 : a[i][r] / a[i][i];
        }
        return x;
    }

    /**
     * @param d a decomposition
     * @return whether it reproduces the tensor exactly
     */
    public boolean exact(Decomposition d) {
        for (int c = 0; c < sizeC; c++) {
            for (int a = 0; a < sizeA; a++) {
                for (int b = 0; b < sizeB; b++) {
                    int s = 0;
                    for (int r = 0; r < d.rank(); r++) {
                        s += d.u()[r][a] * d.v()[r][b] * d.w()[r][c];
                    }
                    if (s != (int) slices[c][a * sizeB + b]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Runs the decomposition as an algorithm on random matrices.
     *
     * @param d the decomposition
     * @param trials random products
     * @param seed the seed
     * @return the largest absolute error against the naive product
     */
    public double randomCheck(Decomposition d, int trials, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double worst = 0;
        for (int t = 0; t < trials; t++) {
            double[] a = new double[sizeA];
            double[] b = new double[sizeB];
            for (int i = 0; i < sizeA; i++) {
                a[i] = rnd.nextGaussian();
            }
            for (int i = 0; i < sizeB; i++) {
                b[i] = rnd.nextGaussian();
            }
            double[] c = new double[sizeC];
            for (int r = 0; r < d.rank(); r++) {
                double left = 0;
                double right = 0;
                for (int i = 0; i < sizeA; i++) {
                    left += d.u()[r][i] * a[i];
                }
                for (int i = 0; i < sizeB; i++) {
                    right += d.v()[r][i] * b[i];
                }
                double prod = left * right;
                for (int i = 0; i < sizeC; i++) {
                    c[i] += d.w()[r][i] * prod;
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    double want = 0;
                    for (int k = 0; k < m; k++) {
                        want += a[i * m + k] * b[k * p + j];
                    }
                    worst = Math.max(worst, Math.abs(want - c[i * p + j]));
                }
            }
        }
        return worst;
    }

    /**
     * MCTS episodes: from the empty state, search and play the sampled action until the episode
     * ends; stops at the first exact decomposition.
     *
     * @param simulations simulations per move
     * @param episodes the episode budget
     * @param seed the seed
     * @return the result, {@code evaluations} counting episodes
     */
    public Search mcts(int simulations, int episodes, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Mcts<State> search = Mcts.of(this).simulations(simulations).rootNoise(0.3f, 0.25f);
        for (int e = 1; e <= episodes; e++) {
            State s = initial();
            while (!evaluate(s).terminal()) {
                Mcts.Result r = search.search(s, rnd);
                s = step(s, r.sample(1.0, rnd)).state();
            }
            if (solved(s)) {
                return new Search(decomposition(s.actions), e);
            }
        }
        return new Search(null, episodes);
    }

    /**
     * Samples {@code rank} distinct actions uniformly and checks them.
     *
     * @param rank the rank
     * @param evaluations the budget
     * @param seed the seed
     * @return the result
     */
    public Search randomSearch(int rank, int evaluations, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int e = 1; e <= evaluations; e++) {
            int[] acts = rnd.ints(0, actions()).distinct().limit(rank).sorted().toArray();
            State s = initial();
            for (int a : acts) {
                s = step(s, a).state();
            }
            if (s.energy < 1e-9) {
                Decomposition d = decomposition(acts);
                if (d != null) {
                    return new Search(d, e);
                }
            }
        }
        return new Search(null, evaluations);
    }

    /**
     * Alternating least squares from random starts, pulled towards integers by a growing ridge
     * to the rounded factors (after rescaling each term so its {@code u} and {@code v} peak at
     * 1), then rounded to {@code {-1, 0, 1}} and verified.
     *
     * @param rank the rank
     * @param restarts the budget of random starts
     * @param iterations ALS sweeps per start
     * @param seed the seed
     * @return the result, {@code evaluations} counting restarts
     */
    public Search alsRounding(int rank, int restarts, int iterations, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][][] t = tensor();
        for (int e = 1; e <= restarts; e++) {
            double[][] u = randomFactor(sizeA, rank, rnd);
            double[][] v = randomFactor(sizeB, rank, rnd);
            double[][] w = randomFactor(sizeC, rank, rnd);
            for (int it = 0; it < iterations; it++) {
                double lambda = it < iterations / 3 ? 0 : 0.5 * (it - iterations / 3.0) / (iterations * 2 / 3.0);
                if (lambda > 0) {
                    balance(u, v, w);
                }
                u = alsUpdate(t, 0, u, v, w, lambda);
                v = alsUpdate(t, 1, v, u, w, lambda);
                w = alsUpdate(t, 2, w, u, v, lambda);
            }
            balance(u, v, w);
            Decomposition d = new Decomposition(round(u), round(v), round(w));
            if (exact(d)) {
                return new Search(d, e);
            }
        }
        return new Search(null, restarts);
    }

    /**
     * CMA-ES over all factor entries, minimising the fit error plus a pull towards integers,
     * then rounding.
     *
     * @param rank the rank
     * @param restarts random starts
     * @param evaluationsPerStart the CMA-ES budget per start
     * @param seed the seed
     * @return the result, {@code evaluations} counting cost evaluations
     */
    public Search cmaEs(int rank, int restarts, int evaluationsPerStart, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][][] t = tensor();
        int dim = rank * (sizeA + sizeB + sizeC);
        int used = 0;
        for (int e = 0; e < restarts; e++) {
            double[] start = new double[dim];
            for (int i = 0; i < dim; i++) {
                start[i] = rnd.nextGaussian() * 0.5;
            }
            double[] best = CmaEs.of(start, 0.5).seed(rnd.nextLong()).minimize(x -> {
                double[][][] f = unpack(x, rank);
                double err = fitError(t, f[0], f[1], f[2]);
                double pull = 0;
                for (double xi : x) {
                    double d = xi - Math.max(-1, Math.min(1, Math.round(xi)));
                    pull += d * d;
                }
                return err + 0.1 * pull;
            }, evaluationsPerStart);
            used += evaluationsPerStart;
            double[][][] f = unpack(best, rank);
            Decomposition d = new Decomposition(round(f[0]), round(f[1]), round(f[2]));
            if (exact(d)) {
                return new Search(d, used);
            }
        }
        return new Search(null, used);
    }

    private double[][][] unpack(double[] x, int rank) {
        double[][] u = new double[sizeA][rank];
        double[][] v = new double[sizeB][rank];
        double[][] w = new double[sizeC][rank];
        int o = 0;
        for (double[][] f : List.of(u, v, w)) {
            for (double[] row : f) {
                for (int r = 0; r < rank; r++) {
                    row[r] = x[o++];
                }
            }
        }
        return new double[][][] {u, v, w};
    }

    private static double fitError(double[][][] t, double[][] u, double[][] v, double[][] w) {
        double err = 0;
        int rank = u[0].length;
        for (int a = 0; a < t.length; a++) {
            for (int b = 0; b < t[0].length; b++) {
                for (int c = 0; c < t[0][0].length; c++) {
                    double s = 0;
                    for (int r = 0; r < rank; r++) {
                        s += u[a][r] * v[b][r] * w[c][r];
                    }
                    err += (t[a][b][c] - s) * (t[a][b][c] - s);
                }
            }
        }
        return err;
    }

    private double[][][] tensor() {
        double[][][] t = new double[sizeA][sizeB][sizeC];
        for (int c = 0; c < sizeC; c++) {
            for (int a = 0; a < sizeA; a++) {
                for (int b = 0; b < sizeB; b++) {
                    t[a][b][c] = slices[c][a * sizeB + b];
                }
            }
        }
        return t;
    }

    private static double[][] randomFactor(int d, int rank, SplittableRandom rnd) {
        double[][] f = new double[d][rank];
        for (double[] row : f) {
            for (int r = 0; r < rank; r++) {
                row[r] = rnd.nextGaussian() * 0.5;
            }
        }
        return f;
    }

    /** Rescales every term so {@code max |u_r| = max |v_r| = 1}, the scale going to {@code w}. */
    private static void balance(double[][] u, double[][] v, double[][] w) {
        for (int r = 0; r < u[0].length; r++) {
            double mu = 0;
            double mv = 0;
            for (double[] row : u) {
                mu = Math.max(mu, Math.abs(row[r]));
            }
            for (double[] row : v) {
                mv = Math.max(mv, Math.abs(row[r]));
            }
            if (mu < 1e-12 || mv < 1e-12) {
                continue;
            }
            for (double[] row : u) {
                row[r] /= mu;
            }
            for (double[] row : v) {
                row[r] /= mv;
            }
            for (double[] row : w) {
                row[r] *= mu * mv;
            }
        }
    }

    /** One ALS update of the factor along {@code mode}, with a ridge towards its rounding. */
    private static double[][] alsUpdate(double[][][] t, int mode, double[][] self, double[][] f1, double[][] f2,
            double lambda) {
        int rank = self[0].length;
        double[][] g = new double[rank][rank];
        for (int a = 0; a < rank; a++) {
            for (int b = 0; b < rank; b++) {
                double s1 = 0;
                double s2 = 0;
                for (double[] row : f1) {
                    s1 += row[a] * row[b];
                }
                for (double[] row : f2) {
                    s2 += row[a] * row[b];
                }
                g[a][b] = s1 * s2 + (a == b ? lambda + 1e-6 : 0);
            }
        }
        double[][] out = new double[self.length][];
        for (int i = 0; i < self.length; i++) {
            double[] rhs = new double[rank];
            for (int j = 0; j < f1.length; j++) {
                for (int k = 0; k < f2.length; k++) {
                    double x = mode == 0 ? t[i][j][k] : mode == 1 ? t[j][i][k] : t[j][k][i];
                    if (x == 0) {
                        continue;
                    }
                    for (int r = 0; r < rank; r++) {
                        rhs[r] += x * f1[j][r] * f2[k][r];
                    }
                }
            }
            for (int r = 0; r < rank; r++) {
                rhs[r] += lambda * Math.max(-1, Math.min(1, Math.round(self[i][r])));
            }
            out[i] = solve(g, rhs);
        }
        return out;
    }

    private static int[][] round(double[][] f) {
        int rank = f[0].length;
        int[][] out = new int[rank][f.length];
        for (int r = 0; r < rank; r++) {
            for (int i = 0; i < f.length; i++) {
                out[r][i] = (int) Math.max(-1, Math.min(1, Math.round(f[i][r])));
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "TensorDecompositionGame(" + n + "x" + m + "x" + p + ", rank " + maxRank + ", " + actions()
                + " actions)";
    }
}
