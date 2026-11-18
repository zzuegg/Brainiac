package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.logic.Csp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Generators of constraint and logic problems for the wave 13 reasoning blocks.
 *
 * <ul>
 *   <li>Binary CSPs ({@link Csp}): N-queens, graph colouring on random graphs or
 *       {@link SyntheticGraphs}, Model RB random CSPs (Xu &amp; Li, JAIR 2000) whose tightness
 *       sweeps across the phase transition {@code p_cr = 1 - exp(-alpha / r)}, and Sudoku
 *       (4x4 and 9x9) puzzles cut from a random valid grid.</li>
 *   <li>CNF families beyond {@link RandomSatFormulas}' random 3-SAT: uniform random k-SAT at a
 *       given clause ratio, pigeonhole formulas (unsatisfiable, exponential for resolution),
 *       graph-colouring CNFs, and planted-solution colouring CNFs (always satisfiable).</li>
 *   <li>Random Bayesian networks with a share of deterministic CPT rows, for weighted model
 *       counting ({@link BayesNet}).</li>
 * </ul>
 * Planning-as-SAT encodings come from {@code SatPlanner.encode} and first-order problems from
 * {@code ResolutionProver}'s problem generators, which build on the same instance families.
 */
public final class ConstraintInstances {

    private ConstraintInstances() {
    }

    // ---- CSPs --------------------------------------------------------------------------------

    /**
     * N-queens: one variable per row, its value the column.
     *
     * @param n the board size
     * @return the CSP
     */
    public static Csp nQueens(int n) {
        int[] dom = new int[n];
        java.util.Arrays.fill(dom, n);
        List<int[]> pairs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                pairs.add(new int[] {i, j});
            }
        }
        return Csp.of("queens-" + n, dom, pairs, (x, y, a, b) -> a != b && Math.abs(a - b) != Math.abs(x - y));
    }

    /**
     * A random graph {@code G(n, m)} with {@code m = degree * n / 2} edges.
     *
     * @param n vertices
     * @param degree mean degree
     * @param seed the seed
     * @return the edges
     */
    public static List<int[]> randomGraph(int n, double degree, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int m = (int) Math.round(degree * n / 2);
        Set<Long> seen = new HashSet<>();
        List<int[]> edges = new ArrayList<>();
        while (edges.size() < m) {
            int a = r.nextInt(n);
            int b = r.nextInt(n);
            if (a == b) {
                continue;
            }
            long key = (long) Math.min(a, b) * n + Math.max(a, b);
            if (seen.add(key)) {
                edges.add(new int[] {Math.min(a, b), Math.max(a, b)});
            }
        }
        return edges;
    }

    /**
     * A planted k-colourable graph: vertices get hidden colours and edges only join different
     * colours (always satisfiable).
     *
     * @param n vertices
     * @param k colours
     * @param degree mean degree
     * @param seed the seed
     * @return the edges
     */
    public static List<int[]> plantedGraph(int n, int k, double degree, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] colour = new int[n];
        for (int i = 0; i < n; i++) {
            colour[i] = r.nextInt(k);
        }
        int m = (int) Math.round(degree * n / 2);
        Set<Long> seen = new HashSet<>();
        List<int[]> edges = new ArrayList<>();
        int guard = 0;
        while (edges.size() < m && guard++ < 100 * m) {
            int a = r.nextInt(n);
            int b = r.nextInt(n);
            if (a == b || colour[a] == colour[b]) {
                continue;
            }
            long key = (long) Math.min(a, b) * n + Math.max(a, b);
            if (seen.add(key)) {
                edges.add(new int[] {Math.min(a, b), Math.max(a, b)});
            }
        }
        return edges;
    }

    /**
     * Graph colouring.
     *
     * @param n vertices
     * @param edges the edges
     * @param k colours
     * @return the CSP
     */
    public static Csp colouring(int n, List<int[]> edges, int k) {
        int[] dom = new int[n];
        java.util.Arrays.fill(dom, k);
        return Csp.of("colouring-" + n + "-" + k, dom, edges, (x, y, a, b) -> a != b);
    }

    /**
     * Graph colouring of a {@link SyntheticGraphs} graph.
     *
     * @param g the graph
     * @param k colours
     * @return the CSP
     */
    public static Csp colouring(SyntheticGraphs.Graph g, int k) {
        List<int[]> edges = new ArrayList<>();
        int n = g.nodes();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (g.adjacency()[i * n + j] != 0) {
                    edges.add(new int[] {i, j});
                }
            }
        }
        return colouring(n, edges, k);
    }

    /**
     * A Model RB random CSP: domain {@code d = round(n^alpha)}, {@code m = round(r n ln n)}
     * constraints on random distinct pairs, each forbidding {@code round(p d^2)} random tuples.
     *
     * @param n variables
     * @param alpha the domain exponent
     * @param r the constraint density
     * @param p the tightness
     * @param seed the seed
     * @return the CSP
     */
    public static Csp modelRb(int n, double alpha, double r, double p, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int d = (int) Math.round(Math.pow(n, alpha));
        int m = (int) Math.round(r * n * Math.log(n));
        int[] dom = new int[n];
        java.util.Arrays.fill(dom, d);
        List<Csp.Constraint> cs = new ArrayList<>();
        int forbid = (int) Math.round(p * d * d);
        for (int c = 0; c < m; c++) {
            int x = rnd.nextInt(n);
            int y;
            do {
                y = rnd.nextInt(n);
            } while (y == x);
            boolean[][] allowed = new boolean[d][d];
            for (boolean[] row : allowed) {
                java.util.Arrays.fill(row, true);
            }
            int done = 0;
            while (done < forbid) {
                int a = rnd.nextInt(d);
                int b = rnd.nextInt(d);
                if (allowed[a][b]) {
                    allowed[a][b] = false;
                    done++;
                }
            }
            cs.add(Csp.Constraint.table(x, y, allowed));
        }
        return new Csp("rb-" + n + "-" + p, dom, cs);
    }

    /**
     * The Model RB phase-transition tightness.
     *
     * @param alpha the domain exponent
     * @param r the constraint density
     * @return {@code 1 - exp(-alpha / r)}
     */
    public static double modelRbCritical(double alpha, double r) {
        return 1 - Math.exp(-alpha / r);
    }

    /**
     * A Sudoku puzzle.
     *
     * @param givens cell values (row-major), -1 for blanks
     * @param solution the grid it was cut from
     * @param box the box side (2 or 3)
     */
    public record Sudoku(int[] givens, int[] solution, int box) {

        /** @return the CSP: one variable per cell, givens as singleton domains via unary pruning */
        public Csp csp() {
            int n = box * box;
            int cells = n * n;
            int[] dom = new int[cells];
            java.util.Arrays.fill(dom, n);
            List<int[]> pairs = new ArrayList<>();
            for (int a = 0; a < cells; a++) {
                for (int b = a + 1; b < cells; b++) {
                    int ra = a / n;
                    int ca = a % n;
                    int rb = b / n;
                    int cb = b % n;
                    if (ra == rb || ca == cb || (ra / box == rb / box && ca / box == cb / box)) {
                        pairs.add(new int[] {a, b});
                    }
                }
            }
            // givens are encoded as constraints against every peer: a given cell keeps only its value
            int[] g = givens.clone();
            return Csp.of("sudoku-" + n, dom, pairs, (x, y, va, vb) -> va != vb && (g[x] < 0 || g[x] == va)
                && (g[y] < 0 || g[y] == vb));
        }
    }

    /**
     * A Sudoku puzzle: a random valid grid (a shifted base pattern with random digit, row-band
     * and column-stack permutations) with random cells blanked.
     *
     * @param box the box side (2 for 4x4, 3 for 9x9)
     * @param blanks cells to blank
     * @param seed the seed
     * @return the puzzle
     */
    public static Sudoku sudoku(int box, int blanks, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = box * box;
        int[] digit = permutation(n, r);
        int[] rows = new int[n];
        int[] cols = new int[n];
        int[] bands = permutation(box, r);
        int[] stacks = permutation(box, r);
        for (int b = 0; b < box; b++) {
            int[] within = permutation(box, r);
            int[] within2 = permutation(box, r);
            for (int i = 0; i < box; i++) {
                rows[b * box + i] = bands[b] * box + within[i];
                cols[b * box + i] = stacks[b] * box + within2[i];
            }
        }
        int[] sol = new int[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ri = rows[i];
                int cj = cols[j];
                sol[i * n + j] = digit[(box * (ri % box) + ri / box + cj) % n];
            }
        }
        int[] givens = sol.clone();
        int[] order = permutation(n * n, r);
        for (int i = 0; i < Math.min(blanks, n * n); i++) {
            givens[order[i]] = -1;
        }
        return new Sudoku(givens, sol, box);
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    // ---- CNF ---------------------------------------------------------------------------------

    /**
     * A CNF formula.
     *
     * @param variables the number of variables
     * @param clauses DIMACS clauses
     * @param name a label
     */
    public record Cnf(int variables, List<int[]> clauses, String name) {

        /** @return as a {@link RandomSatFormulas.Formula} (satisfiability flag unknown: false) */
        public RandomSatFormulas.Formula formula() {
            return new RandomSatFormulas.Formula(variables, clauses, false);
        }
    }

    /**
     * Uniform random k-SAT.
     *
     * @param n variables
     * @param k literals per clause
     * @param ratio clauses per variable (4.26 is the 3-SAT threshold)
     * @param seed the seed
     * @return the formula
     */
    public static Cnf randomKSat(int n, int k, double ratio, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int m = (int) Math.round(ratio * n);
        List<int[]> cs = new ArrayList<>();
        for (int c = 0; c < m; c++) {
            int[] cl = new int[k];
            for (int i = 0; i < k; i++) {
                int v;
                boolean dup;
                do {
                    v = 1 + r.nextInt(n);
                    dup = false;
                    for (int j = 0; j < i; j++) {
                        dup |= Math.abs(cl[j]) == v;
                    }
                } while (dup);
                cl[i] = r.nextBoolean() ? v : -v;
            }
            cs.add(cl);
        }
        return new Cnf(n, cs, k + "sat-" + n + "-" + seed);
    }

    /**
     * The pigeonhole formula: {@code holes + 1} pigeons into {@code holes} holes (unsatisfiable).
     *
     * @param holes the number of holes
     * @return the formula
     */
    public static Cnf pigeonhole(int holes) {
        int pigeons = holes + 1;
        List<int[]> cs = new ArrayList<>();
        for (int p = 0; p < pigeons; p++) {
            int[] c = new int[holes];
            for (int h = 0; h < holes; h++) {
                c[h] = p * holes + h + 1;
            }
            cs.add(c);
        }
        for (int h = 0; h < holes; h++) {
            for (int p = 0; p < pigeons; p++) {
                for (int q = p + 1; q < pigeons; q++) {
                    cs.add(new int[] {-(p * holes + h + 1), -(q * holes + h + 1)});
                }
            }
        }
        return new Cnf(pigeons * holes, cs, "php-" + holes);
    }

    /**
     * The direct CNF encoding of graph colouring.
     *
     * @param n vertices
     * @param edges edges
     * @param k colours
     * @return the formula
     */
    public static Cnf colouringCnf(int n, List<int[]> edges, int k) {
        Csp c = colouring(n, edges, k);
        return new Cnf(c.cnfVariables(), c.toCnf(), "col-" + n + "-" + k);
    }

    // ---- Bayesian networks --------------------------------------------------------------------

    /**
     * A discrete Bayesian network.
     *
     * @param cardinality values per variable
     * @param parents per variable its parents (topologically ordered: parents have lower index)
     * @param cpt per variable, rows by parent configuration (row-major, first parent slowest)
     *     times child value
     */
    public record BayesNet(int[] cardinality, int[][] parents, double[][] cpt) {
    }

    /**
     * A random Bayesian network on binary or ternary variables with up to {@code maxParents}
     * parents among the previous {@code window} variables; a share of CPT rows is deterministic
     * (one value has probability 1).
     *
     * @param n variables
     * @param maxParents the most parents
     * @param window how far back parents may be
     * @param deterministic the share of deterministic rows
     * @param seed the seed
     * @return the network
     */
    public static BayesNet bayesNet(int n, int maxParents, int window, double deterministic, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] card = new int[n];
        int[][] parents = new int[n][];
        double[][] cpt = new double[n][];
        for (int v = 0; v < n; v++) {
            card[v] = r.nextDouble() < 0.7 ? 2 : 3;
            int k = Math.min(v, r.nextInt(maxParents + 1));
            Set<Integer> ps = new java.util.TreeSet<>();
            while (ps.size() < k) {
                ps.add(Math.max(0, v - 1 - r.nextInt(Math.min(v, window))));
            }
            parents[v] = ps.stream().mapToInt(Integer::intValue).toArray();
            int rows = 1;
            for (int p : parents[v]) {
                rows *= card[p];
            }
            cpt[v] = new double[rows * card[v]];
            for (int row = 0; row < rows; row++) {
                if (r.nextDouble() < deterministic) {
                    cpt[v][row * card[v] + r.nextInt(card[v])] = 1;
                } else {
                    double s = 0;
                    for (int x = 0; x < card[v]; x++) {
                        double w = 0.1 + r.nextDouble();
                        cpt[v][row * card[v] + x] = w;
                        s += w;
                    }
                    for (int x = 0; x < card[v]; x++) {
                        cpt[v][row * card[v] + x] /= s;
                    }
                }
            }
        }
        return new BayesNet(card, parents, cpt);
    }

    /**
     * The network as a {@link io.github.zzuegg.brainiac.experimental.ml.FactorGraph}.
     *
     * @param net the network
     * @return the factor graph
     */
    public static io.github.zzuegg.brainiac.experimental.ml.FactorGraph factorGraph(BayesNet net) {
        io.github.zzuegg.brainiac.experimental.ml.FactorGraph g = new io.github.zzuegg.brainiac.experimental.ml.FactorGraph();
        for (int c : net.cardinality()) {
            g.variable(c);
        }
        for (int v = 0; v < net.cardinality().length; v++) {
            g.conditional(v, net.parents()[v], net.cpt()[v]);
        }
        return g;
    }
}
