package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A discrete factor graph (a Bayesian network or Markov random field written as a product of
 * non-negative factors) with three ways to compute single-variable marginals:
 * <ul>
 *   <li>{@link #exactMarginals}: variable elimination with a greedy min-fill order, exact and
 *       exponential only in the induced width;</li>
 *   <li>{@link #loopyBeliefPropagation}: sum-product message passing (Pearl 1988; on graphs with
 *       cycles, Murphy, Weiss and Jordan 1999, <a href="https://arxiv.org/abs/1301.6725">arXiv:1301.6725</a>),
 *       exact on trees and usually a good fixed point on loopy graphs;</li>
 *   <li>{@link #gibbs}: Gibbs sampling, one variable at a time from its conditional given its
 *       Markov blanket; for binary pairwise models this is exactly a p-bit network's update
 *       ({@code PBitIsingMachine}).</li>
 * </ul>
 * Evidence is a delta factor ({@link #observe}).
 */
public final class FactorGraph {

    /** A factor over some variables, as a dense table with the last variable fastest. */
    private record Factor(int[] vars, double[] table) {
    }

    private final List<Integer> cards = new ArrayList<>();
    private final List<Factor> factors = new ArrayList<>();

    /** Creates an empty graph. */
    public FactorGraph() {
    }

    /**
     * Adds a variable.
     *
     * @param cardinality its number of values, at least 2
     * @return its index
     */
    public int variable(int cardinality) {
        cards.add(cardinality);
        return cards.size() - 1;
    }

    /**
     * Adds a factor.
     *
     * @param vars the variables it depends on
     * @param table its values, row-major with the last variable fastest, non-negative
     * @return this graph
     */
    public FactorGraph factor(int[] vars, double[] table) {
        int size = 1;
        for (int v : vars) {
            size *= cards.get(v);
        }
        if (table.length != size) {
            throw new IllegalArgumentException("factor needs " + size + " entries, got " + table.length);
        }
        factors.add(new Factor(vars.clone(), table.clone()));
        return this;
    }

    /**
     * Adds a conditional probability table {@code p(child | parents)}.
     *
     * @param child the child variable
     * @param parents the parents, in the table's order
     * @param table {@code p(child | parents)}, parents slowest and the child fastest
     * @return this graph
     */
    public FactorGraph conditional(int child, int[] parents, double[] table) {
        int[] vars = Arrays.copyOf(parents, parents.length + 1);
        vars[parents.length] = child;
        return factor(vars, table);
    }

    /**
     * Clamps a variable to an observed value.
     *
     * @param variable the variable
     * @param value its value
     * @return this graph
     */
    public FactorGraph observe(int variable, int value) {
        double[] t = new double[cards.get(variable)];
        t[value] = 1;
        return factor(new int[] {variable}, t);
    }

    /** @return the number of variables */
    public int variables() {
        return cards.size();
    }

    /**
     * A binary pairwise Ising model {@code p(m) ~ exp(beta (sum_i<j J_ij m_i m_j + sum_i h_i m_i))}
     * over spins {@code m = 2x - 1}, value 1 meaning spin up.
     *
     * @param j symmetric couplings with a zero diagonal
     * @param h the fields
     * @param beta the inverse temperature
     * @return the factor graph
     */
    public static FactorGraph ising(double[][] j, double[] h, double beta) {
        FactorGraph g = new FactorGraph();
        int n = h.length;
        for (int i = 0; i < n; i++) {
            g.variable(2);
        }
        for (int i = 0; i < n; i++) {
            g.factor(new int[] {i}, new double[] {Math.exp(-beta * h[i]), Math.exp(beta * h[i])});
            for (int k = i + 1; k < n; k++) {
                if (j[i][k] != 0) {
                    double same = Math.exp(beta * j[i][k]);
                    double diff = Math.exp(-beta * j[i][k]);
                    g.factor(new int[] {i, k}, new double[] {same, diff, diff, same});
                }
            }
        }
        return g;
    }

    // ------------------------------------------------------------------ variable elimination

    private Factor multiply(Factor a, Factor b) {
        Set<Integer> s = new java.util.LinkedHashSet<>();
        for (int v : a.vars()) {
            s.add(v);
        }
        for (int v : b.vars()) {
            s.add(v);
        }
        int[] vars = s.stream().mapToInt(Integer::intValue).toArray();
        int size = 1;
        for (int v : vars) {
            size *= cards.get(v);
        }
        double[] t = new double[size];
        int[] assign = new int[vars.length];
        for (int idx = 0; idx < size; idx++) {
            int rest = idx;
            for (int k = vars.length - 1; k >= 0; k--) {
                assign[k] = rest % cards.get(vars[k]);
                rest /= cards.get(vars[k]);
            }
            t[idx] = a.table()[index(a, vars, assign)] * b.table()[index(b, vars, assign)];
        }
        return new Factor(vars, t);
    }

    /** The table index of a factor under a full assignment of {@code vars}. */
    private int index(Factor f, int[] vars, int[] assign) {
        int idx = 0;
        for (int v : f.vars()) {
            int pos = 0;
            while (vars[pos] != v) {
                pos++;
            }
            idx = idx * cards.get(v) + assign[pos];
        }
        return idx;
    }

    private Factor sumOut(Factor f, int variable) {
        int pos = 0;
        while (f.vars()[pos] != variable) {
            pos++;
        }
        int[] vars = new int[f.vars().length - 1];
        for (int k = 0, m = 0; k < f.vars().length; k++) {
            if (k != pos) {
                vars[m++] = f.vars()[k];
            }
        }
        int inner = 1;
        for (int k = pos + 1; k < f.vars().length; k++) {
            inner *= cards.get(f.vars()[k]);
        }
        int card = cards.get(variable);
        double[] t = new double[f.table().length / card];
        for (int idx = 0; idx < f.table().length; idx++) {
            int outer = idx / (inner * card);
            int in = idx % inner;
            t[outer * inner + in] += f.table()[idx];
        }
        return new Factor(vars, t);
    }

    /**
     * The exact marginal of one variable by variable elimination (min-neighbour order).
     *
     * @param query the variable
     * @return {@code p(query)}
     */
    public double[] exactMarginal(int query) {
        List<Factor> pool = new ArrayList<>(factors);
        Set<Integer> remaining = new HashSet<>();
        for (int v = 0; v < cards.size(); v++) {
            if (v != query) {
                remaining.add(v);
            }
        }
        while (!remaining.isEmpty()) {
            int best = -1;
            int bestSize = Integer.MAX_VALUE;
            for (int v : remaining) {
                Set<Integer> nb = new HashSet<>();
                for (Factor f : pool) {
                    if (contains(f, v)) {
                        for (int u : f.vars()) {
                            nb.add(u);
                        }
                    }
                }
                if (nb.size() < bestSize) {
                    bestSize = nb.size();
                    best = v;
                }
            }
            Factor product = null;
            List<Factor> keep = new ArrayList<>();
            for (Factor f : pool) {
                if (contains(f, best)) {
                    product = product == null ? f : multiply(product, f);
                } else {
                    keep.add(f);
                }
            }
            if (product != null) {
                keep.add(sumOut(product, best));
            }
            pool = keep;
            remaining.remove(best);
        }
        Factor result = null;
        for (Factor f : pool) {
            result = result == null ? f : multiply(result, f);
        }
        double[] m = new double[cards.get(query)];
        if (result == null || result.vars().length == 0) {
            Arrays.fill(m, 1.0 / m.length);
            return m;
        }
        System.arraycopy(result.table(), 0, m, 0, m.length);
        return normalise(m);
    }

    /**
     * Exact marginals of every variable.
     *
     * @return {@code [variable][value]}
     */
    public double[][] exactMarginals() {
        double[][] out = new double[cards.size()][];
        for (int v = 0; v < out.length; v++) {
            out[v] = exactMarginal(v);
        }
        return out;
    }

    private static boolean contains(Factor f, int v) {
        for (int u : f.vars()) {
            if (u == v) {
                return true;
            }
        }
        return false;
    }

    private static double[] normalise(double[] m) {
        double s = 0;
        for (double v : m) {
            s += v;
        }
        for (int i = 0; i < m.length; i++) {
            m[i] = s > 0 ? m[i] / s : 1.0 / m.length;
        }
        return m;
    }

    // ------------------------------------------------------------------ loopy belief propagation

    /**
     * Sum-product belief propagation with damped, flooding updates.
     *
     * @param iterations the most message-passing rounds
     * @param damping the share of the old message kept, in {@code [0, 1)}
     * @return the approximate marginals {@code [variable][value]}
     */
    public double[][] loopyBeliefPropagation(int iterations, double damping) {
        int nf = factors.size();
        double[][][] toVar = new double[nf][][];
        double[][][] toFactor = new double[nf][][];
        for (int f = 0; f < nf; f++) {
            int[] vars = factors.get(f).vars();
            toVar[f] = new double[vars.length][];
            toFactor[f] = new double[vars.length][];
            for (int k = 0; k < vars.length; k++) {
                toVar[f][k] = uniform(cards.get(vars[k]));
                toFactor[f][k] = uniform(cards.get(vars[k]));
            }
        }
        List<List<int[]>> adjacency = new ArrayList<>();
        for (int v = 0; v < cards.size(); v++) {
            adjacency.add(new ArrayList<>());
        }
        for (int f = 0; f < nf; f++) {
            int[] vars = factors.get(f).vars();
            for (int k = 0; k < vars.length; k++) {
                adjacency.get(vars[k]).add(new int[] {f, k});
            }
        }
        for (int it = 0; it < iterations; it++) {
            double change = 0;
            // variable -> factor
            for (int v = 0; v < cards.size(); v++) {
                for (int[] fk : adjacency.get(v)) {
                    double[] m = new double[cards.get(v)];
                    Arrays.fill(m, 1);
                    for (int[] gk : adjacency.get(v)) {
                        if (gk[0] != fk[0] || gk[1] != fk[1]) {
                            for (int x = 0; x < m.length; x++) {
                                m[x] *= toVar[gk[0]][gk[1]][x];
                            }
                        }
                    }
                    toFactor[fk[0]][fk[1]] = normalise(m);
                }
            }
            // factor -> variable
            for (int f = 0; f < nf; f++) {
                Factor fac = factors.get(f);
                int[] vars = fac.vars();
                double[][] out = new double[vars.length][];
                for (int k = 0; k < vars.length; k++) {
                    out[k] = new double[cards.get(vars[k])];
                }
                int[] assign = new int[vars.length];
                for (int idx = 0; idx < fac.table().length; idx++) {
                    int rest = idx;
                    for (int k = vars.length - 1; k >= 0; k--) {
                        assign[k] = rest % cards.get(vars[k]);
                        rest /= cards.get(vars[k]);
                    }
                    for (int k = 0; k < vars.length; k++) {
                        double p = fac.table()[idx];
                        for (int u = 0; u < vars.length && p != 0; u++) {
                            if (u != k) {
                                p *= toFactor[f][u][assign[u]];
                            }
                        }
                        out[k][assign[k]] += p;
                    }
                }
                for (int k = 0; k < vars.length; k++) {
                    double[] fresh = normalise(out[k]);
                    for (int x = 0; x < fresh.length; x++) {
                        double v = damping * toVar[f][k][x] + (1 - damping) * fresh[x];
                        change = Math.max(change, Math.abs(v - toVar[f][k][x]));
                        fresh[x] = v;
                    }
                    toVar[f][k] = fresh;
                }
            }
            if (change < 1e-10) {
                break;
            }
        }
        double[][] marginals = new double[cards.size()][];
        for (int v = 0; v < cards.size(); v++) {
            double[] m = new double[cards.get(v)];
            Arrays.fill(m, 1);
            for (int[] fk : adjacency.get(v)) {
                for (int x = 0; x < m.length; x++) {
                    m[x] *= toVar[fk[0]][fk[1]][x];
                }
            }
            marginals[v] = normalise(m);
        }
        return marginals;
    }

    private static double[] uniform(int n) {
        double[] u = new double[n];
        Arrays.fill(u, 1.0 / n);
        return u;
    }

    // ------------------------------------------------------------------ Gibbs sampling

    /**
     * Gibbs sampling: sequential sweeps, each variable drawn from its conditional.
     *
     * @param sweeps sweeps counted toward the estimate
     * @param burnIn sweeps discarded first
     * @param seed the seed
     * @return the estimated marginals {@code [variable][value]}
     */
    public double[][] gibbs(int sweeps, int burnIn, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int n = cards.size();
        List<List<Integer>> touching = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            touching.add(new ArrayList<>());
        }
        for (int f = 0; f < factors.size(); f++) {
            for (int v : factors.get(f).vars()) {
                touching.get(v).add(f);
            }
        }
        int[] state = new int[n];
        // start from a state with positive probability: each variable at its unary-evidence value if any
        for (Factor f : factors) {
            if (f.vars().length == 1) {
                int best = 0;
                for (int x = 1; x < f.table().length; x++) {
                    if (f.table()[x] > f.table()[best]) {
                        best = x;
                    }
                }
                if (f.table()[best] == 1 && Arrays.stream(f.table()).sum() == 1) {
                    state[f.vars()[0]] = best;
                }
            }
        }
        double[][] counts = new double[n][];
        for (int v = 0; v < n; v++) {
            counts[v] = new double[cards.get(v)];
        }
        for (int s = 0; s < sweeps + burnIn; s++) {
            for (int v = 0; v < n; v++) {
                double[] p = new double[cards.get(v)];
                for (int x = 0; x < p.length; x++) {
                    state[v] = x;
                    double prod = 1;
                    for (int f : touching.get(v)) {
                        Factor fac = factors.get(f);
                        int idx = 0;
                        for (int u : fac.vars()) {
                            idx = idx * cards.get(u) + state[u];
                        }
                        prod *= fac.table()[idx];
                    }
                    p[x] = prod;
                }
                normalise(p);
                double u = random.nextDouble();
                int pick = p.length - 1;
                for (int x = 0; x < p.length; x++) {
                    u -= p[x];
                    if (u < 0) {
                        pick = x;
                        break;
                    }
                }
                state[v] = pick;
            }
            if (s >= burnIn) {
                for (int v = 0; v < n; v++) {
                    counts[v][state[v]]++;
                }
            }
        }
        for (double[] c : counts) {
            normalise(c);
        }
        return counts;
    }

    /**
     * The mean absolute difference between two sets of marginals.
     *
     * @param a marginals
     * @param b marginals
     * @return the mean absolute error over every variable and value
     */
    public static double marginalError(double[][] a, double[][] b) {
        double s = 0;
        int c = 0;
        for (int v = 0; v < a.length; v++) {
            for (int x = 0; x < a[v].length; x++) {
                s += Math.abs(a[v][x] - b[v][x]);
                c++;
            }
        }
        return s / c;
    }
}
