package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.experimental.ml.FactorGraph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Markov logic networks (Richardson &amp; Domingos, MLJ 2006) over unary and binary predicates.
 * Each formula is a weighted first-order clause (or a hard one); grounding it over {@code n}
 * constants gives a Markov random field with {@code P(x) ~ exp(sum_i w_i n_i(x))}, where
 * {@code n_i} counts the satisfied groundings of formula {@code i}. Evidence atoms are fixed and
 * simplify the ground clauses away.
 *
 * <ul>
 *   <li>Inference: {@link Inference#GIBBS} (single-site, with hard clauses as a large finite
 *       weight, so it mixes poorly when rules are near-deterministic), {@link Inference#MC_SAT}
 *       (Poon &amp; Domingos, AAAI 2006: a slice sampler that keeps each satisfied ground clause
 *       with probability {@code 1 - exp(-w)}, always the hard ones, and samples near-uniformly
 *       from the assignments satisfying them by SampleSAT, a mix of WalkSAT and simulated
 *       annealing moves), and {@link Inference#EXACT} enumeration for tiny worlds. MAP by
 *       MaxWalkSAT ({@link #map}).</li>
 *   <li>Learning ({@link #learn}): {@link Learning#PSEUDO_LIKELIHOOD} (gradient ascent on the
 *       conditional pseudo-log-likelihood, with per-weight steps scaled by a curvature bound) and
 *       {@link Learning#DIAGONAL_NEWTON} (Lowd &amp; Domingos, ECML 2007: the likelihood gradient
 *       {@code n_i - E[n_i]} divided by {@code Var[n_i]}, both estimated by MC-SAT samples).</li>
 *   <li>{@link #toFactorGraph}: the same ground network as a generic factor graph, for the
 *       variable-elimination, loopy-BP and Gibbs baselines.</li>
 * </ul>
 */
public final class MarkovLogic {

    /** Marginal inference methods. */
    public enum Inference {
        /** Single-site Gibbs sampling. */
        GIBBS,
        /** MC-SAT slice sampling. */
        MC_SAT,
        /** Enumeration of all query assignments (at most about 20 query atoms). */
        EXACT
    }

    /** Weight learning methods. */
    public enum Learning {
        /** Conditional pseudo-likelihood. */
        PSEUDO_LIKELIHOOD,
        /** Diagonal Newton on the conditional likelihood with MC-SAT expectations. */
        DIAGONAL_NEWTON
    }

    /**
     * A literal of a formula.
     *
     * @param positive the sign
     * @param predicate the predicate index
     * @param vars the logical variables (one or two), numbered from 0 within the formula
     */
    public record Literal(boolean positive, int predicate, int... vars) {
    }

    /**
     * A clause template.
     *
     * @param name a label
     * @param literals the disjunction
     * @param hard whether it is a hard constraint
     */
    public record Formula(String name, List<Literal> literals, boolean hard) {
    }

    /**
     * A ground network.
     *
     * @param constants the domain size
     * @param values the atom values (evidence values are fixed; others are the ground truth or
     *     an initial state)
     * @param evidence which atoms are observed
     * @param clauses ground clauses over atoms as DIMACS-like literals ({@code atom + 1} or its
     *     negation), after removing evidence-satisfied clauses and false evidence literals
     * @param formula the formula of each ground clause
     */
    public record Ground(int constants, boolean[] values, boolean[] evidence, List<int[]> clauses, int[] formula) {
    }

    private final int[] arity;
    private final List<Formula> formulas;
    private final double[] weights;
    private final double hardWeight;

    /**
     * Creates a network.
     *
     * @param arity per predicate its arity (1 or 2)
     * @param formulas the formulas
     * @param weights initial weights of the formulas (ignored for hard ones)
     * @param hardWeight the finite weight used for hard formulas by Gibbs and the factor graph
     */
    public MarkovLogic(int[] arity, List<Formula> formulas, double[] weights, double hardWeight) {
        this.arity = arity.clone();
        this.formulas = List.copyOf(formulas);
        this.weights = weights.clone();
        this.hardWeight = hardWeight;
    }

    /** @return the formula weights */
    public double[] weights() {
        return weights.clone();
    }

    /**
     * The number of ground atoms.
     *
     * @param constants the domain size
     * @return atoms over all predicates
     */
    public int atoms(int constants) {
        int a = 0;
        for (int ar : arity) {
            a += ar == 1 ? constants : constants * constants;
        }
        return a;
    }

    /**
     * The index of a ground atom.
     *
     * @param constants the domain size
     * @param predicate the predicate
     * @param args its constants
     * @return the atom index
     */
    public int atom(int constants, int predicate, int... args) {
        int off = 0;
        for (int p = 0; p < predicate; p++) {
            off += arity[p] == 1 ? constants : constants * constants;
        }
        return off + (arity[predicate] == 1 ? args[0] : args[0] * constants + args[1]);
    }

    private double weightOf(int f) {
        return formulas.get(f).hard() ? hardWeight : weights[f];
    }

    /**
     * Grounds the formulas.
     *
     * @param constants the domain size
     * @param values the atom values
     * @param evidence the observed atoms
     * @return the ground network
     */
    public Ground ground(int constants, boolean[] values, boolean[] evidence) {
        List<int[]> clauses = new ArrayList<>();
        List<Integer> formulaOf = new ArrayList<>();
        for (int f = 0; f < formulas.size(); f++) {
            List<Literal> lits = formulas.get(f).literals();
            int vars = 0;
            for (Literal l : lits) {
                for (int v : l.vars()) {
                    vars = Math.max(vars, v + 1);
                }
            }
            int[] bind = new int[vars];
            int total = (int) Math.pow(constants, vars);
            for (int g = 0; g < total; g++) {
                int rem = g;
                for (int v = 0; v < vars; v++) {
                    bind[v] = rem % constants;
                    rem /= constants;
                }
                int[] out = new int[lits.size()];
                int k = 0;
                boolean satisfied = false;
                for (Literal l : lits) {
                    int[] args = new int[l.vars().length];
                    for (int i = 0; i < args.length; i++) {
                        args[i] = bind[l.vars()[i]];
                    }
                    int a = atom(constants, l.predicate(), args);
                    if (evidence[a]) {
                        if (values[a] == l.positive()) {
                            satisfied = true;
                            break;
                        }
                        continue;
                    }
                    int lit = l.positive() ? a + 1 : -(a + 1);
                    boolean dup = false;
                    for (int j = 0; j < k; j++) {
                        if (out[j] == lit) {
                            dup = true;
                        } else if (out[j] == -lit) {
                            satisfied = true;
                        }
                    }
                    if (!dup) {
                        out[k++] = lit;
                    }
                }
                if (satisfied || k == 0) {
                    continue;
                }
                clauses.add(Arrays.copyOf(out, k));
                formulaOf.add(f);
            }
        }
        return new Ground(constants, values.clone(), evidence.clone(), clauses,
            formulaOf.stream().mapToInt(Integer::intValue).toArray());
    }

    // ---- shared sampler state ------------------------------------------------------------------

    /** Clause bookkeeping for local search and sampling over the query atoms. */
    private static final class State {
        final Ground g;
        final boolean[] x;
        final int[][] occ;
        final int[] trueCount;

        State(Ground g, boolean[] x) {
            this.g = g;
            this.x = x;
            int n = x.length;
            int[] deg = new int[n];
            for (int[] c : g.clauses()) {
                for (int l : c) {
                    deg[Math.abs(l) - 1]++;
                }
            }
            occ = new int[n][];
            for (int a = 0; a < n; a++) {
                occ[a] = new int[deg[a]];
            }
            int[] fill = new int[n];
            trueCount = new int[g.clauses().size()];
            for (int c = 0; c < g.clauses().size(); c++) {
                for (int l : g.clauses().get(c)) {
                    int a = Math.abs(l) - 1;
                    occ[a][fill[a]++] = c;
                    if (x[a] == l > 0) {
                        trueCount[c]++;
                    }
                }
            }
        }

        void flip(int a) {
            x[a] = !x[a];
            for (int c : occ[a]) {
                for (int l : g.clauses().get(c)) {
                    if (Math.abs(l) - 1 == a) {
                        trueCount[c] += x[a] == l > 0 ? 1 : -1;
                    }
                }
            }
        }

        /** Change in the summed weight of unsatisfied clauses (restricted to {@code inM} if given) if {@code a} flips. */
        double delta(int a, double[] w, boolean[] inM) {
            double d = 0;
            for (int c : occ[a]) {
                if (inM != null && !inM[c]) {
                    continue;
                }
                int lit = 0;
                for (int l : g.clauses().get(c)) {
                    if (Math.abs(l) - 1 == a) {
                        lit = l;
                    }
                }
                boolean litTrue = x[a] == lit > 0;
                if (litTrue && trueCount[c] == 1) {
                    d += w == null ? 1 : w[c];
                } else if (!litTrue && trueCount[c] == 0) {
                    d -= w == null ? 1 : w[c];
                }
            }
            return d;
        }
    }

    private double[] clauseWeights(Ground g) {
        double[] w = new double[g.clauses().size()];
        for (int c = 0; c < w.length; c++) {
            w[c] = weightOf(g.formula()[c]);
        }
        return w;
    }

    private static List<Integer> queryAtoms(Ground g) {
        List<Integer> q = new ArrayList<>();
        for (int a = 0; a < g.values().length; a++) {
            if (!g.evidence()[a]) {
                q.add(a);
            }
        }
        return q;
    }

    // ---- inference -----------------------------------------------------------------------------

    /**
     * Marginal probabilities of every atom (evidence atoms report their value).
     *
     * @param g the ground network
     * @param method the inference method
     * @param samples samples kept (Gibbs sweeps or MC-SAT steps)
     * @param burnIn samples discarded first
     * @param randomInit whether samplers start from a random state (else all query atoms false)
     * @param seed the seed
     * @return per atom {@code P(true)}
     */
    public double[] marginals(Ground g, Inference method, int samples, int burnIn, boolean randomInit, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Integer> q = queryAtoms(g);
        boolean[] x = g.values().clone();
        for (int a : q) {
            x[a] = randomInit && r.nextBoolean();
        }
        double[] out = new double[x.length];
        for (int a = 0; a < x.length; a++) {
            if (g.evidence()[a]) {
                out[a] = x[a] ? 1 : 0;
            }
        }
        double[] w = clauseWeights(g);
        if (method == Inference.EXACT) {
            exact(g, q, w, out);
            return out;
        }
        if (method == Inference.MC_SAT) {
            mcSat(g, x, samples, burnIn, r, cur -> {
                for (int a : q) {
                    out[a] += cur[a] ? 1.0 / samples : 0;
                }
            });
            return out;
        }
        State s = new State(g, x);
        for (int it = 0; it < samples + burnIn; it++) {
            for (int a : q) {
                // energy gain of a = true over a = false
                double d = s.delta(a, w, null);
                double gainIfFlip = -d;
                double logitTrue = x[a] ? -gainIfFlip : gainIfFlip;
                boolean val = r.nextDouble() < 1 / (1 + Math.exp(-logitTrue));
                if (val != x[a]) {
                    s.flip(a);
                }
            }
            if (it >= burnIn) {
                for (int a : q) {
                    out[a] += x[a] ? 1.0 / samples : 0;
                }
            }
        }
        return out;
    }

    private static void exact(Ground g, List<Integer> q, double[] w, double[] out) {
        if (q.size() > 24) {
            throw new IllegalArgumentException("too many query atoms for enumeration: " + q.size());
        }
        boolean[] x = g.values().clone();
        double z = 0;
        double[] acc = new double[q.size()];
        double[] logs = new double[1 << q.size()];
        double max = Double.NEGATIVE_INFINITY;
        for (int m = 0; m < 1 << q.size(); m++) {
            for (int i = 0; i < q.size(); i++) {
                x[q.get(i)] = (m >> i & 1) == 1;
            }
            double e = 0;
            for (int c = 0; c < g.clauses().size(); c++) {
                for (int l : g.clauses().get(c)) {
                    if (x[Math.abs(l) - 1] == l > 0) {
                        e += w[c];
                        break;
                    }
                }
            }
            logs[m] = e;
            max = Math.max(max, e);
        }
        for (int m = 0; m < logs.length; m++) {
            double p = Math.exp(logs[m] - max);
            z += p;
            for (int i = 0; i < q.size(); i++) {
                if ((m >> i & 1) == 1) {
                    acc[i] += p;
                }
            }
        }
        for (int i = 0; i < q.size(); i++) {
            out[q.get(i)] = acc[i] / z;
        }
    }

    /**
     * Runs MC-SAT from {@code x} (modified in place) and calls {@code onSample} after every kept
     * step. Unit clauses with a negative weight are replaced by their negation with the opposite
     * weight (an exact rewrite); longer clauses with a negative weight are left out of the slice.
     */
    private void mcSat(Ground g, boolean[] x, int samples, int burnIn, SplittableRandom r,
            java.util.function.Consumer<boolean[]> onSample) {
        int nc = g.clauses().size();
        List<int[]> clauses = new ArrayList<>(nc);
        boolean[] hard = new boolean[nc];
        double[] w = new double[nc];
        for (int c = 0; c < nc; c++) {
            int[] lits = g.clauses().get(c);
            hard[c] = formulas.get(g.formula()[c]).hard();
            w[c] = weightOf(g.formula()[c]);
            if (!hard[c] && w[c] < 0 && lits.length == 1) {
                lits = new int[] {-lits[0]};
                w[c] = -w[c];
            }
            clauses.add(lits);
        }
        Ground sampling = new Ground(g.constants(), g.values(), g.evidence(), clauses, g.formula());
        List<Integer> q = queryAtoms(g);
        State s = new State(sampling, x);
        // start from a state satisfying the hard clauses
        sampleSat(s, q, hard, 100 * Math.max(1, q.size()), r, false);
        boolean[] inM = new boolean[nc];
        int flips = 10 * Math.max(1, q.size());
        for (int it = 0; it < samples + burnIn; it++) {
            for (int c = 0; c < nc; c++) {
                inM[c] = s.trueCount[c] > 0 && (hard[c] || w[c] > 0 && r.nextDouble() < 1 - Math.exp(-w[c]));
            }
            sampleSat(s, q, inM, flips, r, true);
            if (it >= burnIn) {
                onSample.accept(x);
            }
        }
    }

    /**
     * SampleSAT (Wei, Erenrich &amp; Selman 2004): WalkSAT moves on unsatisfied clauses of
     * {@code inM} mixed with fixed-temperature simulated-annealing moves.
     */
    private static void sampleSat(State s, List<Integer> q, boolean[] inM, int flips, SplittableRandom r,
            boolean anneal) {
        if (q.isEmpty()) {
            return;
        }
        Ground g = s.g;
        List<Integer> unsat = new ArrayList<>();
        boolean clean = anneal; // MC-SAT enters with every clause of M satisfied
        // after the mixing budget, WalkSAT-only steps repair any clause of M an annealing move broke
        int limit = anneal ? 3 * flips : flips;
        for (int step = 0; step < limit; step++) {
            if (step >= flips && clean) {
                return;
            }
            boolean walk = !clean && (!anneal || step >= flips || r.nextDouble() < 0.5);
            if (walk) {
                // a lazily refreshed list of unsatisfied clauses: stale entries are dropped when drawn
                int chosen = -1;
                for (int pass = 0; pass < 2 && chosen < 0; pass++) {
                    if (pass == 1) {
                        unsat.clear();
                        for (int c = 0; c < inM.length; c++) {
                            if (inM[c] && s.trueCount[c] == 0) {
                                unsat.add(c);
                            }
                        }
                    }
                    while (!unsat.isEmpty()) {
                        int idx = r.nextInt(unsat.size());
                        int c = unsat.get(idx);
                        if (inM[c] && s.trueCount[c] == 0) {
                            chosen = c;
                            break;
                        }
                        unsat.set(idx, unsat.get(unsat.size() - 1));
                        unsat.remove(unsat.size() - 1);
                    }
                }
                if (chosen < 0) {
                    clean = true;
                    if (!anneal || step >= flips) {
                        return;
                    }
                    walk = false;
                } else {
                    int[] c = g.clauses().get(chosen);
                    int pick;
                    if (r.nextDouble() < 0.5) {
                        pick = Math.abs(c[r.nextInt(c.length)]) - 1;
                    } else {
                        pick = -1;
                        double best = Double.POSITIVE_INFINITY;
                        for (int l : c) {
                            double d = s.delta(Math.abs(l) - 1, null, inM);
                            if (d < best) {
                                best = d;
                                pick = Math.abs(l) - 1;
                            }
                        }
                    }
                    s.flip(pick);
                    continue;
                }
            }
            int a = q.get(r.nextInt(q.size()));
            double d = s.delta(a, null, inM);
            if (d <= 0 || r.nextDouble() < Math.exp(-d / 0.5)) {
                clean &= d <= 0;
                s.flip(a);
            }
        }
    }

    /**
     * MAP inference by MaxWalkSAT: minimises the weight of unsatisfied ground clauses.
     *
     * @param g the ground network
     * @param flips the flip budget
     * @param seed the seed
     * @return the best assignment of all atoms
     */
    public boolean[] map(Ground g, int flips, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Integer> q = queryAtoms(g);
        boolean[] x = g.values().clone();
        for (int a : q) {
            x[a] = false;
        }
        State s = new State(g, x);
        double[] w = clauseWeights(g);
        double cost = 0;
        for (int c = 0; c < w.length; c++) {
            cost += s.trueCount[c] == 0 ? w[c] : 0;
        }
        double best = cost;
        boolean[] bestX = x.clone();
        List<Integer> unsat = new ArrayList<>();
        for (int step = 0; step < flips; step++) {
            unsat.clear();
            for (int c = 0; c < w.length; c++) {
                if (s.trueCount[c] == 0) {
                    unsat.add(c);
                }
            }
            if (unsat.isEmpty()) {
                break;
            }
            int[] c = g.clauses().get(unsat.get(r.nextInt(unsat.size())));
            int pick;
            if (r.nextDouble() < 0.2) {
                pick = Math.abs(c[r.nextInt(c.length)]) - 1;
            } else {
                pick = -1;
                double bd = Double.POSITIVE_INFINITY;
                for (int l : c) {
                    double d = s.delta(Math.abs(l) - 1, w, null);
                    if (d < bd) {
                        bd = d;
                        pick = Math.abs(l) - 1;
                    }
                }
            }
            cost += s.delta(pick, w, null);
            s.flip(pick);
            if (cost < best - 1e-12) {
                best = cost;
                bestX = x.clone();
            }
        }
        return bestX;
    }

    /**
     * The ground network as a factor graph over its query atoms (binary variables in the order of
     * the atoms), one factor per ground clause with {@code exp(w)} on its satisfying rows.
     *
     * @param g the ground network
     * @param variableOf receives per atom its factor-graph variable, or -1 for evidence atoms
     * @return the factor graph
     */
    public FactorGraph toFactorGraph(Ground g, int[] variableOf) {
        FactorGraph fg = new FactorGraph();
        Arrays.fill(variableOf, -1);
        for (int a = 0; a < g.values().length; a++) {
            if (!g.evidence()[a]) {
                variableOf[a] = fg.variable(2);
            }
        }
        double[] w = clauseWeights(g);
        for (int c = 0; c < w.length; c++) {
            int[] lits = g.clauses().get(c);
            int[] vars = new int[lits.length];
            for (int i = 0; i < lits.length; i++) {
                vars[i] = variableOf[Math.abs(lits[i]) - 1];
            }
            double[] table = new double[1 << lits.length];
            for (int row = 0; row < table.length; row++) {
                boolean sat = false;
                for (int i = 0; i < lits.length; i++) {
                    boolean val = (row >> (lits.length - 1 - i) & 1) == 1;
                    sat |= val == lits[i] > 0;
                }
                table[row] = sat ? Math.exp(w[c]) : 1;
            }
            fg.factor(vars, table);
        }
        return fg;
    }

    // ---- learning ------------------------------------------------------------------------------

    /**
     * Satisfied groundings per formula.
     *
     * @param g the ground network
     * @param x an assignment of all atoms
     * @return per formula the count
     */
    public double[] counts(Ground g, boolean[] x) {
        double[] n = new double[formulas.size()];
        for (int c = 0; c < g.clauses().size(); c++) {
            for (int l : g.clauses().get(c)) {
                if (x[Math.abs(l) - 1] == l > 0) {
                    n[g.formula()[c]]++;
                    break;
                }
            }
        }
        return n;
    }

    /**
     * Learns the soft formula weights from a fully observed world, conditioning on the atoms of
     * the evidence predicates.
     *
     * @param method the learner
     * @param constants the domain size
     * @param values the world
     * @param evidencePredicates predicates treated as given
     * @param iterations optimisation steps
     * @param samples MC-SAT samples per step (diagonal Newton only)
     * @param sigma the Gaussian prior's standard deviation
     * @param seed the seed
     * @return the learned weights (also set on this network)
     */
    public double[] learn(Learning method, int constants, boolean[] values, int[] evidencePredicates,
            int iterations, int samples, double sigma, long seed) {
        boolean[] evidence = new boolean[values.length];
        for (int p : evidencePredicates) {
            int n = arity[p] == 1 ? constants : constants * constants;
            int off = atom(constants, p, arity[p] == 1 ? new int[] {0} : new int[] {0, 0});
            Arrays.fill(evidence, off, off + n, true);
        }
        Ground g = ground(constants, values, evidence);
        int nf = formulas.size();
        boolean[] soft = new boolean[nf];
        for (int f = 0; f < nf; f++) {
            soft[f] = !formulas.get(f).hard();
        }
        if (method == Learning.PSEUDO_LIKELIHOOD) {
            // per query atom, the change of each formula's count when the atom is true vs false
            List<Integer> q = queryAtoms(g);
            State s = new State(g, values.clone());
            double[][] delta = new double[q.size()][nf];
            for (int i = 0; i < q.size(); i++) {
                int a = q.get(i);
                for (int c : s.occ[a]) {
                    int lit = 0;
                    for (int l : g.clauses().get(c)) {
                        if (Math.abs(l) - 1 == a) {
                            lit = l;
                        }
                    }
                    int others = s.trueCount[c] - (values[a] == lit > 0 ? 1 : 0);
                    if (others == 0) {
                        delta[i][g.formula()[c]] += lit > 0 ? 1 : -1;
                    }
                }
            }
            double[] curv = new double[nf];
            for (double[] d : delta) {
                for (int f = 0; f < nf; f++) {
                    curv[f] += d[f] * d[f] / 4;
                }
            }
            for (int it = 0; it < iterations; it++) {
                double[] grad = new double[nf];
                for (int i = 0; i < q.size(); i++) {
                    double z = 0;
                    for (int f = 0; f < nf; f++) {
                        z += weightOf(f) * delta[i][f];
                    }
                    double p = 1 / (1 + Math.exp(-z));
                    double y = values[q.get(i)] ? 1 : 0;
                    for (int f = 0; f < nf; f++) {
                        grad[f] += (y - p) * delta[i][f];
                    }
                }
                for (int f = 0; f < nf; f++) {
                    if (soft[f]) {
                        weights[f] += (grad[f] - weights[f] / (sigma * sigma)) / (curv[f] + 1 / (sigma * sigma));
                    }
                }
            }
            return weights();
        }
        double[] observed = counts(g, values);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations; it++) {
            double[] mean = new double[nf];
            double[] sq = new double[nf];
            mcSat(g, values.clone(), samples, samples / 5, r, x -> {
                double[] n = counts(g, x);
                for (int f = 0; f < nf; f++) {
                    mean[f] += n[f] / samples;
                    sq[f] += n[f] * n[f] / samples;
                }
            });
            for (int f = 0; f < nf; f++) {
                if (soft[f]) {
                    double var = Math.max(sq[f] - mean[f] * mean[f], 0) + 1 / (sigma * sigma);
                    double step = (observed[f] - mean[f] - weights[f] / (sigma * sigma)) / var;
                    weights[f] += Math.max(-2, Math.min(2, step));
                }
            }
        }
        return weights();
    }
}
