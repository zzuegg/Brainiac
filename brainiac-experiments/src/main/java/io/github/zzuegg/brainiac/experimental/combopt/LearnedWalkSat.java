package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * Learned local search for SAT (Yolcu and Póczos): WalkSAT's loop (pick a random unsatisfied
 * clause, flip one of its variables) with the flip chosen by a graph neural network over the
 * clause–variable factor graph.
 *
 * <p>Variables start from their current value, break count and make count (the clauses a
 * flip would break or fix, scaled by the clause count), clauses from whether they are
 * satisfied and how many true literals they have; two rounds of mean message passing
 * (variables to clauses and back) give variable embeddings, and the policy is a softmax over
 * the scores of the chosen clause's variables, plus a linear skip from the raw features (which
 * lets the policy pick up WalkSAT's break-count rule quickly). Training is REINFORCE on the flips needed: an
 * episode that solves the formula after {@code T} flips rewards every flip with
 * {@code -T / budget} (and an unsolved one with {@code -1}), with the batch mean as baseline,
 * under a curriculum of growing formula size.
 *
 * <p>Paper: Yolcu and Póczos, "Learning Local Search Heuristics for Boolean Satisfiability"
 * (NeurIPS 2019), <a href="https://arxiv.org/abs/1909.11830">arXiv:1909.11830</a>.
 */
public final class LearnedWalkSat {

    private static final int VAR_FEATURES = 4;
    private static final int CLAUSE_FEATURES = 3;

    private final Var wv;
    private final Var wc;
    private final Var v2c;
    private final Var c2v;
    private final Var v2c2;
    private final Var c2v2;
    private final Var out;
    private final Var skip;

    private LearnedWalkSat(int hidden, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.wv = AttentionModel.matrix(r, VAR_FEATURES, hidden);
        this.wc = AttentionModel.matrix(r, CLAUSE_FEATURES, hidden);
        this.v2c = AttentionModel.matrix(r, hidden, hidden);
        this.c2v = AttentionModel.matrix(r, hidden, hidden);
        this.v2c2 = AttentionModel.matrix(r, hidden, hidden);
        this.c2v2 = AttentionModel.matrix(r, hidden, hidden);
        this.out = AttentionModel.matrix(r, hidden, 1);
        this.skip = AttentionModel.matrix(r, VAR_FEATURES, 1);
    }

    /**
     * Creates a policy.
     *
     * @param hidden the embedding width
     * @param seed the seed
     * @return the policy
     */
    public static LearnedWalkSat of(int hidden, long seed) {
        return new LearnedWalkSat(hidden, seed);
    }

    /** A formula's incidence, precomputed once per formula. */
    private record Graph(RandomSatFormulas.Formula f, Tensor varToClause, Tensor clauseToVar) {
        static Graph of(RandomSatFormulas.Formula f) {
            int n = f.variables();
            int m = f.clauses().size();
            float[] vc = new float[m * n];
            float[] cv = new float[n * m];
            int[] deg = new int[n];
            for (int c = 0; c < m; c++) {
                for (int lit : f.clauses().get(c)) {
                    deg[Math.abs(lit) - 1]++;
                }
            }
            for (int c = 0; c < m; c++) {
                int[] cl = f.clauses().get(c);
                for (int lit : cl) {
                    int v = Math.abs(lit) - 1;
                    vc[c * n + v] = 1f / cl.length;
                    cv[v * m + c] = 1f / Math.max(1, deg[v]);
                }
            }
            return new Graph(f, Tensor.of(Shape.of(m, n), vc), Tensor.of(Shape.of(n, m), cv));
        }
    }

    private Var scores(Graph g, boolean[] a) {
        RandomSatFormulas.Formula f = g.f();
        int n = f.variables();
        int m = f.clauses().size();
        float[] vf = new float[n * VAR_FEATURES];
        float[] cf = new float[m * CLAUSE_FEATURES];
        int[] trueCount = new int[m];
        int[] onlyTrue = new int[m];
        for (int c = 0; c < m; c++) {
            for (int lit : f.clauses().get(c)) {
                if (a[Math.abs(lit)] == lit > 0) {
                    trueCount[c]++;
                    onlyTrue[c] = Math.abs(lit);
                }
            }
            cf[c * CLAUSE_FEATURES] = trueCount[c] > 0 ? 1 : 0;
            cf[c * CLAUSE_FEATURES + 1] = trueCount[c] / 3f;
            cf[c * CLAUSE_FEATURES + 2] = 1;
        }
        for (int c = 0; c < m; c++) {
            for (int lit : f.clauses().get(c)) {
                int v = Math.abs(lit);
                if (trueCount[c] == 0) {
                    vf[(v - 1) * VAR_FEATURES + 2] += 1f;           // make
                } else if (trueCount[c] == 1 && onlyTrue[c] == v) {
                    vf[(v - 1) * VAR_FEATURES + 1] += 1f;           // break
                }
            }
        }
        float scale = 10f / m;
        for (int v = 1; v <= n; v++) {
            vf[(v - 1) * VAR_FEATURES] = a[v] ? 1 : -1;
            vf[(v - 1) * VAR_FEATURES + 1] *= scale;
            vf[(v - 1) * VAR_FEATURES + 2] *= scale;
            vf[(v - 1) * VAR_FEATURES + 3] = 1;
        }
        Var vc = Var.of(g.varToClause());
        Var cv = Var.of(g.clauseToVar());
        Var x = Var.of(Tensor.of(Shape.of(n, VAR_FEATURES), vf));
        Var hv = x.matmul(wv).relu();
        Var hc = Var.of(Tensor.of(Shape.of(m, CLAUSE_FEATURES), cf)).matmul(wc).add(vc.matmul(hv).matmul(v2c)).relu();
        hv = hv.add(cv.matmul(hc).matmul(c2v)).relu();
        hc = hc.add(vc.matmul(hv).matmul(v2c2)).relu();
        hv = hv.add(cv.matmul(hc).matmul(c2v2)).relu();
        return hv.matmul(out).add(x.matmul(skip)).reshape(n);
    }

    /**
     * An episode's outcome.
     *
     * @param flips flips taken (the budget if unsolved)
     * @param solved whether the formula was satisfied
     * @param logp the summed log-probabilities of the flips (with gradients when training)
     */
    public record Episode(int flips, boolean solved, Var logp) {
    }

    /**
     * Runs the learned local search.
     *
     * @param f the formula
     * @param budget the flip budget
     * @param random the source
     * @param grad keep the log-likelihood graph for training
     * @return the episode
     */
    public Episode solve(RandomSatFormulas.Formula f, int budget, SplittableRandom random, boolean grad) {
        Graph g = Graph.of(f);
        int n = f.variables();
        boolean[] a = new boolean[n + 1];
        for (int v = 1; v <= n; v++) {
            a[v] = random.nextBoolean();
        }
        Var logp = null;
        for (int s = 0; s < budget; s++) {
            List<int[]> unsat = new ArrayList<>();
            for (int[] c : f.clauses()) {
                boolean ok = false;
                for (int lit : c) {
                    ok |= a[Math.abs(lit)] == lit > 0;
                }
                if (!ok) {
                    unsat.add(c);
                }
            }
            if (unsat.isEmpty()) {
                return new Episode(s, true, logp);
            }
            int[] c = unsat.get(random.nextInt(unsat.size()));
            IntFunction<Var> step = k -> {
                Var sc = scores(g, a);
                float[] idx = new float[c.length];
                for (int i = 0; i < c.length; i++) {
                    idx[i] = Math.abs(c[i]) - 1;
                }
                return sc.gather(0, Tensor.of(Shape.of(c.length), idx)).reshape(1, c.length).logSoftmax(1);
            };
            Var lp = grad ? step.apply(0) : Var.noGrad(() -> step.apply(0));
            float[] p = lp.value().toArray();
            double u = random.nextDouble();
            int pick = c.length - 1;
            for (int i = 0; i < c.length; i++) {
                u -= Math.exp(p[i]);
                if (u <= 0) {
                    pick = i;
                    break;
                }
            }
            if (grad) {
                Var chosen = lp.reshape(c.length).slice(0, pick, pick + 1);
                logp = logp == null ? chosen : logp.add(chosen);
            }
            int v = Math.abs(c[pick]);
            a[v] = !a[v];
        }
        return new Episode(budget, RandomSatFormulas.satisfies(f.clauses(), a), logp);
    }

    /**
     * REINFORCE on a batch of formulas.
     *
     * @param batch formulas
     * @param budget the flip budget per episode
     * @param optimizer the optimizer
     * @param random the source
     * @return the batch's mean flips
     */
    public double trainStep(List<RandomSatFormulas.Formula> batch, int budget, Optimizer optimizer,
            SplittableRandom random) {
        List<Episode> eps = new ArrayList<>();
        double mean = 0;
        for (RandomSatFormulas.Formula f : batch) {
            Episode e = solve(f, budget, random, true);
            eps.add(e);
            mean += cost(e, budget);
        }
        mean /= batch.size();
        Var loss = null;
        double flips = 0;
        for (Episode e : eps) {
            flips += e.flips();
            if (e.logp() == null) {
                continue;
            }
            Var term = e.logp().mul((float) ((cost(e, budget) - mean) / Math.max(1, e.flips())));
            loss = loss == null ? term : loss.add(term);
        }
        if (loss != null) {
            AttentionModel.step(loss.mul(1f / batch.size()), parameters(), optimizer, 1.0);
        }
        return flips / batch.size();
    }

    private static double cost(Episode e, int budget) {
        return e.solved() ? e.flips() / (double) budget : 1.5;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.of(wv, wc, v2c, c2v, v2c2, c2v2, out, skip);
    }
}
