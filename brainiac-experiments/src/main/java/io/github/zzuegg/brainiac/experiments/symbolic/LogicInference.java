package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ConstraintInstances;
import io.github.zzuegg.brainiac.experimental.data.FamilyTreeGraph;
import io.github.zzuegg.brainiac.experimental.data.PlanningDomains;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.logic.AnswerSetSolver;
import io.github.zzuegg.brainiac.experimental.logic.CdclSolver;
import io.github.zzuegg.brainiac.experimental.logic.ClauseSelector;
import io.github.zzuegg.brainiac.experimental.logic.ConstraintSolver;
import io.github.zzuegg.brainiac.experimental.logic.Csp;
import io.github.zzuegg.brainiac.experimental.logic.KnowledgeCompiler;
import io.github.zzuegg.brainiac.experimental.logic.MarkovLogic;
import io.github.zzuegg.brainiac.experimental.logic.ResolutionProver;
import io.github.zzuegg.brainiac.experimental.ml.FactorGraph;
import io.github.zzuegg.brainiac.experimental.nn.DifferentiableIlp;
import io.github.zzuegg.brainiac.experimental.symbolic.SatPlanner;
import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;
import io.github.zzuegg.brainiac.experimental.train.SemanticLoss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 13 logic inference experiments (items 13.2.11 to 13.2.14): knowledge compilation and
 * weighted model counting, saturation theorem proving with learned clause selection, Markov
 * logic, and answer set programming. Every method returns table rows; the tests print them.
 */
public final class LogicInference {

    private LogicInference() {
    }

    /**
     * One table row.
     *
     * @param task the instance or family
     * @param method the method
     * @param value the headline metric (model count, error, rate, ...)
     * @param reference the baseline or ground-truth value of the same metric
     * @param size a size (circuit nodes plus edges, clauses, given clauses, ...)
     * @param millis wall time
     */
    public record Row(String task, String method, double value, double reference, double size, double millis) {

        @Override
        public String toString() {
            return String.format("%s/%s %.4g (ref %.4g) size=%.0f %.1fms", task, method, value, reference, size, millis);
        }
    }

    private static double binomial(int n, int k) {
        double r = 1;
        for (int i = 1; i <= k; i++) {
            r = r * (n - k + i) / i;
        }
        return r;
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    // ---- 13.2.11 knowledge compilation -----------------------------------------------------------

    /**
     * The compilation table: closed-form semantic losses against compiled WMC, grid-path circuit
     * sizes for decision-DNNF and OBDD (static and min-fill orders), exactness against
     * enumeration, and Bayesian-network marginals against variable elimination, loopy BP and
     * Gibbs.
     *
     * @param gridSides grid sizes for the path constraint
     * @param bnVariables variables of the random Bayesian networks
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> compilation(int[] gridSides, int bnVariables, long seed) {
        List<Row> rows = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed);
        // exactly-k against SemanticLoss's closed forms
        for (int[] nk : new int[][] {{10, 1}, {12, 3}}) {
            int n = nk[0];
            int k = nk[1];
            ConstraintInstances.Cnf f = ConstraintInstances.exactlyK(n, k);
            KnowledgeCompiler.Circuit c = KnowledgeCompiler.compileDnnf(f.variables(), f.clauses());
            double[] pos = new double[f.variables()];
            double[] neg = new double[f.variables()];
            Arrays.fill(pos, 1);
            Arrays.fill(neg, 1);
            float[] p = new float[n];
            for (int i = 0; i < n; i++) {
                p[i] = (float) (0.05 + 0.9 * r.nextDouble());
                pos[i] = p[i];
                neg[i] = 1 - p[i];
            }
            long t0 = System.nanoTime();
            double wmc = c.wmc(pos, neg);
            double q = ms(t0);
            Var pv = Var.of(Tensor.of(Shape.of(1, n), p));
            double closed = k == 1 ? Math.exp(-SemanticLoss.exactlyOne(pv).value().toArray()[0])
                : Math.exp(-SemanticLoss.atMostK(pv, k).value().toArray()[0])
                    - Math.exp(-SemanticLoss.atMostK(pv, k - 1).value().toArray()[0]);
            rows.add(new Row(f.name(), "DNNF models", c.modelCount(), binomial(n, k), c.size(), c.millis()));
            rows.add(new Row(f.name(), "DNNF WMC vs closed form", wmc, closed, c.size(), q));
        }
        // Latin squares: the closed form sums per-constraint losses; the circuit is exact
        for (int n : new int[] {3, 4}) {
            ConstraintInstances.Cnf f = ConstraintInstances.latinSquare(n, true);
            KnowledgeCompiler.Circuit c = KnowledgeCompiler.compileDnnf(f.variables(), f.clauses());
            float[] p = new float[n * n * n];
            double[] pos = new double[p.length];
            double[] neg = new double[p.length];
            for (int cell = 0; cell < n * n; cell++) {
                double s = 0;
                double[] w = new double[n];
                for (int v = 0; v < n; v++) {
                    w[v] = Math.exp(2 * r.nextGaussian());
                    s += w[v];
                }
                for (int v = 0; v < n; v++) {
                    p[cell * n + v] = (float) (w[v] / s);
                    pos[cell * n + v] = p[cell * n + v];
                    neg[cell * n + v] = 1 - p[cell * n + v];
                }
            }
            double exact = -Math.log(c.wmc(pos, neg));
            double closed = SemanticLoss.latinSquare(Var.of(Tensor.of(Shape.of(1, p.length), p)), n, true)
                .value().toArray()[0];
            rows.add(new Row(f.name(), "DNNF models", c.modelCount(), n == 3 ? 12 : 576, c.size(), c.millis()));
            rows.add(new Row(f.name(), "exact SL vs summed closed form", exact, closed, c.size(), 0));
        }
        // grid paths: DNNF against OBDD
        for (int side : gridSides) {
            ConstraintInstances.GridPath g = ConstraintInstances.gridPath(side);
            ConstraintInstances.Cnf f = g.cnf();
            double paths = binomial(2 * (side - 1), side - 1);
            KnowledgeCompiler.Circuit d = KnowledgeCompiler.compileDnnf(f.variables(), f.clauses());
            KnowledgeCompiler.Circuit bs = KnowledgeCompiler.compileObdd(f.variables(), f.clauses(), null);
            KnowledgeCompiler.Circuit bm = KnowledgeCompiler.compileObdd(f.variables(), f.clauses(),
                KnowledgeCompiler.minFillOrder(f.variables(), f.clauses()));
            String t = "grid-path-" + side + "x" + side;
            rows.add(new Row(t, "DECISION_DNNF", d.modelCount(), paths, d.size(), d.millis()));
            rows.add(new Row(t, "OBDD static", bs.modelCount(), paths, bs.size(), bs.millis()));
            rows.add(new Row(t, "OBDD min-fill", bm.modelCount(), paths, bm.size(), bm.millis()));
        }
        // exactness against enumeration on a tiny grid
        {
            ConstraintInstances.GridPath g = ConstraintInstances.gridPath(3);
            int m = g.edges().length;
            double[] pos = new double[m];
            double[] neg = new double[m];
            for (int i = 0; i < m; i++) {
                pos[i] = r.nextDouble();
                neg[i] = r.nextDouble();
            }
            long t0 = System.nanoTime();
            double brute = 0;
            for (int mask = 0; mask < 1 << m; mask++) {
                boolean[] on = new boolean[m];
                double w = 1;
                for (int i = 0; i < m; i++) {
                    on[i] = (mask >> i & 1) == 1;
                    w *= on[i] ? pos[i] : neg[i];
                }
                if (g.valid(on)) {
                    brute += w;
                }
            }
            double bt = ms(t0);
            KnowledgeCompiler.Circuit d = KnowledgeCompiler.compileDnnf(m, g.cnf().clauses());
            KnowledgeCompiler.Circuit b = KnowledgeCompiler.compileObdd(m, g.cnf().clauses(), null);
            rows.add(new Row("grid-path-3x3 WMC", "DNNF vs enumeration", d.wmc(pos, neg), brute, d.size(), bt));
            rows.add(new Row("grid-path-3x3 WMC", "OBDD vs enumeration", b.wmc(pos, neg), brute, b.size(), bt));
        }
        // Bayesian networks: WMC marginals against variable elimination, loopy BP and Gibbs
        for (double det : new double[] {0.0, 0.5}) {
            ConstraintInstances.BayesNet net = ConstraintInstances.bayesNet(bnVariables, 3, 4, det, seed + 7);
            KnowledgeCompiler.WeightedCnf enc = KnowledgeCompiler.encodeBayesNet(net.cardinality(), net.parents(),
                net.cpt());
            KnowledgeCompiler.Circuit c = KnowledgeCompiler.compileDnnf(enc.variables(), enc.clauses());
            int[] evidence = new int[bnVariables];
            Arrays.fill(evidence, -1);
            // observe the last variable at a value it can take: its most likely value a priori
            FactorGraph prior = ConstraintInstances.factorGraph(net);
            double[] last = prior.exactMarginal(bnVariables - 1);
            int ev = 0;
            for (int x = 1; x < last.length; x++) {
                if (last[x] > last[ev]) {
                    ev = x;
                }
            }
            evidence[bnVariables - 1] = ev;
            long t0 = System.nanoTime();
            double[][] wmc = KnowledgeCompiler.marginals(enc, c, evidence);
            double qWmc = ms(t0);
            FactorGraph fg = ConstraintInstances.factorGraph(net).observe(bnVariables - 1, ev);
            t0 = System.nanoTime();
            double[][] ve = fg.exactMarginals();
            double qVe = ms(t0);
            t0 = System.nanoTime();
            double[][] lbp = fg.loopyBeliefPropagation(100, 0.3);
            double qLbp = ms(t0);
            t0 = System.nanoTime();
            double[][] gibbs = fg.gibbs(2000, 200, seed);
            double qGibbs = ms(t0);
            String t = String.format("BN n=%d det=%.1f", bnVariables, det);
            rows.add(new Row(t, "WMC (DNNF) max error vs VE", FactorGraph.marginalError(wmc, ve), 0, c.size(),
                qWmc + c.millis()));
            rows.add(new Row(t, "WMC query only", qWmc, qVe, enc.clauses().size(), qWmc));
            rows.add(new Row(t, "VE (all marginals)", 0, 0, 0, qVe));
            rows.add(new Row(t, "loopy BP max error", FactorGraph.marginalError(lbp, ve), 0, 0, qLbp));
            rows.add(new Row(t, "Gibbs max error", FactorGraph.marginalError(gibbs, ve), 0, 0, qGibbs));
        }
        return rows;
    }

    /**
     * Semantic loss with a compiled constraint: an MLP maps noisy edge costs of a grid to the
     * cheapest monotone path (one sigmoid per edge); the plain model is trained with binary cross
     * entropy and the regularised one adds {@code -log WMC(p)} of the compiled path constraint,
     * whose gradient comes from one backward pass over the circuit. Reports the test valid-path
     * rate and the exact-path rate, plus the semantic loss computed by the circuit on test inputs.
     *
     * @param side nodes per grid side
     * @param train training examples
     * @param steps optimiser steps
     * @param weight the semantic-loss weight
     * @param seed the seed
     * @return rows for the plain MLP and the semantic-loss MLP
     */
    public static List<Row> semanticLossPaths(int side, int train, int steps, double weight, long seed) {
        ConstraintInstances.GridPath g = ConstraintInstances.gridPath(side);
        int m = g.edges().length;
        KnowledgeCompiler.Circuit circuit = KnowledgeCompiler.compileDnnf(m, g.cnf().clauses());
        SplittableRandom data = new SplittableRandom(seed);
        int test = 400;
        float[][] xs = new float[train + test][m];
        boolean[][] ys = new boolean[train + test][];
        for (int i = 0; i < train + test; i++) {
            double[] cost = new double[m];
            for (int e = 0; e < m; e++) {
                cost[e] = data.nextDouble();
                xs[i][e] = (float) cost[e];
            }
            ys[i] = g.cheapest(cost);
        }
        List<Row> rows = new ArrayList<>();
        for (boolean semantic : new boolean[] {false, true}) {
            long t0 = System.nanoTime();
            SplittableRandom r = new SplittableRandom(seed + 1);
            int hidden = 64;
            Var w1 = Var.param(Tensor.normal(Shape.of(m, hidden), 0, (float) Math.sqrt(2.0 / m), r));
            Var b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
            Var w2 = Var.param(Tensor.normal(Shape.of(hidden, m), 0, (float) Math.sqrt(1.0 / hidden), r));
            Var b2 = Var.param(Tensor.zeros(Shape.of(m)));
            List<Var> params = List.of(w1, b1, w2, b2);
            Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
            int batch = 32;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[batch * m];
                float[] y = new float[batch * m];
                for (int b = 0; b < batch; b++) {
                    int i = r.nextInt(train);
                    System.arraycopy(xs[i], 0, x, b * m, m);
                    for (int e = 0; e < m; e++) {
                        y[b * m + e] = ys[i][e] ? 1 : 0;
                    }
                }
                Var p = Var.of(Tensor.of(Shape.of(batch, m), x)).matmul(w1).add(b1).relu().matmul(w2).add(b2)
                    .sigmoid().clamp(1e-4f, 1 - 1e-4f);
                Tensor yt = Tensor.of(Shape.of(batch, m), y);
                Var loss = p.log().mul(yt).add(p.neg().add(1f).log().mul(yt.neg().add(1f))).neg().sum()
                    .mul(1f / batch);
                if (semantic) {
                    float[] pv = p.value().toArray();
                    float[] grad = new float[pv.length];
                    double[] pos = new double[m];
                    double[] neg = new double[m];
                    double[] dp = new double[m];
                    double[] dn = new double[m];
                    for (int b = 0; b < batch; b++) {
                        for (int e = 0; e < m; e++) {
                            pos[e] = pv[b * m + e];
                            neg[e] = 1 - pos[e];
                        }
                        double z = circuit.gradient(pos, neg, dp, dn);
                        for (int e = 0; e < m; e++) {
                            // d(-log z)/dp_e with pos = p and neg = 1 - p
                            grad[b * m + e] = (float) (-(dp[e] - dn[e]) / Math.max(z, 1e-300) * weight / batch);
                        }
                    }
                    loss = loss.add(p.mul(Tensor.of(Shape.of(batch, m), grad)).sum());
                }
                loss.backward();
                opt.step(params);
            }
            float[] tx = new float[test * m];
            for (int i = 0; i < test; i++) {
                System.arraycopy(xs[train + i], 0, tx, i * m, m);
            }
            float[] tp = Var.noGrad(() -> Var.of(Tensor.of(Shape.of(test, m), tx)).matmul(w1).add(b1).relu()
                .matmul(w2).add(b2).sigmoid()).value().toArray();
            int valid = 0;
            int exact = 0;
            double sl = 0;
            double[] pos = new double[m];
            double[] neg = new double[m];
            for (int i = 0; i < test; i++) {
                boolean[] on = new boolean[m];
                boolean same = true;
                for (int e = 0; e < m; e++) {
                    on[e] = tp[i * m + e] > 0.5f;
                    same &= on[e] == ys[train + i][e];
                    pos[e] = Math.min(1 - 1e-6, Math.max(1e-6, tp[i * m + e]));
                    neg[e] = 1 - pos[e];
                }
                valid += g.valid(on) ? 1 : 0;
                exact += same ? 1 : 0;
                sl += -Math.log(circuit.wmc(pos, neg)) / test;
            }
            String method = semantic ? "MLP + semantic loss (compiled)" : "MLP";
            rows.add(new Row("grid-path-" + side + " valid rate", method, valid / (double) test, exact / (double) test,
                sl, ms(t0)));
        }
        return rows;
    }

    // ---- 13.2.12 theorem proving ----------------------------------------------------------------

    /**
     * The training problems for the learned selector: small instances of every family.
     *
     * @param seed the seed
     * @return the problems
     */
    public static List<ResolutionProver.Problem> trainingProblems(long seed) {
        List<ResolutionProver.Problem> ps = new ArrayList<>(
            ResolutionProver.family(FamilyTreeGraph.generate(1, 0.3, seed + 100), 8));
        for (int n = 3; n <= 5; n++) {
            ps.add(ResolutionProver.chain(n, 6, seed + n));
            ps.add(ResolutionProver.equivalence(n - 1));
        }
        ps.add(ResolutionProver.equivalence(0));
        ps.add(ResolutionProver.group(0));
        ps.add(ResolutionProver.group(1));
        ps.add(ResolutionProver.pigeonhole(2));
        return ps;
    }

    /**
     * The held-out problems: larger instances of the same families plus unseen theorems.
     *
     * @param seed the seed
     * @return the problems
     */
    public static List<ResolutionProver.Problem> testProblems(long seed) {
        List<ResolutionProver.Problem> ps = new ArrayList<>(
            ResolutionProver.family(FamilyTreeGraph.generate(2, 0.3, seed + 200), 10));
        for (int n = 6; n <= 9; n++) {
            ps.add(ResolutionProver.chain(n, 16, seed + 10 * n));
        }
        for (int n = 5; n <= 7; n++) {
            ps.add(ResolutionProver.equivalence(n));
        }
        for (int t = 2; t < ResolutionProver.GROUP_THEOREMS.size(); t++) {
            ps.add(ResolutionProver.group(t));
        }
        ps.add(ResolutionProver.pigeonhole(3));
        ps.add(ResolutionProver.pigeonhole(4));
        return ps;
    }

    private static String familyOf(String name) {
        int dash = name.indexOf('-');
        return dash < 0 ? name : name.substring(0, dash);
    }

    /**
     * The proving table: proved share and given clauses per selection strategy on held-out
     * problems, with the learned selector trained on proofs of the training problems (found with
     * AGE_WEIGHT); per family rows, CDCL on the pigeonhole CNFs, and dILP learning the
     * grandparent rule instead of proving held-out grandparent facts.
     *
     * @param budget the given-clause budget
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> proving(int budget, long seed) {
        List<Row> rows = new ArrayList<>();
        ResolutionProver.Limits limits = ResolutionProver.Limits.of(budget);
        long t0 = System.nanoTime();
        ClauseSelector selector = new ClauseSelector(10);
        List<ClauseSelector.Example> examples = new ArrayList<>();
        int trainProved = 0;
        List<ResolutionProver.Problem> train = trainingProblems(seed);
        for (ResolutionProver.Problem p : train) {
            ResolutionProver.Result r = ResolutionProver.prove(p, ResolutionProver.Selection.AGE_WEIGHT,
                ResolutionProver.Limits.of(4 * budget), null);
            trainProved += r.proved() ? 1 : 0;
            examples.addAll(selector.examples(p, r));
        }
        selector.train(examples, 20, 0.05, 1e-4, seed);
        rows.add(new Row("training", "ENIGMA examples from AGE_WEIGHT proofs", trainProved, train.size(),
            examples.size(), ms(t0)));
        List<ResolutionProver.Problem> test = testProblems(seed);
        ResolutionProver.Selection[] sels = ResolutionProver.Selection.values();
        ResolutionProver.Result[][] res = new ResolutionProver.Result[sels.length][test.size()];
        for (int s = 0; s < sels.length; s++) {
            for (int i = 0; i < test.size(); i++) {
                res[s][i] = ResolutionProver.prove(test.get(i), sels[s], limits, selector);
            }
        }
        List<String> families = test.stream().map(p -> familyOf(p.name())).distinct().toList();
        for (int s = 0; s < sels.length; s++) {
            int proved = 0;
            double ms = 0;
            // given clauses on the problems every strategy proves
            double given = 0;
            int common = 0;
            for (int i = 0; i < test.size(); i++) {
                proved += res[s][i].proved() ? 1 : 0;
                ms += res[s][i].millis();
                boolean all = true;
                for (ResolutionProver.Result[] other : res) {
                    all &= other[i].proved();
                }
                if (all) {
                    given += res[s][i].given();
                    common++;
                }
            }
            rows.add(new Row("held-out (" + test.size() + ")", sels[s].name(), proved, test.size(),
                common == 0 ? 0 : given / common, ms));
            for (String fam : families) {
                int pf = 0;
                int nf = 0;
                double gf = 0;
                for (int i = 0; i < test.size(); i++) {
                    if (familyOf(test.get(i).name()).equals(fam)) {
                        nf++;
                        pf += res[s][i].proved() ? 1 : 0;
                        gf += res[s][i].given();
                    }
                }
                rows.add(new Row(fam, sels[s].name(), pf, nf, gf / nf, 0));
            }
        }
        // CDCL on the same pigeonhole formulas
        for (int h : new int[] {3, 4}) {
            ConstraintInstances.Cnf f = ConstraintInstances.pigeonhole(h);
            long t1 = System.nanoTime();
            CdclSolver s = CdclSolver.of(f.variables(), f.clauses(), CdclSolver.Options.defaults());
            CdclSolver.Status st = s.solve();
            rows.add(new Row("php-" + h, "CDCL (conflicts)", st == CdclSolver.Status.UNSAT ? 1 : 0, 1,
                s.stats().conflicts(), ms(t1)));
        }
        rows.add(dilpGrandparent(seed));
        return rows;
    }

    private static Tensor relation(List<int[]> triples, int n, int rel, int inverseRel) {
        float[] m = new float[n * n];
        for (int[] t : triples) {
            if (t[1] == rel) {
                m[t[0] * n + t[2]] = 1;
            } else if (t[1] == inverseRel) {
                m[t[2] * n + t[0]] = 1;
            }
        }
        return Tensor.of(Shape.of(n, n), m);
    }

    /** dILP learns grandparent from parent on a training world, then runs on the test world. */
    private static Row dilpGrandparent(long seed) {
        long t0 = System.nanoTime();
        FamilyTreeGraph.Data trainWorld = FamilyTreeGraph.generate(2, 0.0, seed + 300);
        int n = trainWorld.entities();
        List<Tensor> bg = List.of(relation(trainWorld.all(), n, FamilyTreeGraph.PARENT, FamilyTreeGraph.CHILD));
        Tensor target = relation(trainWorld.all(), n, FamilyTreeGraph.GRANDPARENT, -1);
        DifferentiableIlp dilp = new DifferentiableIlp(1, 2, seed);
        Optimizer opt = Optimizer.adam(0.1f);
        for (int s = 0; s < 150; s++) {
            dilp.loss(bg, target).backward();
            opt.step(dilp.parameters());
        }
        FamilyTreeGraph.Data testWorld = FamilyTreeGraph.generate(2, 0.3, seed + 200);
        int m = testWorld.entities();
        boolean[][] derived = dilp.runProgram(List.of(relation(testWorld.train(), m, FamilyTreeGraph.PARENT,
            FamilyTreeGraph.CHILD)));
        int hit = 0;
        int total = 0;
        for (int[] t : testWorld.test()) {
            if (t[1] == FamilyTreeGraph.GRANDPARENT) {
                total++;
                hit += derived[t[0]][t[2]] ? 1 : 0;
            }
        }
        return new Row("family grandparent", "dILP (" + String.join("; ", dilp.program("parent")) + ")", hit, total,
            0, ms(t0));
    }

    // ---- 13.2.13 Markov logic -------------------------------------------------------------------

    private static final int SMOKES = 0;
    private static final int CANCER = 1;
    private static final int FRIENDS = 2;

    /** The friends-and-smokers MLN: propagation along friendship, smoking causes cancer, two priors. */
    private static MarkovLogic smokersMln(double[] weights) {
        List<MarkovLogic.Formula> fs = List.of(
            new MarkovLogic.Formula("Smokes(x) & Friends(x,y) => Smokes(y)", List.of(
                new MarkovLogic.Literal(false, SMOKES, 0), new MarkovLogic.Literal(false, FRIENDS, 0, 1),
                new MarkovLogic.Literal(true, SMOKES, 1)), false),
            new MarkovLogic.Formula("Smokes(x) => Cancer(x)", List.of(new MarkovLogic.Literal(false, SMOKES, 0),
                new MarkovLogic.Literal(true, CANCER, 0)), false),
            new MarkovLogic.Formula("!Smokes(x)", List.of(new MarkovLogic.Literal(false, SMOKES, 0)), false),
            new MarkovLogic.Formula("!Cancer(x)", List.of(new MarkovLogic.Literal(false, CANCER, 0)), false));
        return new MarkovLogic(new int[] {1, 1, 2}, fs, weights, 10);
    }

    /**
     * A planted friends-and-smokers world on a community graph: seed smokers, smoking spreads to
     * friends for two rounds, smokers get cancer with probability 0.6 and others with 0.05.
     */
    private static boolean[] smokersWorld(MarkovLogic mln, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(Math.max(1, n / 6), 6, 0.5, 0.01, 0, seed);
        int size = g.nodes();
        boolean[] v = new boolean[mln.atoms(n)];
        boolean[] smokes = new boolean[n];
        for (int x = 0; x < n; x++) {
            smokes[x] = r.nextDouble() < 0.15;
        }
        for (int round = 0; round < 2; round++) {
            boolean[] next = smokes.clone();
            for (int x = 0; x < n; x++) {
                for (int y = 0; y < n; y++) {
                    if (x != y && x < size && y < size && g.adjacency()[x * size + y] > 0 && smokes[x]
                            && r.nextDouble() < 0.4) {
                        next[y] = true;
                    }
                }
            }
            smokes = next;
        }
        for (int x = 0; x < n; x++) {
            v[mln.atom(n, SMOKES, x)] = smokes[x];
            v[mln.atom(n, CANCER, x)] = r.nextDouble() < (smokes[x] ? 0.6 : 0.05);
            for (int y = 0; y < n; y++) {
                v[mln.atom(n, FRIENDS, x, y)] = x != y && x < size && y < size && g.adjacency()[x * size + y] > 0;
            }
        }
        return v;
    }

    private static boolean[] hideQueries(MarkovLogic mln, int n, double hidden, long seed, int... predicates) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] evidence = new boolean[mln.atoms(n)];
        Arrays.fill(evidence, true);
        for (int p : predicates) {
            int start = mln.atom(n, p, p == FRIENDS ? new int[] {0, 0} : new int[] {0});
            int count = mln.atom(n, p, n - 1, n - 1) - start + 1;
            for (int a = start; a < start + count; a++) {
                evidence[a] = r.nextDouble() >= hidden;
            }
        }
        return evidence;
    }

    /** AUROC of scores against labels (ties count half). */
    static double auroc(double[] score, boolean[] label) {
        double pos = 0;
        double neg = 0;
        double acc = 0;
        for (int i = 0; i < score.length; i++) {
            for (int j = 0; j < score.length; j++) {
                if (label[i] && !label[j]) {
                    acc += score[i] > score[j] ? 1 : score[i] == score[j] ? 0.5 : 0;
                }
            }
            if (label[i]) {
                pos++;
            } else {
                neg++;
            }
        }
        return pos == 0 || neg == 0 ? 0.5 : acc / (pos * neg);
    }

    private static double[] queryScores(double[] marg, boolean[] evidence, boolean[] truth, boolean[] labels,
            double[] cll) {
        int k = 0;
        double[] s = new double[labels.length];
        cll[0] = 0;
        for (int a = 0; a < evidence.length; a++) {
            if (!evidence[a]) {
                s[k] = marg[a];
                labels[k] = truth[a];
                double p = Math.min(0.99, Math.max(0.01, marg[a]));
                cll[0] += Math.log(truth[a] ? p : 1 - p);
                k++;
            }
        }
        cll[0] /= Math.max(1, k);
        return s;
    }

    private static double[] factorGraphMarginals(MarkovLogic mln, MarkovLogic.Ground g, boolean lbp, long seed) {
        int[] varOf = new int[g.values().length];
        FactorGraph fg = mln.toFactorGraph(g, varOf);
        double[][] m = lbp ? fg.loopyBeliefPropagation(200, 0.5) : fg.gibbs(500, 100, seed);
        double[] out = new double[varOf.length];
        for (int a = 0; a < out.length; a++) {
            out[a] = varOf[a] < 0 ? (g.values()[a] ? 1 : 0) : m[varOf[a]][1];
        }
        return out;
    }

    private static int countHidden(boolean[] evidence) {
        int k = 0;
        for (boolean e : evidence) {
            k += e ? 0 : 1;
        }
        return k;
    }

    /**
     * The Markov-logic table: weight learning (pseudo-likelihood against diagonal Newton) on a
     * planted friends-and-smokers world, query AUROC and conditional log-likelihood of hidden
     * Smokes/Cancer atoms on a fresh world for MC-SAT, Gibbs, the factor-graph baselines and a
     * logistic regression on relational features, exactness on a tiny world, and a family world
     * with hard ancestor rules where samplers start from random states.
     *
     * @param people persons per world
     * @param hidden the share of query atoms hidden
     * @param samples sampler steps kept
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> markovLogic(int people, double hidden, int samples, long seed) {
        List<Row> rows = new ArrayList<>();
        MarkovLogic template = smokersMln(new double[4]);
        boolean[] trainWorld = smokersWorld(template, people, seed);
        // learning
        double[][] learned = new double[2][];
        for (MarkovLogic.Learning l : MarkovLogic.Learning.values()) {
            MarkovLogic m = smokersMln(new double[4]);
            long t0 = System.nanoTime();
            learned[l.ordinal()] = m.learn(l, people, trainWorld, new int[] {FRIENDS},
                l == MarkovLogic.Learning.PSEUDO_LIKELIHOOD ? 100 : 15, 100, 2.0, seed);
            double[] w = learned[l.ordinal()];
            rows.add(new Row("smokers learn " + String.format("w=[%.2f, %.2f, %.2f, %.2f]", w[0], w[1], w[2], w[3]),
                l.name(), w[0] > 0 && w[1] > 0 ? 1 : 0, 1, w.length, ms(t0)));
        }
        // inference on a held-out world with pseudo-likelihood weights
        MarkovLogic mln = smokersMln(learned[0]);
        boolean[] testWorld = smokersWorld(mln, people, seed + 1);
        boolean[] evidence = hideQueries(mln, people, hidden, seed + 2, SMOKES, CANCER);
        MarkovLogic.Ground g = mln.ground(people, testWorld, evidence);
        int q = countHidden(evidence);
        boolean[] labels = new boolean[q];
        double[] cll = new double[1];
        String task = "smokers n=" + people + " hidden=" + q;
        for (MarkovLogic.Inference inf : new MarkovLogic.Inference[] {MarkovLogic.Inference.MC_SAT,
            MarkovLogic.Inference.GIBBS}) {
            long t0 = System.nanoTime();
            double[] marg = mln.marginals(g, inf, samples, samples / 5, false, seed);
            double auc = auroc(queryScores(marg, evidence, testWorld, labels, cll), labels);
            rows.add(new Row(task, inf.name() + " AUROC (ref = CLL)", auc, cll[0], g.clauses().size(), ms(t0)));
        }
        for (boolean lbp : new boolean[] {true, false}) {
            long t0 = System.nanoTime();
            double[] marg = factorGraphMarginals(mln, g, lbp, seed);
            double auc = auroc(queryScores(marg, evidence, testWorld, labels, cll), labels);
            rows.add(new Row(task, (lbp ? "FactorGraph loopy BP" : "FactorGraph Gibbs") + " AUROC", auc, cll[0],
                g.clauses().size(), ms(t0)));
        }
        {
            long t0 = System.nanoTime();
            double[] marg = logisticBaseline(mln, people, trainWorld, testWorld, evidence, hidden, seed);
            double auc = auroc(queryScores(marg, evidence, testWorld, labels, cll), labels);
            rows.add(new Row(task, "logistic regression on relational features AUROC", auc, cll[0], 0, ms(t0)));
        }
        // exactness on a tiny world: every Smokes and Cancer atom hidden
        {
            int n = 5;
            boolean[] tiny = smokersWorld(mln, n, seed + 3);
            boolean[] ev = hideQueries(mln, n, 1.0, seed, SMOKES, CANCER);
            MarkovLogic.Ground tg = mln.ground(n, tiny, ev);
            double[] exact = mln.marginals(tg, MarkovLogic.Inference.EXACT, 0, 0, false, seed);
            String t = "smokers n=5 all hidden (max |error| vs exact)";
            for (MarkovLogic.Inference inf : new MarkovLogic.Inference[] {MarkovLogic.Inference.MC_SAT,
                MarkovLogic.Inference.GIBBS}) {
                long t0 = System.nanoTime();
                double[] m = mln.marginals(tg, inf, 4 * samples, samples / 5, false, seed);
                rows.add(new Row(t, inf.name(), maxError(m, exact), 0, tg.clauses().size(), ms(t0)));
            }
            long t0 = System.nanoTime();
            rows.add(new Row(t, "FactorGraph loopy BP", maxError(factorGraphMarginals(mln, tg, true, seed), exact), 0,
                tg.clauses().size(), ms(t0)));
        }
        rows.addAll(hardRules(0.9, samples, seed));
        return rows;
    }

    private static double maxError(double[] a, double[] b) {
        double e = 0;
        for (int i = 0; i < a.length; i++) {
            e = Math.max(e, Math.abs(a[i] - b[i]));
        }
        return e;
    }

    /**
     * Logistic regression per query predicate on hand-built relational features: own observed
     * other attribute (and whether it is observed), observed smoking friends, observed friends.
     */
    private static double[] logisticBaseline(MarkovLogic mln, int n, boolean[] trainWorld, boolean[] testWorld,
            boolean[] testEvidence, double hidden, long seed) {
        boolean[] trainEvidence = hideQueries(mln, n, hidden, seed + 5, SMOKES, CANCER);
        double[][] w = new double[2][5];
        for (int p : new int[] {SMOKES, CANCER}) {
            for (int epoch = 0; epoch < 300; epoch++) {
                for (int x = 0; x < n; x++) {
                    int a = mln.atom(n, p, x);
                    double[] f = relationalFeatures(mln, n, trainWorld, trainEvidence, p, x);
                    double z = 0;
                    for (int k = 0; k < f.length; k++) {
                        z += w[p][k] * f[k];
                    }
                    double err = (trainWorld[a] ? 1 : 0) - 1 / (1 + Math.exp(-z));
                    for (int k = 0; k < f.length; k++) {
                        w[p][k] += 0.05 * (err * f[k] - 1e-3 * w[p][k]);
                    }
                }
            }
        }
        double[] out = new double[testWorld.length];
        for (int a = 0; a < out.length; a++) {
            out[a] = testWorld[a] ? 1 : 0;
        }
        for (int p : new int[] {SMOKES, CANCER}) {
            for (int x = 0; x < n; x++) {
                int a = mln.atom(n, p, x);
                if (!testEvidence[a]) {
                    double[] f = relationalFeatures(mln, n, testWorld, testEvidence, p, x);
                    double z = 0;
                    for (int k = 0; k < f.length; k++) {
                        z += w[p][k] * f[k];
                    }
                    out[a] = 1 / (1 + Math.exp(-z));
                }
            }
        }
        return out;
    }

    private static double[] relationalFeatures(MarkovLogic mln, int n, boolean[] world, boolean[] evidence, int p,
            int x) {
        int other = mln.atom(n, p == SMOKES ? CANCER : SMOKES, x);
        double smokingFriends = 0;
        double friends = 0;
        for (int y = 0; y < n; y++) {
            if (world[mln.atom(n, FRIENDS, x, y)]) {
                friends++;
                int s = mln.atom(n, SMOKES, y);
                smokingFriends += evidence[s] && world[s] ? 1 : 0;
            }
        }
        return new double[] {1, evidence[other] && world[other] ? 1 : 0, evidence[other] ? 1 : 0,
            smokingFriends, friends / 4};
    }

    /** A family world: parent observed, ancestor partly hidden, ancestor rules hard. */
    private static List<Row> hardRules(double hidden, int samples, long seed) {
        FamilyTreeGraph.Data fam = FamilyTreeGraph.generate(1, 0.0, seed + 9);
        int n = fam.entities();
        int parent = 0;
        int anc = 1;
        List<MarkovLogic.Formula> fs = List.of(
            new MarkovLogic.Formula("parent(x,y) => anc(x,y)", List.of(new MarkovLogic.Literal(false, parent, 0, 1),
                new MarkovLogic.Literal(true, anc, 0, 1)), true),
            new MarkovLogic.Formula("anc(x,y) & anc(y,z) => anc(x,z)", List.of(
                new MarkovLogic.Literal(false, anc, 0, 1), new MarkovLogic.Literal(false, anc, 1, 2),
                new MarkovLogic.Literal(true, anc, 0, 2)), true),
            new MarkovLogic.Formula("!anc(x,y)", List.of(new MarkovLogic.Literal(false, anc, 0, 1)), false));
        MarkovLogic mln = new MarkovLogic(new int[] {2, 2}, fs, new double[] {0, 0, 1.0}, 10);
        boolean[] world = new boolean[mln.atoms(n)];
        boolean[][] closure = new boolean[n][n];
        for (int[] t : fam.all()) {
            if (t[1] == FamilyTreeGraph.PARENT) {
                world[mln.atom(n, parent, t[0], t[2])] = true;
                closure[t[0]][t[2]] = true;
            }
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    closure[i][j] |= closure[i][k] && closure[k][j];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                world[mln.atom(n, anc, i, j)] = closure[i][j];
            }
        }
        SplittableRandom r = new SplittableRandom(seed + 11);
        boolean[] evidence = new boolean[world.length];
        Arrays.fill(evidence, true);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                evidence[mln.atom(n, anc, i, j)] = r.nextDouble() >= hidden;
            }
        }
        MarkovLogic.Ground g = mln.ground(n, world, evidence);
        int q = countHidden(evidence);
        boolean[] labels = new boolean[q];
        double[] cll = new double[1];
        List<Row> rows = new ArrayList<>();
        String task = "family hard rules n=" + n + " hidden=" + q;
        for (MarkovLogic.Inference inf : new MarkovLogic.Inference[] {MarkovLogic.Inference.MC_SAT,
            MarkovLogic.Inference.GIBBS}) {
            long t0 = System.nanoTime();
            double[] marg = mln.marginals(g, inf, samples, samples / 5, true, seed);
            double auc = auroc(queryScores(marg, evidence, world, labels, cll), labels);
            rows.add(new Row(task, inf.name() + " (random start) AUROC", auc, cll[0], g.clauses().size(), ms(t0)));
        }
        long t0 = System.nanoTime();
        double[] marg = factorGraphMarginals(mln, g, true, seed);
        double auc = auroc(queryScores(marg, evidence, world, labels, cll), labels);
        rows.add(new Row(task, "FactorGraph loopy BP AUROC", auc, cll[0], g.clauses().size(), ms(t0)));
        t0 = System.nanoTime();
        boolean[] map = mln.map(g, 20_000, seed);
        int wrong = 0;
        for (int a = 0; a < world.length; a++) {
            wrong += map[a] != world[a] ? 1 : 0;
        }
        rows.add(new Row(task, "MaxWalkSAT MAP wrong atoms", wrong, 0, g.clauses().size(), ms(t0)));
        return rows;
    }

    // ---- 13.2.14 answer set programming ---------------------------------------------------------

    /** Enumerates up to {@code max} models of a CNF projected on its first {@code shown} variables. */
    private static int enumerate(int vars, List<int[]> clauses, int shown, int max) {
        CdclSolver s = CdclSolver.of(vars, clauses, CdclSolver.Options.defaults());
        int count = 0;
        while (count < max && s.solve() == CdclSolver.Status.SAT) {
            count++;
            boolean[] m = s.model();
            int[] block = new int[shown];
            for (int v = 0; v < shown; v++) {
                block[v] = m[v] ? -(v + 1) : v + 1;
            }
            if (!s.addClause(block)) {
                break;
            }
        }
        return count;
    }

    /**
     * The colouring program.
     *
     * @param n nodes
     * @param edges undirected edges
     * @param k colours
     * @return the program
     */
    public static AnswerSetSolver.Program colouringProgram(int n, List<int[]> edges, int k) {
        StringBuilder sb = new StringBuilder();
        for (int v = 0; v < n; v++) {
            sb.append("node(").append(v).append(").");
        }
        for (int c = 0; c < k; c++) {
            sb.append("col(c").append(c).append(").");
        }
        for (int[] e : edges) {
            sb.append("edge(").append(e[0]).append(',').append(e[1]).append(").");
        }
        sb.append("1 { color(X,C) : col(C) } 1 :- node(X).");
        sb.append(":- edge(X,Y), color(X,C), color(Y,C).");
        return AnswerSetSolver.Program.parse(sb.toString());
    }

    /**
     * The N-queens program.
     *
     * @param n the board size
     * @return the program
     */
    public static AnswerSetSolver.Program queensProgram(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= n; i++) {
            sb.append("row(").append(i).append("). column(").append(i).append(").");
        }
        sb.append("1 { q(R,C) : column(C) } 1 :- row(R).");
        sb.append(":- q(R,C), q(R2,C), R < R2.");
        sb.append(":- q(R,C), q(R2,C2), R < R2, R-C == R2-C2.");
        sb.append(":- q(R,C), q(R2,C2), R < R2, R+C == R2+C2.");
        return AnswerSetSolver.Program.parse(sb.toString());
    }

    /**
     * The Hamiltonian-cycle program; {@code reached} is positively recursive, so the program is
     * not tight.
     *
     * @param n nodes
     * @param arcs directed arcs
     * @return the program
     */
    public static AnswerSetSolver.Program hamiltonProgram(int n, List<int[]> arcs) {
        StringBuilder sb = new StringBuilder("start(0).");
        for (int v = 0; v < n; v++) {
            sb.append("node(").append(v).append(").");
        }
        for (int[] e : arcs) {
            sb.append("edge(").append(e[0]).append(',').append(e[1]).append(").");
        }
        sb.append("{ in(X,Y) } :- edge(X,Y).");
        sb.append(":- in(X,Y), in(X,Z), Y < Z.");
        sb.append(":- in(X,Y), in(Z,Y), X < Z.");
        sb.append("hasout(X) :- in(X,Y). hasin(Y) :- in(X,Y).");
        sb.append(":- node(X), not hasout(X). :- node(X), not hasin(X).");
        sb.append("reached(Y) :- start(X), in(X,Y).");
        sb.append("reached(Y) :- reached(X), in(X,Y).");
        sb.append(":- node(X), not reached(X).");
        return AnswerSetSolver.Program.parse(sb.toString());
    }

    private static boolean hamiltonian(Set<String> model, int n) {
        int[] succ = new int[n];
        Arrays.fill(succ, -1);
        for (String a : model) {
            if (a.startsWith("in(")) {
                String[] xy = a.substring(3, a.length() - 1).split(",");
                succ[Integer.parseInt(xy[0])] = Integer.parseInt(xy[1]);
            }
        }
        int v = 0;
        for (int k = 0; k < n; k++) {
            v = succ[v];
            if (v < 0 || v == 0 && k < n - 1) {
                return false;
            }
        }
        return v == 0;
    }

    /** A random digraph with a planted Hamiltonian cycle and {@code extra} more arcs per node. */
    private static List<int[]> hamiltonGraph(int n, int extra, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        Set<Long> seen = new java.util.HashSet<>();
        List<int[]> arcs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int a = perm[i];
            int b = perm[(i + 1) % n];
            seen.add((long) a * n + b);
            arcs.add(new int[] {a, b});
        }
        for (int a = 0; a < n; a++) {
            for (int k = 0; k < extra; k++) {
                int b = r.nextInt(n);
                if (b != a && seen.add((long) a * n + b)) {
                    arcs.add(new int[] {a, b});
                }
            }
        }
        return arcs;
    }

    /**
     * The move-based Blocksworld program with horizon {@code steps}.
     *
     * @param initBelow per block the block below, -1 for the table
     * @param goalBelow the goal configuration
     * @param steps the horizon
     * @return the program
     */
    public static AnswerSetSolver.Program blocksProgram(int[] initBelow, int[] goalBelow, int steps) {
        StringBuilder sb = new StringBuilder("loc(table). loc(X) :- block(X).");
        for (int b = 0; b < initBelow.length; b++) {
            sb.append("block(b").append(b).append(").");
            sb.append("on(b").append(b).append(',').append(initBelow[b] < 0 ? "table" : "b" + initBelow[b])
                .append(",0).");
            sb.append("goal(b").append(b).append(',').append(goalBelow[b] < 0 ? "table" : "b" + goalBelow[b])
                .append(").");
        }
        for (int t = 0; t < steps; t++) {
            sb.append("step(").append(t).append("). next(").append(t).append(',').append(t + 1).append(").");
        }
        sb.append("last(").append(steps).append(").");
        sb.append("{ move(B,L,T) } :- block(B), loc(L), step(T), B != L.");
        sb.append(":- move(B,L,T), move(B2,L2,T), B < B2.");
        sb.append(":- move(B,L,T), move(B,L2,T), L < L2.");
        sb.append("moved(B,T) :- move(B,L,T).");
        sb.append("on(B,L,T1) :- move(B,L,T), next(T,T1).");
        sb.append("on(B,L,T1) :- on(B,L,T), next(T,T1), not moved(B,T).");
        sb.append(":- move(B,L,T), on(B2,B,T).");
        sb.append(":- move(B,L,T), on(B2,L,T), block(L).");
        sb.append(":- move(B,L,T), on(B,L,T).");
        sb.append(":- goal(B,L), last(T), not on(B,L,T).");
        return AnswerSetSolver.Program.parse(sb.toString());
    }

    private static int[] towers(int n, SplittableRandom r) {
        int[] below = new int[n];
        List<Integer> tops = new ArrayList<>();
        for (int x = 0; x < n; x++) {
            int k = r.nextInt(tops.size() + 1);
            if (k == tops.size()) {
                below[x] = -1;
                tops.add(x);
            } else {
                below[x] = tops.get(k);
                tops.set(k, x);
            }
        }
        return below;
    }

    /**
     * The ASP table: Hamiltonian cycles with plain completion against loop formulas, colouring and
     * N-queens against MAC and a direct CNF (first model and full enumeration), and Blocksworld
     * planning with the shortest horizon against sequential SATPlan on the STRIPS version.
     *
     * @param hamiltonInstances Hamiltonian-cycle graphs
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> answerSets(int hamiltonInstances, long seed) {
        List<Row> rows = new ArrayList<>();
        // Hamiltonian cycles: plain completion admits disjoint sub-cycles
        for (boolean loops : new boolean[] {false, true}) {
            int validFirst = 0;
            int validAll = 0;
            int total = 0;
            int loopClauses = 0;
            double ms = 0;
            double ground = 0;
            for (int i = 0; i < hamiltonInstances; i++) {
                int n = 10;
                AnswerSetSolver.Result res = AnswerSetSolver.solve(hamiltonProgram(n, hamiltonGraph(n, 2, seed + i)),
                    20, loops);
                validFirst += !res.models().isEmpty() && hamiltonian(res.models().get(0), n) ? 1 : 0;
                for (Set<String> m : res.models()) {
                    validAll += hamiltonian(m, n) ? 1 : 0;
                    total++;
                }
                loopClauses += res.loopFormulas();
                ms += res.totalMillis();
                ground += res.groundRules();
            }
            rows.add(new Row("hamilton n=10 x" + hamiltonInstances + " (valid first models; ref = instances)",
                loops ? "ASP + loop formulas" : "plain completion", validFirst, hamiltonInstances,
                ground / hamiltonInstances, ms));
            rows.add(new Row("hamilton n=10 enumerated (valid of up to 20 per instance; size = loop clauses)",
                loops ? "ASP + loop formulas" : "plain completion", validAll, total, loopClauses, ms));
        }
        // colouring: ASP against MAC and a direct CNF
        {
            int n = 40;
            int k = 3;
            List<int[]> edges = ConstraintInstances.plantedGraph(n, k, 4.0, seed);
            AnswerSetSolver.Result first = AnswerSetSolver.solve(colouringProgram(n, edges, k), 1, true);
            rows.add(new Row("colouring n=40 k=3 first model", "ASP", first.models().size(), 1, first.clauses(),
                first.firstMillis()));
            long t0 = System.nanoTime();
            ConstraintInstances.Cnf cnf = ConstraintInstances.colouringCnf(n, edges, k);
            boolean sat = CdclSolver.solve(cnf.variables(), cnf.clauses(), CdclSolver.Options.defaults()) != null;
            rows.add(new Row("colouring n=40 k=3 first model", "direct CNF + CDCL", sat ? 1 : 0, 1,
                cnf.clauses().size(), ms(t0)));
            ConstraintSolver.Result mac = new ConstraintSolver(ConstraintInstances.colouring(n, edges, k),
                ConstraintSolver.Algorithm.MAC, ConstraintSolver.VariableOrder.MRV_DEG, true, true, 100_000).solve();
            rows.add(new Row("colouring n=40 k=3 first model", "MAC", mac.solved() ? 1 : 0, 1, mac.nodes(),
                mac.millis()));
            int m = 12;
            List<int[]> small = ConstraintInstances.plantedGraph(m, k, 3.0, seed + 1);
            AnswerSetSolver.Result all = AnswerSetSolver.solve(colouringProgram(m, small, k), 100_000, true);
            t0 = System.nanoTime();
            ConstraintInstances.Cnf sc = ConstraintInstances.colouringCnf(m, small, k);
            int count = enumerate(sc.variables(), sc.clauses(), m * k, 100_000);
            double direct = ms(t0);
            rows.add(new Row("colouring n=12 all models (ref = direct CNF count)", "ASP", all.models().size(), count,
                all.groundRules(), all.totalMillis()));
            rows.add(new Row("colouring n=12 all models", "direct CNF + CDCL", count, count, sc.clauses().size(),
                direct));
        }
        // N-queens
        {
            int n = 8;
            AnswerSetSolver.Result all = AnswerSetSolver.solve(queensProgram(n), 1000, true);
            Csp csp = ConstraintInstances.nQueens(n);
            long t0 = System.nanoTime();
            int count = enumerate(csp.cnfVariables(), csp.toCnf(), csp.cnfVariables(), 1000);
            double direct = ms(t0);
            rows.add(new Row("queens-8 first model", "ASP", 1, 1, all.groundRules(), all.firstMillis()));
            rows.add(new Row("queens-8 all models (ref = 92)", "ASP", all.models().size(), 92, all.clauses(),
                all.totalMillis()));
            rows.add(new Row("queens-8 all models (ref = 92)", "direct CNF + CDCL", count, 92, csp.toCnf().size(),
                direct));
            ConstraintSolver.Result mac = new ConstraintSolver(csp, ConstraintSolver.Algorithm.MAC,
                ConstraintSolver.VariableOrder.MRV, true, false, 100_000).solve();
            rows.add(new Row("queens-8 first model", "MAC", mac.solved() ? 1 : 0, 1, mac.nodes(), mac.millis()));
        }
        // Blocksworld planning: shortest horizon by ASP against SATPlan on the 4-operator STRIPS task
        {
            SplittableRandom r = new SplittableRandom(seed);
            int blocks = 5;
            int[] init = towers(blocks, r);
            int[] goal = towers(blocks, r);
            long t0 = System.nanoTime();
            int horizon = -1;
            int groundRules = 0;
            for (int h = 0; h <= 12 && horizon < 0; h++) {
                AnswerSetSolver.Result res = AnswerSetSolver.solve(blocksProgram(init, goal, h), 1, true);
                groundRules = res.groundRules();
                if (!res.models().isEmpty()) {
                    horizon = h;
                }
            }
            rows.add(new Row("blocks-5 shortest plan (moves; ref = SATPlan length / 2)", "ASP", horizon, 0,
                groundRules, ms(t0)));
            StripsTask task = PlanningDomains.blocksworld(init, goal, "asp-blocks");
            SatPlanner.Result sp = SatPlanner.plan(task, SatPlanner.Encoding.SEQUENTIAL, false, 30, 1_000_000);
            Row last = rows.remove(rows.size() - 1);
            rows.add(new Row(last.task(), last.method(), last.value(), sp.plan().size() / 2.0, last.size(),
                last.millis()));
            rows.add(new Row("blocks-5 shortest plan (STRIPS actions)", "SATPlan sequential", sp.plan().size(),
                sp.makespan(), sp.largestCnf(), sp.millis()));
        }
        return rows;
    }
}
