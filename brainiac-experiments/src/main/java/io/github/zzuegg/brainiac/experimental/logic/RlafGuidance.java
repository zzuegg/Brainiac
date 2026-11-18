package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.graph.NeuroSat;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * RLAF: one-shot GNN guidance of a SAT solver trained by reinforcement learning from algorithm
 * feedback (Tönshoff &amp; Grohe, 2025, <a href="https://arxiv.org/abs/2505.16053">arXiv:2505.16053</a>).
 *
 * <p>A message-passing network on the literal-clause graph (wave 7's {@link NeuroSat} as the
 * encoder) reads the formula once and outputs, per variable, the mean {@code mu} of a log-normal
 * branching weight ({@code sigma = 0.1}, as in the paper) and the logit of a Bernoulli polarity.
 * Sampled weights multiply the variable's VSIDS score in {@link CdclSolver} (bumps scaled by the
 * weight) and polarities set the first value tried (phase saving stays on). Training is group
 * relative policy optimisation without a critic: {@code M} parameterisations are sampled per
 * formula, each is scored by the solver's cost (decisions; reward {@code -cost}), advantages are
 * standardised within the group, and the policy gradient {@code -sum A log pi} is followed (one
 * step per group, so no ratio clipping is needed). At test time the mode is used.
 *
 * <p>{@link #trainSupervised} is the paper's supervised baseline (backbone prediction): a
 * literal-level classifier on the same encoder, trained with cross-entropy on backbone labels of
 * satisfiable formulas, mapped to weights {@code w = 1 + alpha p} and polarities by comparing the
 * two literal probabilities.
 */
public final class RlafGuidance {

    /** Guidance for one formula. */
    public record Guidance(double[] weights, boolean[] polarities) {
    }

    private static final double SIGMA = 0.1;

    private final NeuroSat encoder;
    private final int width;
    private final int rounds;
    private final Var wMu;
    private final Var bMu;
    private final Var wPol;
    private final Var bPol;
    private final Var wLit;
    private final Var bLit;
    private double alpha = 1;

    /**
     * Creates an untrained model.
     *
     * @param width the encoder width
     * @param rounds message-passing rounds
     * @param seed the seed
     */
    public RlafGuidance(int width, int rounds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.encoder = new NeuroSat(width, r);
        this.width = width;
        this.rounds = rounds;
        float s = (float) (0.1 / Math.sqrt(2 * width));
        wMu = Var.param(Tensor.normal(Shape.of(2 * width, 1), 0, s, r));
        bMu = Var.param(Tensor.zeros(Shape.of(1, 1)));
        wPol = Var.param(Tensor.normal(Shape.of(2 * width, 1), 0, s, r));
        bPol = Var.param(Tensor.zeros(Shape.of(1, 1)));
        wLit = Var.param(Tensor.normal(Shape.of(width, 1), 0, s, r));
        bLit = Var.param(Tensor.zeros(Shape.of(1, 1)));
    }

    private List<Var> rlParameters() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(List.of(wMu, bMu, wPol, bPol));
        return p;
    }

    private List<Var> supervisedParameters() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(List.of(wLit, bLit));
        return p;
    }

    /** Literal embeddings {@code [2n, width]}: positive literals first. */
    private Var literals(RandomSatFormulas.Formula f) {
        return encoder.literals(NeuroSat.batch(List.of(f)), rounds);
    }

    /** Per-variable heads: {@code [n, 1]} mu and polarity logits. */
    private Var[] heads(RandomSatFormulas.Formula f) {
        int n = f.variables();
        Var l = literals(f);
        Var v = Var.concat(1, l.slice(0, 0, n), l.slice(0, n, 2 * n));
        return new Var[] {v.matmul(wMu).add(bMu), v.matmul(wPol).add(bPol)};
    }

    /**
     * Solves with guidance and returns the cost.
     *
     * @param f the formula
     * @param g the guidance (null for plain CDCL)
     * @param options solver options
     * @return {@code {decisions, conflicts, propagations, solved (1/0)}}
     */
    public static long[] cost(RandomSatFormulas.Formula f, Guidance g, CdclSolver.Options options) {
        CdclSolver s = CdclSolver.of(f.variables(), f.clauses(), options);
        if (g != null) {
            s.setWeights(g.weights());
            s.setPolarities(g.polarities());
        }
        CdclSolver.Status st = s.solve();
        CdclSolver.Stats x = s.stats();
        return new long[] {x.decisions(), x.conflicts(), x.propagations(), st == CdclSolver.Status.UNKNOWN ? 0 : 1};
    }

    /**
     * Trains by group-relative policy gradient.
     *
     * @param formulas training formulas
     * @param iterations passes over the formulas
     * @param group samples per formula ({@code M})
     * @param learningRate Adam's rate
     * @param options solver options (with a conflict budget)
     * @param seed the seed
     * @return the mean cost of the last pass's samples relative to plain CDCL
     */
    public double trainRl(List<RandomSatFormulas.Formula> formulas, int iterations, int group, float learningRate,
            CdclSolver.Options options, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        double[] base = new double[formulas.size()];
        for (int i = 0; i < base.length; i++) {
            base[i] = Math.max(1, cost(formulas.get(i), null, options)[0]);
        }
        double rel = 0;
        for (int it = 0; it < iterations; it++) {
            rel = 0;
            for (int i = 0; i < formulas.size(); i++) {
                RandomSatFormulas.Formula f = formulas.get(i);
                int n = f.variables();
                Var[] h = heads(f);
                float[] mu = h[0].value().toArray();
                float[] rho = h[1].value().toArray();
                double[] rewards = new double[group];
                float[][] eps = new float[group][n];
                float[][] pol = new float[group][n];
                for (int k = 0; k < group; k++) {
                    double[] w = new double[n];
                    boolean[] p = new boolean[n];
                    for (int v = 0; v < n; v++) {
                        eps[k][v] = (float) (mu[v] + SIGMA * r.nextGaussian());
                        w[v] = Math.exp(eps[k][v]);
                        p[v] = r.nextDouble() < 1 / (1 + Math.exp(-rho[v]));
                        pol[k][v] = p[v] ? 1 : 0;
                    }
                    long c = cost(f, new Guidance(w, p), options)[0];
                    rewards[k] = -c / base[i];
                    rel += c / base[i] / (group * formulas.size());
                }
                double mean = 0;
                for (double x : rewards) {
                    mean += x / group;
                }
                double sd = 0;
                for (double x : rewards) {
                    sd += (x - mean) * (x - mean) / group;
                }
                sd = Math.sqrt(sd) + 1e-6;
                Var loss = null;
                for (int k = 0; k < group; k++) {
                    float adv = (float) ((rewards[k] - mean) / sd);
                    Var z = Var.of(Tensor.of(Shape.of(n, 1), eps[k]));
                    Var logpW = z.sub(h[0]).square().mul((float) (-0.5 / (SIGMA * SIGMA))).sum();
                    Var y = Var.of(Tensor.of(Shape.of(n, 1), pol[k]));
                    Var logpP = y.mul(h[1]).sub(h[1].softplus()).sum();
                    Var term = logpW.add(logpP).mul(-adv / (group * (float) n));
                    loss = loss == null ? term : loss.add(term);
                }
                loss.backward();
                opt.step(rlParameters());
            }
        }
        return rel;
    }

    /**
     * The RL policy's mode for a formula.
     *
     * @param f the formula
     * @return weights {@code exp(mu)} and polarities {@code rho > 0}
     */
    public Guidance guidance(RandomSatFormulas.Formula f) {
        return Var.noGrad(() -> {
            Var[] h = heads(f);
            float[] mu = h[0].value().toArray();
            float[] rho = h[1].value().toArray();
            double[] w = new double[mu.length];
            boolean[] p = new boolean[mu.length];
            for (int v = 0; v < mu.length; v++) {
                w[v] = Math.exp(mu[v]);
                p[v] = rho[v] > 0;
            }
            return new Guidance(w, p);
        });
    }

    /**
     * The backbone of a satisfiable formula: literals true in every model.
     *
     * @param f the formula
     * @return per literal node ({@code +v} at {@code v-1}, {@code -v} at {@code n+v-1}) 1 if in
     *     the backbone, or null if unsatisfiable
     */
    public static float[] backbone(RandomSatFormulas.Formula f) {
        int n = f.variables();
        boolean[] m = CdclSolver.solve(n, f.clauses(), CdclSolver.Options.defaults());
        if (m == null) {
            return null;
        }
        float[] out = new float[2 * n];
        for (int v = 0; v < n; v++) {
            List<int[]> cs = new ArrayList<>(f.clauses());
            cs.add(new int[] {m[v] ? -(v + 1) : v + 1});
            if (CdclSolver.solve(n, cs, CdclSolver.Options.defaults()) == null) {
                out[m[v] ? v : n + v] = 1;
            }
        }
        return out;
    }

    /**
     * Trains the supervised backbone baseline and tunes {@code alpha} on the training formulas.
     *
     * @param formulas training formulas (unsatisfiable ones are skipped)
     * @param epochs passes
     * @param learningRate Adam's rate
     * @param options solver options for tuning alpha
     */
    public void trainSupervised(List<RandomSatFormulas.Formula> formulas, int epochs, float learningRate,
            CdclSolver.Options options) {
        List<RandomSatFormulas.Formula> sat = new ArrayList<>();
        List<float[]> labels = new ArrayList<>();
        for (RandomSatFormulas.Formula f : formulas) {
            float[] b = backbone(f);
            if (b != null) {
                sat.add(f);
                labels.add(b);
            }
        }
        Optimizer opt = Optimizer.adam(learningRate);
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < sat.size(); i++) {
                RandomSatFormulas.Formula f = sat.get(i);
                Var logit = literals(f).matmul(wLit).add(bLit);
                Var y = Var.of(Tensor.of(Shape.of(2 * f.variables(), 1), labels.get(i)));
                Var loss = logit.softplus().sub(y.mul(logit)).mean();
                loss.backward();
                opt.step(supervisedParameters());
            }
        }
        double chosen = 1;
        double best = Double.POSITIVE_INFINITY;
        for (double a : new double[] {1e-1, 1, 10, 100}) {
            alpha = a;
            double c = 0;
            for (RandomSatFormulas.Formula f : formulas) {
                c += cost(f, supervisedGuidance(f), options)[0];
            }
            if (c < best) {
                best = c;
                chosen = a;
            }
        }
        alpha = chosen;
    }

    /**
     * The supervised baseline's guidance.
     *
     * @param f the formula
     * @return weights {@code 1 + alpha max(p+, p-)} and polarities {@code p+ > p-}
     */
    public Guidance supervisedGuidance(RandomSatFormulas.Formula f) {
        return Var.noGrad(() -> {
            int n = f.variables();
            float[] lg = literals(f).matmul(wLit).add(bLit).sigmoid().value().toArray();
            double[] w = new double[n];
            boolean[] p = new boolean[n];
            for (int v = 0; v < n; v++) {
                w[v] = 1 + alpha * Math.max(lg[v], lg[n + v]);
                p[v] = lg[v] > lg[n + v];
            }
            return new Guidance(w, p);
        });
    }

    /**
     * Random guidance: log-normal(0, 1) weights and fair-coin polarities.
     *
     * @param n variables
     * @param seed the seed
     * @return the guidance
     */
    public static Guidance random(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[n];
        boolean[] p = new boolean[n];
        for (int v = 0; v < n; v++) {
            w[v] = Math.exp(r.nextGaussian());
            p[v] = r.nextBoolean();
        }
        return new Guidance(w, p);
    }

    /** @return the encoder width */
    public int width() {
        return width;
    }
}
