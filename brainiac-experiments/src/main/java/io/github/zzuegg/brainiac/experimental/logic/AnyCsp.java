package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * ANYCSP: a graph neural network as a global search heuristic for any CSP (Tönshoff, Kisin,
 * Lindner &amp; Grohe, IJCAI 2023, <a href="https://arxiv.org/abs/2208.10227">arXiv:2208.10227</a>).
 *
 * <p>The instance is one graph whose nodes are the (variable, value) pairs; two value nodes are
 * linked when a constraint forbids that pair, so one network handles any domain size and
 * constraint graph (this block covers binary constraints with a common domain size). At every step
 * of a stochastic search the recurrent network reads the current assignment (which values are
 * chosen, and how many conflicts every value would have against the neighbours' current values),
 * passes messages along the conflict edges and within each variable, and outputs a distribution
 * over the values of every variable; a new value is sampled for all variables at once (the global
 * action space) and the best assignment seen is kept. Training needs no labels: REINFORCE with
 * discounted returns, where the reward is the improvement of the best number of satisfied
 * constraints so far (so the policy is not punished for exploring).
 */
public final class AnyCsp {

    private final int hidden;
    private final Var wIn;
    private final Var bIn;
    private final Var wOut;
    private final Var bOut;
    private final List<Var> params;

    /**
     * A search run.
     *
     * @param solved whether a solution was found
     * @param steps steps until solved (or the budget)
     * @param bestViolations the fewest violated constraints seen
     * @param assignment the best assignment
     */
    public record Run(boolean solved, int steps, int bestViolations, int[] assignment) {
    }

    /**
     * Creates an untrained network.
     *
     * @param hidden the state width
     * @param seed the initialisation seed
     */
    public AnyCsp(int hidden, long seed) {
        this.hidden = hidden;
        SplittableRandom r = new SplittableRandom(seed);
        int in = hidden + 4 + hidden + hidden;
        wIn = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) (1 / Math.sqrt(in)), r));
        bIn = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        wOut = Var.param(Tensor.normal(Shape.of(hidden, 1), 0, (float) (1 / Math.sqrt(hidden)), r));
        bOut = Var.param(Tensor.zeros(Shape.of(1, 1)));
        params = List.of(wIn, bIn, wOut, bOut);
    }

    /** @return the parameters */
    public List<Var> parameters() {
        return params;
    }

    /** The static graph of an instance. */
    private static final class Graph {
        final Csp csp;
        final int n;
        final int d;
        final Tensor conflict;
        final float[] conflictArr;
        final Tensor varMean;

        Graph(Csp csp) {
            this.csp = csp;
            n = csp.variables();
            d = csp.domain(0);
            for (int x = 0; x < n; x++) {
                if (csp.domain(x) != d) {
                    throw new IllegalArgumentException("ANYCSP block needs a common domain size");
                }
            }
            int v = n * d;
            conflictArr = new float[v * v];
            for (Csp.Constraint k : csp.constraints()) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        if (!k.allowed(a, b)) {
                            conflictArr[(k.x() * d + a) * v + k.y() * d + b] += 1;
                            conflictArr[(k.y() * d + b) * v + k.x() * d + a] += 1;
                        }
                    }
                }
            }
            conflict = Tensor.of(Shape.of(v, v), conflictArr);
            float[] vm = new float[v * v];
            for (int x = 0; x < n; x++) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        vm[(x * d + a) * v + x * d + b] = 1f / d;
                    }
                }
            }
            varMean = Tensor.of(Shape.of(v, v), vm);
        }
    }

    /** One recurrent step: new state and value logits {@code [n, d]}. */
    private Var[] step(Graph g, Var h, int[] assign) {
        int v = g.n * g.d;
        float[] a = new float[v];
        for (int x = 0; x < g.n; x++) {
            a[x * g.d + assign[x]] = 1;
        }
        float[] cnt = new float[v];
        for (int i = 0; i < v; i++) {
            float s = 0;
            int base = i * v;
            for (int x = 0; x < g.n; x++) {
                s += g.conflictArr[base + x * g.d + assign[x]];
            }
            cnt[i] = s;
        }
        float[] feat = new float[v * 4];
        for (int i = 0; i < v; i++) {
            feat[i * 4] = a[i];
            feat[i * 4 + 1] = (float) Math.log1p(cnt[i]);
            feat[i * 4 + 2] = cnt[i] == 0 ? 1 : 0;
            feat[i * 4 + 3] = a[i] * (float) Math.log1p(cnt[i]);
        }
        Var x = Var.of(Tensor.of(Shape.of(v, 4), feat));
        Var chosen = h.mul(Var.of(Tensor.of(Shape.of(v, 1), a)));
        Var msg = Var.of(g.conflict).matmul(chosen).div(4f);
        Var vm = Var.of(g.varMean).matmul(h);
        Var nh = Var.concat(1, h, x, msg, vm).matmul(wIn).add(bIn).tanh();
        Var logits = nh.matmul(wOut).add(bOut).reshape(g.n, g.d);
        return new Var[] {nh, logits};
    }

    private static int[] sample(float[] probs, int n, int d, SplittableRandom r) {
        int[] out = new int[n];
        for (int x = 0; x < n; x++) {
            double u = r.nextDouble();
            double c = 0;
            out[x] = d - 1;
            for (int a = 0; a < d; a++) {
                c += probs[x * d + a];
                if (u < c) {
                    out[x] = a;
                    break;
                }
            }
        }
        return out;
    }

    /**
     * Searches for a solution.
     *
     * @param csp the problem (common domain size)
     * @param steps the step budget
     * @param seed the sampling seed
     * @return the run
     */
    public Run solve(Csp csp, int steps, long seed) {
        Graph g = new Graph(csp);
        SplittableRandom r = new SplittableRandom(seed);
        return Var.noGrad(() -> {
            int[] assign = new int[g.n];
            for (int x = 0; x < g.n; x++) {
                assign[x] = r.nextInt(g.d);
            }
            Var h = Var.of(Tensor.zeros(Shape.of(g.n * g.d, hidden)));
            int best = csp.violations(assign);
            int[] bestA = assign.clone();
            for (int t = 0; t < steps && best > 0; t++) {
                Var[] o = step(g, h, assign);
                h = o[0];
                assign = sample(o[1].softmax(1).value().toArray(), g.n, g.d, r);
                int v = csp.violations(assign);
                if (v < best) {
                    best = v;
                    bestA = assign.clone();
                    if (v == 0) {
                        return new Run(true, t + 1, 0, bestA);
                    }
                }
            }
            return new Run(best == 0, steps, best, bestA);
        });
    }

    /**
     * Trains by REINFORCE on random instances.
     *
     * @param instances training instances (common domain size each)
     * @param episodes training episodes (one instance each, cycled)
     * @param horizon steps per episode
     * @param gamma the discount
     * @param learningRate Adam's rate
     * @param seed the seed
     * @return the mean final best-violation count over the last tenth of the episodes
     */
    public double train(List<Csp> instances, int episodes, int horizon, double gamma, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        double tail = 0;
        int tailCount = 0;
        for (int e = 0; e < episodes; e++) {
            Csp csp = instances.get(e % instances.size());
            Graph g = new Graph(csp);
            int m = Math.max(1, csp.constraints().size());
            int[] assign = new int[g.n];
            for (int x = 0; x < g.n; x++) {
                assign[x] = r.nextInt(g.d);
            }
            Var h = Var.of(Tensor.zeros(Shape.of(g.n * g.d, hidden)));
            int best = csp.violations(assign);
            List<Var> logps = new ArrayList<>();
            double[] rewards = new double[horizon];
            for (int t = 0; t < horizon; t++) {
                Var[] o = step(g, h, assign);
                h = o[0];
                Var logp = o[1].logSoftmax(1);
                assign = sample(logp.value().exp().toArray(), g.n, g.d, r);
                float[] onehot = new float[g.n * g.d];
                for (int x = 0; x < g.n; x++) {
                    onehot[x * g.d + assign[x]] = 1;
                }
                logps.add(logp.mul(Var.of(Tensor.of(Shape.of(g.n, g.d), onehot))).sum());
                int v = csp.violations(assign);
                rewards[t] = Math.max(0, best - v) / (double) m;
                best = Math.min(best, v);
            }
            double ret = 0;
            double[] returns = new double[horizon];
            for (int t = horizon - 1; t >= 0; t--) {
                ret = rewards[t] + gamma * ret;
                returns[t] = ret;
            }
            double mean = 0;
            for (double x : returns) {
                mean += x / horizon;
            }
            Var loss = null;
            for (int t = 0; t < horizon; t++) {
                Var term = logps.get(t).mul((float) -(returns[t] - mean) * 10f);
                loss = loss == null ? term : loss.add(term);
            }
            loss.backward();
            opt.step(params);
            if (e >= episodes - Math.max(1, episodes / 10)) {
                tail += best;
                tailCount++;
            }
        }
        return tail / Math.max(1, tailCount);
    }
}
