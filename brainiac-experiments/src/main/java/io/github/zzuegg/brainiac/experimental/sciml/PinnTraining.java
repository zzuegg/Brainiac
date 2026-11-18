package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Training recipes for {@link PhysicsInformedNetwork}s: the usual Adam-then-L-BFGS schedule
 * plus the four fixes for PINN optimisation failures, each switched on in {@link Options}:
 *
 * <ul>
 *   <li>{@link Options#selfAdaptive}: SA-PINN (McClenny and Braga-Neto, 2023,
 *       <a href="https://arxiv.org/abs/2009.04544">arXiv:2009.04544</a>). Each collocation
 *       point gets a trainable weight {@code lambda_i} on its residual; the network minimises
 *       {@code mean((lambda_i r_i)^2)} while the weights take gradient-ascent steps, so stubborn
 *       points gain weight.
 *   <li>{@link Options#ntkBalancing}: NTK loss balancing (Wang, Yu and Perdikaris, 2022,
 *       <a href="https://arxiv.org/abs/2007.14527">arXiv:2007.14527</a>). Every few steps the
 *       trace of each loss term's neural-tangent-kernel block, {@code mean_j |d e_j / d theta|^2}
 *       over a few points, is estimated, and term {@code k} is weighted by
 *       {@code sum_l tr_l / tr_k}, so every term converges at a similar rate.
 *   <li>{@link Options#causalWeights}: causal training (Wang, Sankaran and Perdikaris, 2024,
 *       <a href="https://arxiv.org/abs/2203.07404">arXiv:2203.07404</a>). The time axis is cut
 *       into bins with mean squared residuals {@code L_i}; bin {@code i} is weighted by
 *       {@code exp(-eps sum_{k<i} L_k)} (no gradient through the weight), so later times only
 *       count once earlier ones are solved.
 *   <li>{@link Options#residualResampling}: RAR-D (Wu et al., 2023,
 *       <a href="https://arxiv.org/abs/2207.10289">arXiv:2207.10289</a>), via
 *       {@link CollocationSampler#residualAdaptive}: the collocation set is redrawn every few
 *       steps with density proportional to the residual.
 * </ul>
 *
 * <p>For problems that stay ill-conditioned with all four, {@link #curriculum} (coefficient
 * curriculum), {@link #timeMarching} (sequence-to-sequence time windows) and
 * {@link #periodicField} (exact periodic boundaries by a Fourier embedding) change the problem
 * the optimiser sees; {@link PinnNaturalGradient} and {@link NystromNewtonCg} change the
 * optimiser.
 */
public final class PinnTraining {

    private PinnTraining() {
    }

    /**
     * A data or condition loss term: per-point errors computed from the network.
     *
     * @param name a label
     * @param errors the errors {@code [m, k]}; the term's loss is their mean square
     */
    public record Term(String name, Function<PhysicsInformedNetwork, Var> errors) {
    }

    /**
     * A Dirichlet (or initial) condition: the field equals given values at given points.
     *
     * @param name a label
     * @param points {@code [m][d]}
     * @param values {@code [m]}
     * @return the term
     */
    public static Term dirichlet(String name, double[][] points, double[] values) {
        Tensor x = CollocationSampler.tensor(points);
        Tensor y = CollocationSampler.column(values);
        return new Term(name, p -> p.forward(Var.of(x)).slice(1, 0, 1).sub(y));
    }

    /**
     * A periodic condition: the field (and optionally its derivative along an axis) agrees at
     * paired points.
     *
     * @param name a label
     * @param left points on one side {@code [m][d]}
     * @param right the matching points on the other side
     * @param axis the periodic axis, for the derivative match; -1 to match values only
     * @return the term
     */
    public static Term periodic(String name, double[][] left, double[][] right, int axis) {
        Tensor a = CollocationSampler.tensor(left);
        Tensor b = CollocationSampler.tensor(right);
        return new Term(name, p -> {
            if (axis < 0) {
                return p.forward(Var.of(a)).sub(p.forward(Var.of(b)));
            }
            PhysicsInformedNetwork.Derivatives da = p.derivatives(a);
            PhysicsInformedNetwork.Derivatives db = p.derivatives(b);
            return Var.concat(1, da.u().sub(db.u()), da.first(axis).sub(db.first(axis)));
        });
    }

    /**
     * A PDE problem: residual, domain and conditions.
     *
     * @param residual the PDE residual
     * @param domain the collocation domain
     * @param timeAxis the time coordinate, for causal weights; -1 for a steady problem
     * @param conditions boundary, initial and data terms
     */
    public record Problem(PhysicsInformedNetwork.Residual residual, CollocationSampler domain, int timeAxis,
            List<Term> conditions) {
    }

    /** Training settings; the defaults are plain Adam then L-BFGS. */
    public static final class Options {
        int adamSteps = 1000;
        float learningRate = 1e-3f;
        int lbfgsIterations = 0;
        int collocation = 512;
        float selfAdaptiveRate = 0;
        int ntkEvery = 0;
        int ntkSample = 16;
        double causalEpsilon = 0;
        int causalBins = 16;
        int resampleEvery = 0;
        double resampleK = 1;
        double resampleC = 1;

        /** Creates the default options. */
        public Options() {
        }

        /**
         * @param steps Adam steps
         * @param rate the learning rate
         * @return these options
         */
        public Options adam(int steps, float rate) {
            this.adamSteps = steps;
            this.learningRate = rate;
            return this;
        }

        /**
         * @param iterations L-BFGS iterations after Adam, 0 for none
         * @return these options
         */
        public Options lbfgs(int iterations) {
            this.lbfgsIterations = iterations;
            return this;
        }

        /**
         * @param points collocation points
         * @return these options
         */
        public Options collocation(int points) {
            this.collocation = points;
            return this;
        }

        /**
         * Turns on SA-PINN point weights.
         *
         * @param rate the ascent rate of the weights (Adam), for example 0.01
         * @return these options
         */
        public Options selfAdaptive(float rate) {
            this.selfAdaptiveRate = rate;
            return this;
        }

        /**
         * Turns on NTK loss balancing.
         *
         * @param every steps between weight updates
         * @param sample points per term in each trace estimate
         * @return these options
         */
        public Options ntkBalancing(int every, int sample) {
            this.ntkEvery = every;
            this.ntkSample = sample;
            return this;
        }

        /**
         * Turns on causal weights.
         *
         * @param epsilon the causality strength, for example 1 to 100
         * @param bins the time bins
         * @return these options
         */
        public Options causalWeights(double epsilon, int bins) {
            this.causalEpsilon = epsilon;
            this.causalBins = bins;
            return this;
        }

        /**
         * Turns on residual-based resampling (RAR-D).
         *
         * @param every steps between redraws
         * @param k the residual exponent
         * @param c the uniform floor
         * @return these options
         */
        public Options residualResampling(int every, double k, double c) {
            this.resampleEvery = every;
            this.resampleK = k;
            this.resampleC = c;
            return this;
        }
    }

    /**
     * The outcome of training.
     *
     * @param loss the final total loss
     * @param millis wall time
     * @param termWeights the final weights, residual first, then each condition
     * @param causalMinWeight the smallest causal weight at the end (1 without causal training)
     */
    public record Result(double loss, double millis, double[] termWeights, double causalMinWeight) {
    }

    /**
     * The causal weights {@code w_i = exp(-eps sum_{k<i} L_k)}.
     *
     * @param binLosses the mean squared residual of each time bin, in time order
     * @param epsilon the causality strength
     * @return the weights
     */
    public static double[] causalWeights(double[] binLosses, double epsilon) {
        double[] w = new double[binLosses.length];
        double acc = 0;
        for (int i = 0; i < w.length; i++) {
            w[i] = Math.exp(-epsilon * acc);
            acc += binLosses[i];
        }
        return w;
    }

    /**
     * The NTK trace estimate of one loss term: the mean over its first {@code sample} points of
     * the squared norm of the parameter gradient of each error.
     *
     * @param errors per-point errors {@code [m, k]}
     * @param parameters the network parameters
     * @param sample how many points to use
     * @return the per-point trace
     */
    public static double ntkTrace(Var errors, List<Var> parameters, int sample) {
        int m = Math.min(sample, errors.shape().dim(0));
        double sum = 0;
        for (int j = 0; j < m; j++) {
            List<Var> g = Var.gradients(errors.slice(0, j, j + 1).sum(), parameters, false);
            for (Var v : g) {
                for (float f : v.value().toArray()) {
                    sum += (double) f * f;
                }
            }
        }
        return sum / m;
    }

    private static final class State {
        final PhysicsInformedNetwork pinn;
        final Problem problem;
        final Options options;
        double[][] points;
        Tensor pointTensor;
        Tensor bins;
        Var lambda;
        double[] weights;
        double causalMin = 1;

        State(PhysicsInformedNetwork pinn, Problem problem, Options options) {
            this.pinn = pinn;
            this.problem = problem;
            this.options = options;
            this.weights = new double[problem.conditions().size() + 1];
            java.util.Arrays.fill(weights, 1);
        }

        void points(double[][] p) {
            points = p;
            pointTensor = CollocationSampler.tensor(p);
            if (options.causalEpsilon > 0) {
                int t = problem.timeAxis();
                double lo = problem.domain().low(t);
                double hi = problem.domain().high(t);
                int m = options.causalBins;
                float[] b = new float[p.length * m];
                int[] counts = new int[m];
                int[] idx = new int[p.length];
                for (int i = 0; i < p.length; i++) {
                    idx[i] = Math.min(m - 1, (int) ((p[i][t] - lo) / (hi - lo) * m));
                    counts[idx[i]]++;
                }
                for (int i = 0; i < p.length; i++) {
                    b[i * m + idx[i]] = 1f / Math.max(1, counts[idx[i]]);
                }
                bins = Tensor.of(Shape.of(p.length, m), b);
            }
            if (options.selfAdaptiveRate > 0) {
                lambda = Var.param(Tensor.ones(Shape.of(p.length, 1)));
            }
        }

        Var residualLoss(Var r) {
            Var e = lambda != null ? r.mul(lambda) : r;
            Var sq = e.square().mean(1, true);   // [n, 1]
            if (bins == null) {
                return sq.mean();
            }
            Var binLoss = sq.transpose().matmul(Var.of(bins));   // [1, bins]
            double[] w = causalWeights(toDouble(binLoss.value()), options.causalEpsilon);
            causalMin = w[w.length - 1];
            float[] wf = new float[w.length];
            for (int i = 0; i < w.length; i++) {
                wf[i] = (float) w[i];
            }
            return binLoss.mul(Var.of(Tensor.of(Shape.of(1, w.length), wf))).mean();
        }

        Var total() {
            Var loss = residualLoss(pinn.residual(pointTensor, problem.residual())).mul((float) weights[0]);
            List<Term> c = problem.conditions();
            for (int k = 0; k < c.size(); k++) {
                loss = loss.add(c.get(k).errors().apply(pinn).square().mean().mul((float) weights[k + 1]));
            }
            return loss;
        }

        void balance() {
            List<Var> params = pinn.parameters();
            double[] tr = new double[weights.length];
            tr[0] = ntkTrace(pinn.residual(pointTensor, problem.residual()), params, options.ntkSample);
            List<Term> c = problem.conditions();
            for (int k = 0; k < c.size(); k++) {
                tr[k + 1] = ntkTrace(c.get(k).errors().apply(pinn), params, options.ntkSample);
            }
            double sum = 0;
            for (double v : tr) {
                sum += v;
            }
            for (int k = 0; k < tr.length; k++) {
                double w = sum / Math.max(tr[k], 1e-12);
                weights[k] = 0.9 * weights[k] + 0.1 * Math.min(w, 1e4);
            }
            // the residual weight sets the scale; keep it near 1 so the learning rate holds
            double r0 = weights[0];
            for (int k = 0; k < weights.length; k++) {
                weights[k] /= r0;
            }
        }
    }

    private static double[] toDouble(Tensor t) {
        float[] f = t.toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /**
     * The absolute residual at points, without a graph that outlives the call.
     *
     * @param pinn the network
     * @param residual the PDE residual
     * @param points {@code [n][d]}
     * @return {@code |r|} per point (the norm over components for systems)
     */
    public static double[] absoluteResidual(PhysicsInformedNetwork pinn, PhysicsInformedNetwork.Residual residual,
            double[][] points) {
        Var r = pinn.residual(CollocationSampler.tensor(points), residual);
        int k = r.shape().dim(1);
        float[] v = r.value().toArray();
        double[] out = new double[points.length];
        for (int i = 0; i < out.length; i++) {
            double s = 0;
            for (int j = 0; j < k; j++) {
                s += (double) v[i * k + j] * v[i * k + j];
            }
            out[i] = Math.sqrt(s);
        }
        return out;
    }

    /**
     * Trains a PINN.
     *
     * @param pinn the network
     * @param problem the PDE problem
     * @param options the recipe
     * @param seed the seed for collocation sampling
     * @return the outcome
     */
    public static Result train(PhysicsInformedNetwork pinn, Problem problem, Options options, long seed) {
        long start = System.nanoTime();
        SplittableRandom random = new SplittableRandom(seed);
        State s = new State(pinn, problem, options);
        s.points(problem.domain().latinHypercube(options.collocation, random));
        List<Var> params = new ArrayList<>(pinn.parameters());
        Optimizer adam = Optimizer.adam(options.learningRate);
        Optimizer ascent = Optimizer.adam(options.selfAdaptiveRate > 0 ? options.selfAdaptiveRate : 1e-3f);
        double loss = Double.NaN;
        for (int step = 0; step < options.adamSteps; step++) {
            if (options.resampleEvery > 0 && step > 0 && step % options.resampleEvery == 0) {
                if (s.lambda != null) {
                    ascent.forget(s.lambda);
                }
                s.points(problem.domain().residualAdaptive(options.collocation, 4 * options.collocation,
                        p -> absoluteResidual(pinn, problem.residual(), p), options.resampleK, options.resampleC,
                        random));
            }
            if (options.ntkEvery > 0 && step % options.ntkEvery == 0) {
                s.balance();
            }
            for (Var p : params) {
                p.zeroGrad();
            }
            if (s.lambda != null) {
                s.lambda.zeroGrad();
            }
            Var l = s.total();
            loss = l.value().item();
            l.backward();
            adam.step(params);
            if (s.lambda != null) {
                s.lambda.setGrad(s.lambda.grad().neg());
                ascent.step(List.of(s.lambda));
            }
        }
        if (options.lbfgsIterations > 0) {
            Lbfgs lbfgs = Lbfgs.fullBatch(1f, 20);
            Optimizer.Reevaluation re = () -> {
                for (Var p : params) {
                    p.zeroGrad();
                }
                Var l = s.total();
                l.backward();
                if (s.lambda != null) {
                    s.lambda.zeroGrad();
                }
                return l.value().item();
            };
            for (int it = 0; it < options.lbfgsIterations; it++) {
                loss = re.lossAndGradients();
                if (!Double.isFinite(loss)) {
                    break;
                }
                lbfgs.step(params, re);
            }
            loss = s.total().value().item();
        }
        return new Result(loss, (System.nanoTime() - start) / 1e6, s.weights.clone(), s.causalMin);
    }

    // ---------------------------------------------------------------- 9.5.14 curriculum, time marching, exact periodicity

    /**
     * Exact periodic boundary conditions by construction (Dong and Ni, 2021,
     * <a href="https://arxiv.org/abs/2007.07442">arXiv:2007.07442</a>): the periodic coordinate
     * enters the network only as {@code sin(2 pi (x - low) / period)} and
     * {@code cos(2 pi (x - low) / period)}, so the field and all its derivatives along that axis
     * are periodic for any weights, and the periodic loss term can be dropped
     * ({@link #withoutTerm}). Derivatives with respect to {@code x} flow through the embedding
     * by autograd as usual.
     *
     * @param inner the network over the {@code d + 1} embedded coordinates (the periodic one
     *     replaced by its sine and cosine, first; the others in order)
     * @param axis the periodic coordinate
     * @param low the start of the period
     * @param period the period length
     * @return a field over the original {@code d} coordinates
     */
    public static Layer periodicField(Layer inner, int axis, double low, double period) {
        float w = (float) (2 * Math.PI / period);
        float lo = (float) low;
        return new Layer() {
            @Override
            public Var forward(Var x) {
                int d = x.shape().dim(1);
                Var theta = x.slice(1, axis, axis + 1).sub(lo).mul(w);
                List<Var> parts = new ArrayList<>();
                parts.add(theta.sin());
                parts.add(theta.cos());
                for (int j = 0; j < d; j++) {
                    if (j != axis) {
                        parts.add(x.slice(1, j, j + 1));
                    }
                }
                return inner.forward(Var.concat(1, parts.toArray(new Var[0])));
            }

            @Override
            public List<Var> parameters() {
                return inner.parameters();
            }

            @Override
            public int outputSize(int inputSize) {
                return inner.outputSize(inputSize + 1);
            }

            @Override
            public String toString() {
                return "PeriodicField(axis " + axis + ", " + inner + ")";
            }
        };
    }

    /**
     * A problem with the condition terms of one name removed, for example the
     * {@code "periodic"} term once {@link #periodicField} satisfies it exactly.
     *
     * @param problem the problem
     * @param name the term name to drop
     * @return the reduced problem
     */
    public static Problem withoutTerm(Problem problem, String name) {
        List<Term> kept = new ArrayList<>();
        for (Term t : problem.conditions()) {
            if (!t.name().equals(name)) {
                kept.add(t);
            }
        }
        return new Problem(problem.residual(), problem.domain(), problem.timeAxis(), kept);
    }

    /**
     * Curriculum regularisation (Krishnapriyan et al., NeurIPS 2021,
     * <a href="https://arxiv.org/abs/2109.01050">arXiv:2109.01050</a>): train on an easy version
     * of the PDE first (for advection, a small speed {@code beta}) and move through harder
     * versions to the target, warm-starting each stage from the last. The loss landscape of the
     * hard problem is badly conditioned; the curriculum starts the optimiser in the basin of the
     * right solution. Each stage runs the full {@code options} recipe.
     *
     * @param pinn the network, trained in place
     * @param stages the problems from easiest to the target
     * @param options the recipe for every stage
     * @param seed the seed
     * @return the last stage's outcome, with the total time
     */
    public static Result curriculum(PhysicsInformedNetwork pinn, List<Problem> stages, Options options, long seed) {
        double millis = 0;
        Result last = null;
        for (int i = 0; i < stages.size(); i++) {
            last = train(pinn, stages.get(i), options, seed + i);
            millis += last.millis();
        }
        if (last == null) {
            throw new IllegalArgumentException("a curriculum needs at least one stage");
        }
        return new Result(last.loss(), millis, last.termWeights(), last.causalMinWeight());
    }

    /**
     * The outcome of time marching.
     *
     * @param field the piecewise-in-time field over the whole domain
     * @param loss the last window's final loss
     * @param millis the total training time
     */
    public record Marched(Layer field, double loss, double millis) {
    }

    /**
     * Sequence-to-sequence time marching (Krishnapriyan et al., 2021; also Wight and Zhao,
     * 2020): the time axis is cut into {@code windows} equal slabs that are solved one after
     * another, each by its own network (warm-started from the previous window's weights) whose
     * initial condition is the previous window's prediction at the shared time. Each window is a
     * short-time problem, which PINNs solve well; errors can only propagate forward in time.
     *
     * @param fields makes a fresh field network (all of the same architecture)
     * @param problem the full problem; its condition terms are ignored and rebuilt per window
     * @param initial the initial condition at a point
     * @param boundaries the boundary terms for a window's box (empty when the field is
     *     {@link #periodicField exactly periodic})
     * @param windows the number of time windows
     * @param initialPoints initial-condition points per window
     * @param options the recipe for every window
     * @param seed the seed
     * @return the piecewise field and the cost
     */
    public static Marched timeMarching(java.util.function.Supplier<Layer> fields, Problem problem,
            java.util.function.ToDoubleFunction<double[]> initial,
            Function<CollocationSampler, List<Term>> boundaries, int windows, int initialPoints, Options options,
            long seed) {
        int ta = problem.timeAxis();
        if (ta < 0) {
            throw new IllegalArgumentException("time marching needs a time axis");
        }
        CollocationSampler domain = problem.domain();
        int d = domain.dimension();
        double t0 = domain.low(ta);
        double t1 = domain.high(ta);
        SplittableRandom random = new SplittableRandom(seed);
        List<Layer> nets = new ArrayList<>();
        double[] edges = new double[windows + 1];
        double millis = 0;
        double loss = Double.NaN;
        Layer previous = null;
        for (int k = 0; k <= windows; k++) {
            edges[k] = t0 + (t1 - t0) * k / windows;
        }
        for (int k = 0; k < windows; k++) {
            double[] low = new double[d];
            double[] high = new double[d];
            for (int a = 0; a < d; a++) {
                low[a] = domain.low(a);
                high[a] = domain.high(a);
            }
            low[ta] = edges[k];
            high[ta] = edges[k + 1];
            CollocationSampler box = CollocationSampler.box(low, high);
            double[][] ic = box.face(ta, false, initialPoints, random);
            double[] values;
            if (previous == null) {
                values = new double[ic.length];
                for (int i = 0; i < ic.length; i++) {
                    values[i] = initial.applyAsDouble(ic[i]);
                }
            } else {
                values = PhysicsInformedNetwork.of(previous).predict(ic);
            }
            List<Term> terms = new ArrayList<>();
            terms.add(dirichlet("initial", ic, values));
            terms.addAll(boundaries.apply(box));
            Layer field = fields.get();
            if (previous != null) {
                io.github.zzuegg.brainiac.experimental.train.CurvatureProducts.assign(field.parameters(),
                        io.github.zzuegg.brainiac.experimental.train.CurvatureProducts.values(previous.parameters()));
            }
            Result r = train(PhysicsInformedNetwork.of(field), new Problem(problem.residual(), box, ta, terms), options,
                    random.nextLong());
            millis += r.millis();
            loss = r.loss();
            nets.add(field);
            previous = field;
        }
        return new Marched(piecewiseInTime(nets, edges, ta), loss, millis);
    }

    /** One field per time slab, each used for the points whose time falls in its slab. */
    private static Layer piecewiseInTime(List<Layer> nets, double[] edges, int axis) {
        return new Layer() {
            @Override
            public Var forward(Var x) {
                int n = x.shape().dim(0);
                int d = x.shape().dim(1);
                float[] v = x.value().toArray();
                Var out = null;
                for (int k = 0; k < nets.size(); k++) {
                    float[] m = new float[n];
                    for (int i = 0; i < n; i++) {
                        double t = v[i * d + axis];
                        boolean last = k == nets.size() - 1;
                        m[i] = (t >= edges[k] || k == 0) && (t < edges[k + 1] || last) ? 1 : 0;
                    }
                    Var part = nets.get(k).forward(x).mul(Tensor.of(Shape.of(n, 1), m));
                    out = out == null ? part : out.add(part);
                }
                return out;
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>();
                nets.forEach(l -> p.addAll(l.parameters()));
                return p;
            }

            @Override
            public int outputSize(int inputSize) {
                return nets.get(0).outputSize(inputSize);
            }

            @Override
            public String toString() {
                return "TimeMarched(" + nets.size() + " windows)";
            }
        };
    }
}
