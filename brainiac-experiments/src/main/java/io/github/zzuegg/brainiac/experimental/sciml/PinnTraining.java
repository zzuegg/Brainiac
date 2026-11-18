package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
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
}
