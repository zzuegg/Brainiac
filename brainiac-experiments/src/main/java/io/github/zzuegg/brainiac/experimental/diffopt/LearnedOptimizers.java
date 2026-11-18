package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned optimisers: small networks that map per-parameter features to updates, meta-trained
 * on a distribution of training tasks, and usable afterwards as an ordinary {@link Optimizer}.
 *
 * <ul>
 *   <li>{@link LstmL2O} (Andrychowicz et al., NeurIPS 2016, <a href="https://arxiv.org/abs/1606.04474">
 *       arXiv:1606.04474</a>): a coordinate-wise two-input LSTM (the gradient preprocessed into
 *       log-magnitude and sign) whose output, times 0.1, is the update. It is meta-trained by
 *       truncated backpropagation through the unrolled optimisation of random quadratics, with
 *       second derivatives of the optimisee dropped as in the paper.</li>
 *   <li>{@link CeloLite}, after Celo (Moudgil et al., TMLR 2025, <a href="https://arxiv.org/abs/2501.12670">
 *       arXiv:2501.12670</a>): a per-parameter MLP rule on Adam-normalised features (gradient and
 *       two momenta divided by the RMS of the gradient) whose output is a direction times an
 *       exponentiated magnitude, added as a learned residual to the Adam direction and scaled by a
 *       learned base step and the parameter tensor's RMS; plus a tiny LSTM scheduler over the
 *       loss trajectory that outputs an exponential step multiplier. Both are meta-trained by
 *       truncated persistent evolution strategies (PES, Vicol et al. 2021), with task augmentation
 *       by random per-parameter rescaling, the rule first and the scheduler second.</li>
 * </ul>
 * Meta-training tasks implement {@link Task} on flat double vectors, which keeps evolution
 * strategies fast; the trained optimisers then step any {@link Var} parameters.
 */
public final class LearnedOptimizers {

    private LearnedOptimizers() {
    }

    /** A meta-training task: a differentiable loss on a flat parameter vector. */
    public interface Task {

        /**
         * The parameter count.
         *
         * @return the dimension
         */
        int dimension();

        /**
         * A fresh initialisation.
         *
         * @param random the random source
         * @return the initial parameters
         */
        double[] init(SplittableRandom random);

        /**
         * The loss and its gradient.
         *
         * @param theta the parameters
         * @param gradient receives the gradient
         * @return the loss
         */
        double lossAndGradient(double[] theta, double[] gradient);
    }

    /**
     * A quadratic task {@code |W theta - y|^2 / d}.
     *
     * @param w the matrix {@code [d][d]}
     * @param y the target {@code [d]}
     */
    public record Quadratic(double[][] w, double[] y) implements Task {

        @Override
        public int dimension() {
            return y.length;
        }

        @Override
        public double[] init(SplittableRandom r) {
            double[] t = new double[y.length];
            for (int i = 0; i < t.length; i++) {
                t[i] = r.nextGaussian();
            }
            return t;
        }

        @Override
        public double lossAndGradient(double[] theta, double[] gradient) {
            int d = y.length;
            double[] res = new double[d];
            double loss = 0;
            for (int i = 0; i < d; i++) {
                double s = -y[i];
                for (int j = 0; j < d; j++) {
                    s += w[i][j] * theta[j];
                }
                res[i] = s;
                loss += s * s / d;
            }
            java.util.Arrays.fill(gradient, 0);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    gradient[j] += 2 * w[i][j] * res[i] / d;
                }
            }
            return loss;
        }
    }

    /**
     * A random quadratic {@code |W theta - y|^2 / d} with Gaussian {@code W / sqrt(d)} and {@code y}.
     *
     * @param d the dimension
     * @param random the random source
     * @return the task
     */
    public static Quadratic quadratic(int d, SplittableRandom random) {
        double[][] w = new double[d][d];
        double[] y = new double[d];
        for (int i = 0; i < d; i++) {
            y[i] = random.nextGaussian();
            for (int j = 0; j < d; j++) {
                w[i][j] = random.nextGaussian() / Math.sqrt(d);
            }
        }
        return new Quadratic(w, y);
    }

    /**
     * A {@code 2 -> hidden -> classes} tanh MLP with softmax cross-entropy on fixed points.
     *
     * @param x inputs {@code [n][2]}
     * @param labels classes
     * @param classes the class count
     * @param hidden the hidden width
     * @return the task
     */
    public static Task mlp(double[][] x, int[] labels, int classes, int hidden) {
        int in = x[0].length;
        int dim = in * hidden + hidden + hidden * classes + classes;
        return new Task() {
            @Override
            public int dimension() {
                return dim;
            }

            @Override
            public double[] init(SplittableRandom r) {
                double[] t = new double[dim];
                for (int i = 0; i < in * hidden; i++) {
                    t[i] = r.nextGaussian() / Math.sqrt(in);
                }
                int o = in * hidden + hidden;
                for (int i = 0; i < hidden * classes; i++) {
                    t[o + i] = r.nextGaussian() / Math.sqrt(hidden);
                }
                return t;
            }

            @Override
            public double lossAndGradient(double[] t, double[] g) {
                java.util.Arrays.fill(g, 0);
                int b1 = in * hidden;
                int w2 = b1 + hidden;
                int b2 = w2 + hidden * classes;
                double loss = 0;
                double[] h = new double[hidden];
                double[] z = new double[classes];
                double[] dh = new double[hidden];
                int n = x.length;
                for (int s = 0; s < n; s++) {
                    for (int k = 0; k < hidden; k++) {
                        double a = t[b1 + k];
                        for (int i = 0; i < in; i++) {
                            a += x[s][i] * t[i * hidden + k];
                        }
                        h[k] = Math.tanh(a);
                    }
                    double max = Double.NEGATIVE_INFINITY;
                    for (int c = 0; c < classes; c++) {
                        double a = t[b2 + c];
                        for (int k = 0; k < hidden; k++) {
                            a += h[k] * t[w2 + k * classes + c];
                        }
                        z[c] = a;
                        max = Math.max(max, a);
                    }
                    double sum = 0;
                    for (int c = 0; c < classes; c++) {
                        z[c] = Math.exp(z[c] - max);
                        sum += z[c];
                    }
                    loss += -Math.log(z[labels[s]] / sum + 1e-300) / n;
                    java.util.Arrays.fill(dh, 0);
                    for (int c = 0; c < classes; c++) {
                        double dz = (z[c] / sum - (c == labels[s] ? 1 : 0)) / n;
                        g[b2 + c] += dz;
                        for (int k = 0; k < hidden; k++) {
                            g[w2 + k * classes + c] += dz * h[k];
                            dh[k] += dz * t[w2 + k * classes + c];
                        }
                    }
                    for (int k = 0; k < hidden; k++) {
                        double da = dh[k] * (1 - h[k] * h[k]);
                        g[b1 + k] += da;
                        for (int i = 0; i < in; i++) {
                            g[i * hidden + k] += da * x[s][i];
                        }
                    }
                }
                return loss;
            }
        };
    }

    // ------------------------------------------------------------------------------- LSTM L2O

    /** The coordinate-wise LSTM optimiser. */
    public static final class LstmL2O implements Optimizer {
        private static final int HIDDEN = 10;
        private final Var wx;
        private final Var wh;
        private final Var bias;
        private final Var wo;
        private final Map<Var, Tensor[]> state = new IdentityHashMap<>();
        private float scale = 0.1f;

        private LstmL2O(SplittableRandom random) {
            this.wx = Var.param(Tensor.normal(Shape.of(2, 4 * HIDDEN), 0, 0.3f, random));
            this.wh = Var.param(Tensor.normal(Shape.of(HIDDEN, 4 * HIDDEN), 0, 0.15f, random));
            float[] b = new float[4 * HIDDEN];
            for (int k = HIDDEN; k < 2 * HIDDEN; k++) {
                b[k] = 1;   // forget gates start open
            }
            this.bias = Var.param(Tensor.of(Shape.of(1, 4 * HIDDEN), b));
            this.wo = Var.param(Tensor.normal(Shape.of(HIDDEN, 1), 0, 0.1f, random));
        }

        /**
         * Creates an untrained optimiser.
         *
         * @param seed the seed
         * @return the optimiser
         */
        public static LstmL2O create(long seed) {
            return new LstmL2O(new SplittableRandom(seed));
        }

        /**
         * The meta-parameters.
         *
         * @return the LSTM weights
         */
        public List<Var> metaParameters() {
            return List.of(wx, wh, bias, wo);
        }

        /** Andrychowicz's preprocessing with p = 10: (log|g| / p, sign g) or (-1, e^p g). */
        static Tensor preprocess(double[] g) {
            float[] v = new float[g.length * 2];
            double p = 10;
            for (int i = 0; i < g.length; i++) {
                double a = Math.abs(g[i]);
                if (a >= Math.exp(-p)) {
                    v[2 * i] = (float) (Math.log(a) / p);
                    v[2 * i + 1] = (float) Math.signum(g[i]);
                } else {
                    v[2 * i] = -1;
                    v[2 * i + 1] = (float) (Math.exp(p) * g[i]);
                }
            }
            return Tensor.of(Shape.of(g.length, 2), v);
        }

        /** One LSTM step on {@code [k, 2]} inputs: returns {update, h, c}. */
        Var[] cell(Var input, Var h, Var c) {
            int k = input.shape().dim(0);
            Var gates = input.matmul(wx).add(h.matmul(wh)).add(bias);
            Var i = gates.slice(1, 0, HIDDEN).sigmoid();
            Var f = gates.slice(1, HIDDEN, 2 * HIDDEN).sigmoid();
            Var o = gates.slice(1, 2 * HIDDEN, 3 * HIDDEN).sigmoid();
            Var gg = gates.slice(1, 3 * HIDDEN, 4 * HIDDEN).tanh();
            Var c2 = f.mul(c).add(i.mul(gg));
            Var h2 = o.mul(c2.tanh());
            Var upd = h2.matmul(wo).reshape(k).mul(scale);
            return new Var[] {upd, h2, c2};
        }

        /**
         * Meta-trains on random quadratics by truncated backpropagation through time.
         *
         * @param dimension the quadratics' dimension
         * @param iterations meta-iterations (one task each)
         * @param horizon inner steps per task
         * @param truncation steps per backpropagated segment
         * @param metaLearningRate Adam's step on the meta-parameters
         * @param seed the seed
         * @return the mean meta-loss of the last ten iterations
         */
        public double metaTrain(int dimension, int iterations, int horizon, int truncation, float metaLearningRate,
                long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            Optimizer meta = Optimizer.adam(metaLearningRate);
            double recent = 0;
            for (int it = 0; it < iterations; it++) {
                Quadratic task = quadratic(dimension, r);
                float[] wf = new float[dimension * dimension];
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        wf[i * dimension + j] = (float) task.w()[i][j];
                    }
                }
                Tensor wt = Tensor.of(Shape.of(dimension, dimension), wf);
                Tensor yt = Tensor.of(Shape.of(dimension, 1), toFloat(task.y()));
                double[] theta0 = task.init(r);
                Var theta = Var.of(Tensor.of(Shape.of(dimension, 1), toFloat(theta0)));
                Var h = Var.of(Tensor.zeros(Shape.of(dimension, HIDDEN)));
                Var c = Var.of(Tensor.zeros(Shape.of(dimension, HIDDEN)));
                double total = 0;
                for (int seg = 0; seg < horizon / truncation; seg++) {
                    Var loss = null;
                    for (int t = 0; t < truncation; t++) {
                        Var res = Var.of(wt).matmul(theta).sub(yt);
                        Var f = res.square().sum().div(dimension);
                        loss = loss == null ? f : loss.add(f);
                        double[] grad = new double[dimension];
                        task.lossAndGradient(theta.value().toDoubleArray(), grad);
                        Var[] out = cell(Var.of(preprocess(grad)), h, c);
                        theta = theta.add(out[0].reshape(dimension, 1));
                        h = out[1];
                        c = out[2];
                    }
                    loss.div(truncation).backward();
                    total += loss.value().toDoubleArray()[0];
                    meta.step(metaParameters());
                    for (Var p : metaParameters()) {
                        p.zeroGrad();
                    }
                    theta = Var.of(theta.value());
                    h = Var.of(h.value());
                    c = Var.of(c.value());
                }
                if (it >= iterations - 10) {
                    recent += total / horizon / Math.min(10, iterations);
                }
            }
            return recent;
        }

        @Override
        public void step(List<Var> parameters) {
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                int k = (int) p.value().shape().size();
                Tensor[] st = state.computeIfAbsent(p, q -> new Tensor[] {Tensor.zeros(Shape.of(k, HIDDEN)),
                    Tensor.zeros(Shape.of(k, HIDDEN))});
                Var[] out = Var.noGrad(() -> {
                    Var[] o = cell(Var.of(preprocess(g.toDoubleArray())), Var.of(st[0]), Var.of(st[1]));
                    return o;
                });
                st[0] = out[1].value();
                st[1] = out[2].value();
                p.assign(p.value().add(out[0].value().reshape(p.value().shape())));
                p.zeroGrad();
            }
        }

        @Override
        public void forget(Var parameter) {
            state.remove(parameter);
        }

        @Override
        public float learningRate() {
            return scale;
        }

        @Override
        public void learningRate(float learningRate) {
            this.scale = learningRate;
        }
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    // ------------------------------------------------------------------------------- Celo-lite

    /** The Celo-style learned optimiser: an MLP update rule and an LSTM step-size scheduler. */
    public static final class CeloLite implements Optimizer {
        static final int FEATURES = 3;
        static final int HIDDEN = 8;
        static final int RULE_SIZE = FEATURES * HIDDEN + HIDDEN + HIDDEN * 2 + 2 + 1;
        static final int SCHED_HIDDEN = 4;
        static final int SCHED_SIZE = 2 * 4 * SCHED_HIDDEN + SCHED_HIDDEN * 4 * SCHED_HIDDEN + 4 * SCHED_HIDDEN
                + SCHED_HIDDEN + 1;

        private final double[] rule;
        private final double[] sched;
        private final Map<Var, double[][]> slots = new IdentityHashMap<>();
        private final double[] schedState = new double[2 * SCHED_HIDDEN];
        private double firstLoss = Double.NaN;
        private double lastLoss = Double.NaN;
        private long steps;
        private long horizon = 1000;
        private float multiplier = 1;

        private CeloLite(double[] rule, double[] sched) {
            this.rule = rule;
            this.sched = sched;
        }

        /**
         * An untrained optimiser: a zero residual on Adam with a base step of {@code 1e-2}, and a
         * scheduler that outputs a multiplier of one.
         *
         * @return the optimiser
         */
        public static CeloLite create() {
            double[] rule = new double[RULE_SIZE];
            rule[RULE_SIZE - 1] = Math.log(1e-2);
            return new CeloLite(rule, new double[SCHED_SIZE]);
        }

        /**
         * A copy with other meta-parameters.
         *
         * @param rule the rule parameters
         * @param sched the scheduler parameters
         * @return the optimiser
         */
        public static CeloLite of(double[] rule, double[] sched) {
            return new CeloLite(rule.clone(), sched.clone());
        }

        /**
         * The rule parameters.
         *
         * @return a copy
         */
        public double[] rule() {
            return rule.clone();
        }

        /**
         * The scheduler parameters.
         *
         * @return a copy
         */
        public double[] scheduler() {
            return sched.clone();
        }

        /**
         * Sets the horizon the scheduler's progress feature {@code t / T} refers to.
         *
         * @param steps the expected number of steps
         * @return this optimiser
         */
        public CeloLite horizon(long steps) {
            this.horizon = steps;
            return this;
        }

        /**
         * Feeds the current training loss to the scheduler (optional).
         *
         * @param loss the loss
         */
        public void observeLoss(double loss) {
            if (Double.isNaN(firstLoss)) {
                firstLoss = loss;
            }
            lastLoss = loss;
        }

        /** The scheduler's log step multiplier for the current step, advancing its LSTM. */
        double schedule() {
            double rel = Double.isNaN(lastLoss) ? 0 : Math.log((lastLoss + 1e-12) / (firstLoss + 1e-12));
            double prog = steps / (double) horizon;
            return lstmStep(sched, schedState, rel, prog);
        }

        @Override
        public void step(List<Var> parameters) {
            double logMul = schedule();
            steps++;
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                double[] theta = p.value().toDoubleArray();
                double[] gv = g.toDoubleArray();
                double[][] s = slots.computeIfAbsent(p, q -> new double[3][theta.length]);
                update(rule, theta, gv, s, steps, logMul + Math.log(multiplier));
                float[] f = new float[theta.length];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) theta[i];
                }
                p.assign(Tensor.of(p.value().shape(), f));
                p.zeroGrad();
            }
        }

        @Override
        public void forget(Var parameter) {
            slots.remove(parameter);
        }

        @Override
        public float learningRate() {
            return multiplier;
        }

        @Override
        public void learningRate(float learningRate) {
            this.multiplier = learningRate;
        }
    }

    /**
     * One rule application to a parameter tensor (in place). Slots: momentum 0.9, momentum 0.99,
     * second moment 0.999.
     */
    static void update(double[] rule, double[] theta, double[] g, double[][] slot, long t, double logMultiplier) {
        int n = theta.length;
        double bc1 = 1 - Math.pow(0.9, t);
        double bc2 = 1 - Math.pow(0.99, t);
        double bc3 = 1 - Math.pow(0.999, t);
        double rms = 0;
        for (double v : theta) {
            rms += v * v / n;
        }
        double tensorScale = Math.max(Math.sqrt(rms), 0.1);
        double base = Math.exp(rule[CeloLite.RULE_SIZE - 1] + logMultiplier) * tensorScale;
        int b1 = CeloLite.FEATURES * CeloLite.HIDDEN;
        int w2 = b1 + CeloLite.HIDDEN;
        int b2 = w2 + CeloLite.HIDDEN * 2;
        double[] x = new double[CeloLite.FEATURES];
        double[] h = new double[CeloLite.HIDDEN];
        for (int i = 0; i < n; i++) {
            slot[0][i] = 0.9 * slot[0][i] + 0.1 * g[i];
            slot[1][i] = 0.99 * slot[1][i] + 0.01 * g[i];
            slot[2][i] = 0.999 * slot[2][i] + 0.001 * g[i] * g[i];
            double rho = Math.sqrt(slot[2][i] / bc3) + 1e-8;
            x[0] = g[i] / rho;
            x[1] = slot[0][i] / bc1 / rho;
            x[2] = slot[1][i] / bc2 / rho;
            for (int k = 0; k < CeloLite.HIDDEN; k++) {
                double a = rule[b1 + k];
                for (int f = 0; f < CeloLite.FEATURES; f++) {
                    a += rule[f * CeloLite.HIDDEN + k] * Math.max(-5, Math.min(5, x[f]));
                }
                h[k] = Math.tanh(a);
            }
            double dir = rule[b2];
            double mag = rule[b2 + 1];
            for (int k = 0; k < CeloLite.HIDDEN; k++) {
                dir += h[k] * rule[w2 + 2 * k];
                mag += h[k] * rule[w2 + 2 * k + 1];
            }
            double step = (x[1] + dir) * Math.exp(Math.max(-3, Math.min(3, mag)));
            theta[i] -= base * step;
        }
    }

    /** A one-layer LSTM scheduler step on (relative log loss, progress); returns the log multiplier. */
    static double lstmStep(double[] p, double[] state, double rel, double prog) {
        int hd = CeloLite.SCHED_HIDDEN;
        int wh = 2 * 4 * hd;
        int bb = wh + hd * 4 * hd;
        int wo = bb + 4 * hd;
        double[] gates = new double[4 * hd];
        double[] in = {Math.max(-5, Math.min(5, rel)), prog};
        for (int k = 0; k < 4 * hd; k++) {
            double a = p[bb + k] + (k >= hd && k < 2 * hd ? 1 : 0);
            for (int j = 0; j < 2; j++) {
                a += p[j * 4 * hd + k] * in[j];
            }
            for (int j = 0; j < hd; j++) {
                a += p[wh + j * 4 * hd + k] * state[j];
            }
            gates[k] = a;
        }
        double out = p[wo + hd];
        for (int k = 0; k < hd; k++) {
            double ig = 1 / (1 + Math.exp(-gates[k]));
            double fg = 1 / (1 + Math.exp(-gates[hd + k]));
            double og = 1 / (1 + Math.exp(-gates[2 * hd + k]));
            double cg = Math.tanh(gates[3 * hd + k]);
            state[hd + k] = fg * state[hd + k] + ig * cg;
            state[k] = og * Math.tanh(state[hd + k]);
            out += p[wo + k] * state[k];
        }
        return Math.max(-4, Math.min(4, out));
    }

    /**
     * Runs a Celo-lite optimiser on a task for some steps from a given state.
     *
     * @return the mean loss over the steps
     */
    private static double unroll(double[] rule, double[] sched, Particle pt, int steps) {
        double total = 0;
        double[] g = new double[pt.theta.length];
        for (int s = 0; s < steps; s++) {
            double[] th = new double[pt.theta.length];
            for (int i = 0; i < th.length; i++) {
                th[i] = pt.theta[i] * pt.scale[i];
            }
            double loss = pt.task.lossAndGradient(th, g);
            if (!Double.isFinite(loss)) {
                return 1e3;
            }
            if (Double.isNaN(pt.first)) {
                pt.first = loss;
            }
            for (int i = 0; i < g.length; i++) {
                g[i] *= pt.scale[i];
            }
            double rel = Math.log((loss + 1e-12) / (pt.first + 1e-12));
            double logMul = lstmStep(sched, pt.sched, rel, pt.t / (double) pt.horizon);
            pt.t++;
            update(rule, pt.theta, g, pt.slots, pt.t, logMul);
            total += Math.min(loss / pt.first, 10) / steps;   // normalised so tasks weigh alike
        }
        return total;
    }

    /** The persistent inner state of one PES particle. */
    private static final class Particle {
        final Task task;
        final double[] theta;
        final double[] scale;
        final double[][] slots;
        final double[] sched = new double[2 * CeloLite.SCHED_HIDDEN];
        final int horizon;
        double first = Double.NaN;
        long t;

        Particle(Task task, double[] theta, double[] scale, int horizon) {
            this.task = task;
            this.theta = theta;
            this.scale = scale;
            this.slots = new double[3][theta.length];
            this.horizon = horizon;
        }

        Particle copy() {
            Particle p = new Particle(task, theta.clone(), scale, horizon);
            for (int k = 0; k < 3; k++) {
                System.arraycopy(slots[k], 0, p.slots[k], 0, theta.length);
            }
            System.arraycopy(sched, 0, p.sched, 0, sched.length);
            p.first = first;
            p.t = t;
            return p;
        }
    }

    /**
     * Meta-trains a Celo-lite optimiser by antithetic persistent evolution strategies: the rule
     * first (scheduler fixed at zero output), then the scheduler (rule fixed).
     *
     * @param tasks task factories, sampled uniformly per horizon
     * @param augment rescale parameters by {@code exp(U(-1, 1))} per coordinate
     * @param iterations meta-iterations per stage
     * @param pairs antithetic pairs
     * @param horizon inner steps per task
     * @param truncation inner steps per meta-iteration
     * @param sigma the perturbation scale
     * @param metaLearningRate Adam's step on the meta-parameters
     * @param seed the seed
     * @return the trained optimiser
     */
    public static CeloLite metaTrainCelo(List<java.util.function.Function<SplittableRandom, Task>> tasks,
            boolean augment, int iterations, int pairs, int horizon, int truncation, double sigma,
            double metaLearningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        CeloLite init = CeloLite.create();
        double[] rule = init.rule();
        double[] sched = init.scheduler();
        for (int stage = 0; stage < 2; stage++) {
            double[] phi = stage == 0 ? rule : sched;
            double[] m = new double[phi.length];
            double[] v = new double[phi.length];
            List<Particle[]> particles = new ArrayList<>();
            double[][] xi = new double[pairs][phi.length];
            for (int it = 0; it < iterations; it++) {
                if (it * truncation % horizon == 0) {
                    particles.clear();
                    for (int k = 0; k < pairs; k++) {
                        Task task = tasks.get(r.nextInt(tasks.size())).apply(r);
                        double[] scale = new double[task.dimension()];
                        double[] th = task.init(r);
                        for (int i = 0; i < scale.length; i++) {
                            scale[i] = augment ? Math.exp(r.nextDouble(-1, 1)) : 1;
                            th[i] /= scale[i];
                        }
                        Particle base = new Particle(task, th, scale, horizon);
                        particles.add(new Particle[] {base, base.copy()});
                        java.util.Arrays.fill(xi[k], 0);
                    }
                }
                double[] grad = new double[phi.length];
                for (int k = 0; k < pairs; k++) {
                    double[] eps = new double[phi.length];
                    double[] plus = new double[phi.length];
                    double[] minus = new double[phi.length];
                    for (int j = 0; j < phi.length; j++) {
                        eps[j] = sigma * r.nextGaussian();
                        plus[j] = phi[j] + eps[j];
                        minus[j] = phi[j] - eps[j];
                        xi[k][j] += eps[j];
                    }
                    Particle[] pair = particles.get(k);
                    double lp = stage == 0 ? unroll(plus, sched, pair[0], truncation)
                            : unroll(rule, plus, pair[0], truncation);
                    double lm = stage == 0 ? unroll(minus, sched, pair[1], truncation)
                            : unroll(rule, minus, pair[1], truncation);
                    for (int j = 0; j < phi.length; j++) {
                        grad[j] += xi[k][j] * (lp - lm) / (2 * pairs * sigma * sigma);
                    }
                }
                for (int j = 0; j < phi.length; j++) {
                    double gj = Math.max(-10, Math.min(10, grad[j]));
                    m[j] = 0.9 * m[j] + 0.1 * gj;
                    v[j] = 0.999 * v[j] + 0.001 * gj * gj;
                    phi[j] -= metaLearningRate * (m[j] / (1 - Math.pow(0.9, it + 1)))
                            / (Math.sqrt(v[j] / (1 - Math.pow(0.999, it + 1))) + 1e-8);
                }
            }
        }
        return CeloLite.of(rule, sched);
    }
}
