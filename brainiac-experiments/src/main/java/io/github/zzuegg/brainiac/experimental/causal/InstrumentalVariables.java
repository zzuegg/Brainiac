package io.github.zzuegg.brainiac.experimental.causal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GaussianHead;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Instrumental-variable estimators of a structural response {@code h(t, x)} when an unobserved
 * confounder drives both the treatment {@code t} and the outcome {@code y}. An instrument {@code z}
 * moves the treatment but affects the outcome only through it, so
 * {@code E[y | x, z] = E_{t ~ p(t | x, z)}[h(t, x)]}: fitting {@code h} through the first-stage
 * distribution of {@code t} removes the confounding that regressing {@code y} on {@code t} keeps.
 * <ul>
 *   <li>{@link #twoStageLeastSquares}: linear 2SLS (Angrist and Pischke): regress {@code t} on
 *       {@code (x, z)}, then {@code y} on {@code (t-hat, x)}.</li>
 *   <li>{@link DeepIv} (Hartford et al., ICML 2017, <a href="https://arxiv.org/abs/1612.09596">
 *       arXiv:1612.09596</a>): stage one fits a mixture density network for {@code p(t | x, z)}
 *       (mixture components are {@link GaussianHead}s over a shared MLP); stage two trains an MLP
 *       {@code h(t, x)} so that the Monte Carlo average of {@code h} over sampled treatments
 *       matches {@code y}.</li>
 *   <li>{@link DeepIv#fitDirect}: the same outcome network trained on the observed treatment (the
 *       confounded baseline).</li>
 * </ul>
 */
public final class InstrumentalVariables {

    private InstrumentalVariables() {
    }

    /**
     * A fitted linear structural function {@code h(t, x) = b0 + bt t + bx . x}.
     *
     * @param coefficients {@code {b0, bt, bx...}}
     */
    public record Linear(double[] coefficients) {

        /**
         * @param t the treatment
         * @param x the covariates
         * @return {@code h(t, x)}
         */
        public double predict(double t, double[] x) {
            double s = coefficients[0] + coefficients[1] * t;
            for (int j = 0; j < x.length; j++) {
                s += coefficients[2 + j] * x[j];
            }
            return s;
        }

        /** @return the treatment coefficient */
        public double effect() {
            return coefficients[1];
        }
    }

    /**
     * Two-stage least squares.
     *
     * @param x exogenous covariates {@code [n][d]} (may be empty rows)
     * @param z instruments {@code [n][k]}
     * @param t the endogenous treatment
     * @param y the outcome
     * @return the structural function
     */
    public static Linear twoStageLeastSquares(double[][] x, double[][] z, double[] t, double[] y) {
        int n = t.length;
        double[][] first = new double[n][];
        for (int i = 0; i < n; i++) {
            first[i] = concat(x[i], z[i]);
        }
        double[] tHat = Regressor.ridge(1e-8).fit(first, t, null).predict(first);
        double[][] second = new double[n][];
        for (int i = 0; i < n; i++) {
            second[i] = concat(new double[] {tHat[i]}, x[i]);
        }
        int d = second[0].length + 1;
        double[][] a = new double[d][d];
        double[] b = new double[d];
        for (int i = 0; i < n; i++) {
            double[] r = concat(new double[] {1}, second[i]);
            for (int p = 0; p < d; p++) {
                b[p] += r[p] * y[i];
                for (int q = 0; q < d; q++) {
                    a[p][q] += r[p] * r[q];
                }
            }
        }
        for (int p = 0; p < d; p++) {
            a[p][p] += 1e-8;
        }
        return new Linear(Regressor.solve(a, b));
    }

    private static double[] concat(double[] a, double[] b) {
        double[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * DeepIV: a mixture-density first stage and a Monte Carlo second stage. Inputs and outputs are
     * standardised internally.
     */
    public static final class DeepIv {

        private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2 * Math.PI));
        private final int components;
        private final int samples;
        private final Model trunk;
        private final List<GaussianHead> heads = new ArrayList<>();
        private final Var mixWeight;
        private final Var mixBias;
        private final Model outcome;
        private final long seed;
        private Scaler xs;
        private Scaler zs;
        private Scaler ts;
        private Scaler ys;

        /**
         * Creates untrained networks.
         *
         * @param covariates the width of {@code x}
         * @param instruments the width of {@code z}
         * @param components mixture components
         * @param hidden the hidden width of both networks
         * @param samples Monte Carlo treatment samples per row in stage two
         * @param seed the seed
         */
        public DeepIv(int covariates, int instruments, int components, int hidden, int samples, long seed) {
            this.components = components;
            this.samples = samples;
            this.seed = seed;
            SplittableRandom r = new SplittableRandom(seed);
            trunk = Model.sequential().input(covariates + instruments).dense(hidden, Activation.RELU)
                    .dense(hidden, Activation.RELU).seed(seed).build();
            for (int k = 0; k < components; k++) {
                heads.add(new GaussianHead(hidden, r));
            }
            mixWeight = Var.param(Tensor.normal(Shape.of(hidden, components), 0, 0.1f, r));
            mixBias = Var.param(Tensor.zeros(Shape.of(1, components)));
            outcome = Model.sequential().input(1 + covariates).dense(hidden, Activation.RELU)
                    .dense(hidden, Activation.RELU).dense(1).seed(seed + 1).build();
        }

        private List<Var> stageOneParameters() {
            List<Var> p = new ArrayList<>(trunk.parameters());
            for (GaussianHead h : heads) {
                p.addAll(h.parameters());
            }
            p.add(mixWeight);
            p.add(mixBias);
            return p;
        }

        /** @return the outcome network {@code h(t, x)} on standardised inputs */
        public Model outcomeNetwork() {
            return outcome;
        }

        /**
         * The mixture's parameters for a batch.
         *
         * @param input standardised {@code [batch, x + z]}
         * @return {@code {logWeights [b, K], means [b, K], stds [b, K]}}
         */
        private Var[] mixture(Var input, boolean training) {
            Var h = trunk.forward(input, training);
            Var logits = h.matmul(mixWeight).add(mixBias);
            Var logPi = logits.sub(logits.logsumexp(1, true));
            List<Var> means = new ArrayList<>();
            List<Var> stds = new ArrayList<>();
            for (GaussianHead g : heads) {
                Var o = g.forward(h, training);
                means.add(o.slice(1, 0, 1));
                stds.add(o.slice(1, 1, 2));
            }
            return new Var[] {logPi, Var.concat(1, means.toArray(Var[]::new)), Var.concat(1, stds.toArray(Var[]::new))};
        }

        /**
         * Fits both stages.
         *
         * @param x covariates
         * @param z instruments
         * @param t treatments
         * @param y outcomes
         * @param stageOneSteps Adam steps for the mixture density network
         * @param stageTwoSteps Adam steps for the outcome network
         * @return this
         */
        public DeepIv fit(double[][] x, double[][] z, double[] t, double[] y, int stageOneSteps, int stageTwoSteps) {
            scale(x, z, t, y);
            int n = t.length;
            SplittableRandom r = new SplittableRandom(seed + 2);
            int batch = Math.min(128, n);
            Optimizer opt1 = Optimizer.adam(3e-3f);
            for (int s = 0; s < stageOneSteps; s++) {
                int[] rows = rows(n, batch, r);
                Var[] m = mixture(Var.of(stageOneInput(x, z, rows)), true);
                Tensor target = column(t, rows, ts);
                Var zt = m[1].neg().add(target).div(m[2]);
                Var logN = zt.square().mul(-0.5f).sub(m[2].log()).sub(HALF_LOG_TWO_PI);
                m[0].add(logN).logsumexp(1, true).mean().neg().backward();
                opt1.step(stageOneParameters());
            }
            Optimizer opt2 = Optimizer.adam(3e-3f);
            int d = x[0].length;
            for (int s = 0; s < stageTwoSteps; s++) {
                int[] rows = rows(n, batch, r);
                Var[] m = Var.noGrad(() -> mixture(Var.of(stageOneInput(x, z, rows)), false));
                float[] logPi = m[0].value().toArray();
                float[] mu = m[1].value().toArray();
                float[] sd = m[2].value().toArray();
                float[] in = new float[batch * samples * (1 + d)];
                for (int i = 0; i < batch; i++) {
                    for (int k = 0; k < samples; k++) {
                        int c = sampleComponent(logPi, i, r);
                        double tt = mu[i * components + c] + sd[i * components + c] * r.nextGaussian();
                        int o = (i * samples + k) * (1 + d);
                        in[o] = (float) tt;
                        for (int j = 0; j < d; j++) {
                            in[o + 1 + j] = (float) xs.apply(x[rows[i]][j], j);
                        }
                    }
                }
                Var h = outcome.forward(Var.of(Tensor.of(Shape.of(batch * samples, 1 + d), in)), true)
                        .reshape(batch, samples).mean(1, true);
                h.sub(column(y, rows, ys)).square().mean().backward();
                opt2.step(outcome.parameters());
            }
            return this;
        }

        /**
         * Trains the outcome network directly on the observed treatment (no instrument): the
         * confounded baseline with the same architecture and budget.
         *
         * @param x covariates
         * @param t treatments
         * @param y outcomes
         * @param steps Adam steps
         * @return this
         */
        public DeepIv fitDirect(double[][] x, double[] t, double[] y, int steps) {
            double[][] z = new double[t.length][1];
            scale(x, z, t, y);
            int n = t.length;
            int d = x[0].length;
            SplittableRandom r = new SplittableRandom(seed + 3);
            int batch = Math.min(128, n);
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                int[] rows = rows(n, batch, r);
                float[] in = new float[batch * (1 + d)];
                for (int i = 0; i < batch; i++) {
                    in[i * (1 + d)] = (float) ts.apply(t[rows[i]], 0);
                    for (int j = 0; j < d; j++) {
                        in[i * (1 + d) + 1 + j] = (float) xs.apply(x[rows[i]][j], j);
                    }
                }
                outcome.forward(Var.of(Tensor.of(Shape.of(batch, 1 + d), in)), true).sub(column(y, rows, ys))
                        .square().mean().backward();
                opt.step(outcome.parameters());
            }
            return this;
        }

        /**
         * The structural prediction {@code h(t, x)} in original units.
         *
         * @param t treatments
         * @param x covariates
         * @return one prediction per row
         */
        public double[] predict(double[] t, double[][] x) {
            int n = t.length;
            int d = x[0].length;
            float[] in = new float[n * (1 + d)];
            for (int i = 0; i < n; i++) {
                in[i * (1 + d)] = (float) ts.apply(t[i], 0);
                for (int j = 0; j < d; j++) {
                    in[i * (1 + d) + 1 + j] = (float) xs.apply(x[i][j], j);
                }
            }
            float[] out = Var.noGrad(() -> outcome.forward(Var.of(Tensor.of(Shape.of(n, 1 + d), in)), false))
                    .value().toArray();
            double[] p = new double[n];
            for (int i = 0; i < n; i++) {
                p[i] = ys.invert(out[i]);
            }
            return p;
        }

        private int sampleComponent(float[] logPi, int row, SplittableRandom r) {
            double u = r.nextDouble();
            double acc = 0;
            for (int c = 0; c < components; c++) {
                acc += Math.exp(logPi[row * components + c]);
                if (u <= acc) {
                    return c;
                }
            }
            return components - 1;
        }

        private void scale(double[][] x, double[][] z, double[] t, double[] y) {
            xs = Scaler.of(x);
            zs = Scaler.of(z);
            double[][] tc = new double[t.length][];
            double[][] yc = new double[y.length][];
            for (int i = 0; i < t.length; i++) {
                tc[i] = new double[] {t[i]};
                yc[i] = new double[] {y[i]};
            }
            ts = Scaler.of(tc);
            ys = Scaler.of(yc);
        }

        private Tensor stageOneInput(double[][] x, double[][] z, int[] rows) {
            int d = x[0].length;
            int k = z[0].length;
            float[] f = new float[rows.length * (d + k)];
            for (int i = 0; i < rows.length; i++) {
                for (int j = 0; j < d; j++) {
                    f[i * (d + k) + j] = (float) xs.apply(x[rows[i]][j], j);
                }
                for (int j = 0; j < k; j++) {
                    f[i * (d + k) + d + j] = (float) zs.apply(z[rows[i]][j], j);
                }
            }
            return Tensor.of(Shape.of(rows.length, d + k), f);
        }

        private static Tensor column(double[] v, int[] rows, Scaler s) {
            float[] f = new float[rows.length];
            for (int i = 0; i < rows.length; i++) {
                f[i] = (float) s.apply(v[rows[i]], 0);
            }
            return Tensor.of(Shape.of(rows.length, 1), f);
        }

        private static int[] rows(int n, int batch, SplittableRandom r) {
            int[] rows = new int[batch];
            for (int i = 0; i < batch; i++) {
                rows[i] = r.nextInt(n);
            }
            return rows;
        }
    }

    /** Per-column standardisation. */
    private record Scaler(double[] mean, double[] sd) {

        static Scaler of(double[][] x) {
            int d = x[0].length;
            double[] m = new double[d];
            double[] s = new double[d];
            for (double[] row : x) {
                for (int j = 0; j < d; j++) {
                    m[j] += row[j] / x.length;
                }
            }
            for (double[] row : x) {
                for (int j = 0; j < d; j++) {
                    s[j] += (row[j] - m[j]) * (row[j] - m[j]) / x.length;
                }
            }
            for (int j = 0; j < d; j++) {
                s[j] = Math.sqrt(s[j]) + 1e-8;
            }
            return new Scaler(m, s);
        }

        double apply(double v, int j) {
            return (v - mean[j]) / sd[j];
        }

        double invert(double v) {
            return v * sd[0] + mean[0];
        }
    }
}
