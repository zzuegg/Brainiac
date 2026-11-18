package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A trainable conditional density {@code q(y | c)}, the shared estimator under neural
 * posterior ({@code y = theta, c = x}), neural likelihood ({@code y = x, c = theta}) and
 * sequential estimation. Two families:
 * <ul>
 *   <li>{@link Kind#MDN}: a mixture density network with diagonal Gaussian components
 *       (Bishop 1994), as in {@code SimulationBasedInference.neuralPosterior};</li>
 *   <li>{@link Kind#COUPLING_FLOW}: a conditional normalising flow of affine coupling layers
 *       (Dinh et al., RealNVP, <a href="https://arxiv.org/abs/1605.08803">arXiv:1605.08803</a>)
 *       with alternating masks; each layer's scale and shift come from an MLP that sees the
 *       untouched coordinates and the condition, and its log-determinant is the sum of the
 *       (tanh-bounded) log scales.</li>
 * </ul>
 * Inputs and outputs are standardised by the statistics of the first training set, which stay
 * fixed when training continues (so sequential rounds refine the same model). Densities
 * reported in data units include the standardisation's Jacobian.
 */
public final class ConditionalDensity {

    /** The density family. */
    public enum Kind {
        /** Mixture density network with diagonal Gaussian components. */
        MDN,
        /** Conditional affine-coupling normalising flow. */
        COUPLING_FLOW
    }

    /**
     * Training settings.
     *
     * @param components mixture components (MDN) or coupling layers (flow)
     * @param hidden hidden width
     * @param iterations Adam steps per call to {@link #train}
     * @param batch minibatch size
     * @param learningRate Adam learning rate
     */
    public record Config(int components, int hidden, int iterations, int batch, double learningRate) {

        /**
         * A small default: 4 components or layers, 32 hidden units, 600 steps of 128.
         *
         * @return the settings
         */
        public static Config small() {
            return new Config(4, 32, 600, 128, 3e-3);
        }

        /**
         * A copy with another number of iterations.
         *
         * @param n the iterations
         * @return the settings
         */
        public Config withIterations(int n) {
            return new Config(components, hidden, n, batch, learningRate);
        }
    }

    private static final double LOG_2PI = Math.log(2 * Math.PI);

    private final Kind kind;
    private final int dy;
    private final int components;
    private final double[] yMean;
    private final double[] ySd;
    private final double[] cMean;
    private final double[] cSd;
    private final SimpleMlp trunk;
    private final List<SimpleMlp> couplings;
    private final List<Tensor> masks;
    private final Config config;
    private final List<Var> params;
    private final Optimizer optimizer;

    private ConditionalDensity(Kind kind, double[][] y, double[][] c, Config config, long seed) {
        this.kind = kind;
        this.config = config;
        this.dy = y[0].length;
        int dc = c[0].length;
        this.components = config.components();
        yMean = SbiArrays.mean(y);
        ySd = SbiArrays.sd(y, yMean);
        cMean = SbiArrays.mean(c);
        cSd = SbiArrays.sd(c, cMean);
        params = new ArrayList<>();
        couplings = new ArrayList<>();
        masks = new ArrayList<>();
        if (kind == Kind.MDN) {
            trunk = SimpleMlp.tanh(seed, dc, config.hidden(), config.hidden(), components * (1 + 2 * dy));
            // spread the initial means so the components do not start identical
            SplittableRandom r = new SplittableRandom(seed + 1);
            Var b = trunk.bias(trunk.depth() - 1);
            float[] bias = new float[components * (1 + 2 * dy)];
            for (int i = components; i < components + components * dy; i++) {
                bias[i] = (float) r.nextGaussian();
            }
            b.assign(Tensor.of(Shape.of(1, bias.length), bias));
            Var w = trunk.weight(trunk.depth() - 1);
            w.assign(w.value().mul(0.1f));
            params.addAll(trunk.parameters());
        } else {
            trunk = null;
            for (int l = 0; l < components; l++) {
                SimpleMlp net = SimpleMlp.tanh(seed + 17L * l, dy + dc, config.hidden(), config.hidden(), 2 * dy);
                Var w = net.weight(net.depth() - 1);
                w.assign(w.value().mul(0.01f));
                couplings.add(net);
                params.addAll(net.parameters());
                float[] m = new float[dy];
                for (int j = 0; j < dy; j++) {
                    m[j] = (j + l) % 2 == 0 ? 1f : 0f;
                }
                if (dy == 1) {
                    m[0] = 0f;
                }
                masks.add(Tensor.of(Shape.of(1, dy), m));
            }
        }
        optimizer = Optimizer.adam((float) config.learningRate());
    }

    /**
     * Fits a new conditional density by maximum likelihood.
     *
     * @param kind the family
     * @param y the modelled variables {@code [n][dy]}
     * @param c the conditions {@code [n][dc]}
     * @param config training settings
     * @param seed the seed
     * @return the trained density
     */
    public static ConditionalDensity fit(Kind kind, double[][] y, double[][] c, Config config, long seed) {
        Objects.requireNonNull(kind, "kind");
        ConditionalDensity q = new ConditionalDensity(kind, y, c, config, seed);
        q.train(y, c, config.iterations(), seed + 3);
        return q;
    }

    /**
     * Continues maximum-likelihood training on (possibly new) data with the fixed
     * standardisation.
     *
     * @param y the modelled variables
     * @param c the conditions
     * @param iterations Adam steps
     * @param seed the batch seed
     */
    public void train(double[][] y, double[][] c, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int batch = Math.min(config.batch(), y.length);
        for (int it = 0; it < iterations; it++) {
            int[] idx = SbiArrays.batch(y.length, batch, r);
            Var loss = logProbStandardised(Var.of(SbiArrays.standardisedRows(y, idx, yMean, ySd)),
                    SbiArrays.standardisedRows(c, idx, cMean, cSd)).mean().neg();
            loss.backward();
            optimizer.step(params);
        }
    }

    /**
     * Takes one optimiser step on a caller-built loss over {@link #parameters()}.
     *
     * @param loss the scalar loss
     */
    public void step(Var loss) {
        loss.backward();
        optimizer.step(params);
    }

    /**
     * The log density in standardised units, per row (the standardisation Jacobian is a
     * constant and left out).
     *
     * @param y standardised variables {@code [B, dy]}
     * @param c standardised conditions {@code [B, dc]}
     * @return {@code [B]}
     */
    public Var logProbStandardised(Var y, Tensor c) {
        int b = c.shape().dim(0);
        if (kind == Kind.MDN) {
            Var o = trunk.forward(Var.of(c));
            int k = components;
            int kd = k * dy;
            Var logits = o.slice(1, 0, k);
            Var logPi = logits.sub(logits.logsumexp(1, true));
            Var mu = o.slice(1, k, k + kd).reshape(b, k, dy);
            Var ls = o.slice(1, k + kd, k + 2 * kd).clamp(-7f, 3f).reshape(b, k, dy);
            Var z = y.reshape(b, 1, dy).sub(mu).mul(ls.neg().exp());
            Var comp = z.square().mul(-0.5f).sub(ls).sum(2, false).sub((float) (0.5 * dy * LOG_2PI));
            return logPi.add(comp).logsumexp(1, false);
        }
        Var z = y;
        Var logDet = null;
        Var cv = Var.of(c);
        for (int l = 0; l < couplings.size(); l++) {
            Tensor m = masks.get(l);
            Tensor inv = m.mul(-1f).add(1f);
            Var st = couplings.get(l).forward(Var.concat(1, z.mul(m), cv));
            Var s = st.slice(1, 0, dy).tanh().mul(inv);
            Var t = st.slice(1, dy, 2 * dy).mul(inv);
            z = z.mul(s.exp()).add(t);
            Var ld = s.sum(1, false);
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        Var base = z.square().sum(1, false).mul(-0.5f).sub((float) (0.5 * dy * LOG_2PI));
        return logDet == null ? base : base.add(logDet);
    }

    private double jacobian() {
        double s = 0;
        for (double v : ySd) {
            s += Math.log(v);
        }
        return s;
    }

    /**
     * Log densities of rows, in data units.
     *
     * @param y variables {@code [n][dy]}
     * @param c conditions {@code [n][dc]}
     * @return {@code log q(y_i | c_i)}
     */
    public double[] logDensities(double[][] y, double[][] c) {
        Var out = Var.noGrad(() -> logProbStandardised(Var.of(SbiArrays.standardised(y, yMean, ySd)),
                SbiArrays.standardised(c, cMean, cSd)));
        double[] v = out.value().toDoubleArray();
        double j = jacobian();
        for (int i = 0; i < v.length; i++) {
            v[i] -= j;
        }
        return v;
    }

    /**
     * The log density of one point.
     *
     * @param y the variables
     * @param c the condition
     * @return {@code log q(y | c)}
     */
    public double logDensity(double[] y, double[] c) {
        return logDensities(new double[][] {y}, new double[][] {c})[0];
    }

    /**
     * Draws from {@code q(. | c)}.
     *
     * @param c the condition
     * @param n the number of draws
     * @param random the random source
     * @return {@code [n][dy]}
     */
    public double[][] sample(double[] c, int n, SplittableRandom random) {
        Tensor ct = SbiArrays.repeated(c, n, cMean, cSd);
        double[][] z = new double[n][dy];
        if (kind == Kind.MDN) {
            double[] o = Var.noGrad(() -> trunk.forward(Var.of(ct.slice(0, 0, 1)))).value().toDoubleArray();
            int k = components;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < k; i++) {
                max = Math.max(max, o[i]);
            }
            double[] p = new double[k];
            double tot = 0;
            for (int i = 0; i < k; i++) {
                p[i] = Math.exp(o[i] - max);
                tot += p[i];
            }
            for (int i = 0; i < n; i++) {
                double u = random.nextDouble() * tot;
                int comp = 0;
                while (comp < k - 1 && u > p[comp]) {
                    u -= p[comp];
                    comp++;
                }
                for (int j = 0; j < dy; j++) {
                    double ls = Math.max(-7, Math.min(3, o[k + k * dy + comp * dy + j]));
                    z[i][j] = o[k + comp * dy + j] + Math.exp(ls) * random.nextGaussian();
                }
            }
            return SbiArrays.unstandardise(z, yMean, ySd);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dy; j++) {
                z[i][j] = random.nextGaussian();
            }
        }
        Tensor cur = SbiArrays.standardised(z, new double[dy], ones(dy));
        for (int l = couplings.size() - 1; l >= 0; l--) {
            Tensor m = masks.get(l);
            Tensor inv = m.mul(-1f).add(1f);
            Tensor in = cur;
            int layer = l;
            Tensor st = Var.noGrad(() -> couplings.get(layer).forward(Var.concat(1, Var.of(in.mul(m)), Var.of(ct))))
                    .value();
            Tensor s = Var.of(st.slice(1, 0, dy)).tanh().value().mul(inv);
            Tensor t = st.slice(1, dy, 2 * dy).mul(inv);
            cur = cur.sub(t).mul(s.mul(-1f).exp());
        }
        return SbiArrays.unstandardise(SbiArrays.rows(cur), yMean, ySd);
    }

    private static double[] ones(int d) {
        double[] o = new double[d];
        java.util.Arrays.fill(o, 1);
        return o;
    }

    /**
     * Standardises variables with the fixed statistics.
     *
     * @param y variables {@code [n][dy]}
     * @return {@code [n, dy]}
     */
    public Tensor standardiseY(double[][] y) {
        return SbiArrays.standardised(y, yMean, ySd);
    }

    /**
     * Standardises conditions with the fixed statistics.
     *
     * @param c conditions {@code [n][dc]}
     * @return {@code [n, dc]}
     */
    public Tensor standardiseC(double[][] c) {
        return SbiArrays.standardised(c, cMean, cSd);
    }

    /**
     * The trainable parameters.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /**
     * The family.
     *
     * @return the kind
     */
    public Kind kind() {
        return kind;
    }
}
