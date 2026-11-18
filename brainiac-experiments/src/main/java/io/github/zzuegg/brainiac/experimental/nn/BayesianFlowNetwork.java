package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Bayesian flow networks (Graves et al., <a href="https://arxiv.org/abs/2308.07037">arXiv:2308.07037</a>):
 * generation as Bayesian inference on the <em>parameters</em> of a factorised distribution.
 * A sender reveals the data through ever less noisy observations; a receiver keeps a
 * posterior over each variable, updated in closed form; the network sees only that posterior
 * (never a noisy sample) and predicts the data, and its prediction defines the next
 * observation's distribution. Because the network's input is a set of continuous parameters
 * even for discrete data, one recipe covers both:
 * <ul>
 *   <li>{@link #discrete}: each variable is a categorical over {@code K} classes; the
 *       posterior is the probability vector {@code theta = softmax(y)} with
 *       {@code y ~ N(beta(t) (K e_x - 1), beta(t) K I)}, {@code beta(t) = beta1 t^2}, and the
 *       continuous-time loss is {@code K beta1 t |e_x - e_hat|^2};</li>
 *   <li>{@link #continuous}: each variable is real-valued with a Gaussian posterior mean
 *       {@code mu ~ N(gamma x, gamma (1 - gamma))}, {@code gamma(t) = 1 - sigma1^(2t)}; the
 *       network predicts the noise; the bound's loss is {@code -ln(sigma1) sigma1^(-2t) |x - x_hat|^2},
 *       trained here in its unweighted form {@code |x - x_hat|^2} (as DDPM's simple loss).</li>
 * </ul>
 * Both losses bound the negative log-likelihood (in the continuous-time limit), and sampling
 * with {@code n} steps runs the same Bayesian updates with the network's own predictions as
 * the observed data.
 */
public final class BayesianFlowNetwork {
    private static final int TIME_HALF = 8;
    private final boolean discrete;
    private final int dim;
    private final int classes;
    private final float precision;
    private final GenerativeOps.Mlp net;

    private BayesianFlowNetwork(boolean discrete, int dim, int classes, float precision, int width, long seed) {
        this.discrete = discrete;
        this.dim = dim;
        this.classes = classes;
        this.precision = precision;
        int in = discrete ? dim * classes : dim;
        net = new GenerativeOps.Mlp(new SplittableRandom(seed), false, in + 2 * TIME_HALF, width, width, width, in);
    }

    /**
     * A BFN for {@code dim} categorical variables.
     *
     * @param dim the number of variables
     * @param classes the classes per variable {@code K}
     * @param width the network width
     * @param beta1 the final accuracy {@code beta(1)}, about 3 for binary data
     * @param seed the initialization seed
     * @return the model
     */
    public static BayesianFlowNetwork discrete(int dim, int classes, int width, float beta1, long seed) {
        return new BayesianFlowNetwork(true, dim, classes, beta1, width, seed);
    }

    /**
     * A BFN for {@code dim} real variables.
     *
     * @param dim the number of variables
     * @param width the network width
     * @param sigma1 the final posterior standard deviation, such as 0.02
     * @param seed the initialization seed
     * @return the model
     */
    public static BayesianFlowNetwork continuous(int dim, int width, float sigma1, long seed) {
        return new BayesianFlowNetwork(false, dim, 1, sigma1, width, seed);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return net.parameters();
    }

    private Var output(Var input, float[] t) {
        return net.apply(Var.concat(1, input, Var.of(GenerativeOps.timeFeatures(t, TIME_HALF, 30))));
    }

    /** Softmax over each variable's classes: {@code [n, dim * K] -> [n, dim, K]}. */
    private Var classProbabilities(Var logits) {
        int n = logits.shape().dim(0);
        Var l = logits.reshape(n, dim, classes);
        Var e = l.sub(Var.of(l.value().max(2, true))).exp();
        return e.div(e.sum(2, true));
    }

    /**
     * The continuous-time loss for categorical data.
     *
     * @param x class indices {@code [n][dim]}
     * @param random the source of times and observations
     * @return the scalar loss (nats per sequence, averaged over the batch)
     */
    public Var discreteLoss(int[][] x, RandomGenerator random) {
        int n = x.length;
        float[] t = new float[n];
        float[] theta = new float[n * dim * classes];
        float[] onehot = new float[n * dim * classes];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) random.nextDouble();
            double beta = precision * t[i] * t[i];
            weight[i] = classes * precision * t[i];
            for (int d = 0; d < dim; d++) {
                int base = (i * dim + d) * classes;
                onehot[base + x[i][d]] = 1;
                double[] y = new double[classes];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < classes; k++) {
                    double mean = beta * (classes * onehot[base + k] - 1);
                    y[k] = mean + Math.sqrt(beta * classes) * random.nextGaussian();
                    max = Math.max(max, y[k]);
                }
                softmaxInto(y, max, theta, base);
            }
        }
        Var p = classProbabilities(output(Var.of(Tensor.of(Shape.of(n, dim * classes), theta)), t));
        Var err = p.sub(Var.of(Tensor.of(Shape.of(n, dim, classes), onehot))).square().sum(2, false).sum(1, true);
        return err.mul(GenerativeOps.column(weight)).mean();
    }

    private void softmaxInto(double[] y, double max, float[] out, int base) {
        double sum = 0;
        for (int k = 0; k < classes; k++) {
            sum += Math.exp(y[k] - max);
        }
        for (int k = 0; k < classes; k++) {
            out[base + k] = (float) (Math.exp(y[k] - max) / sum);
        }
    }

    /**
     * Generates categorical samples with {@code steps} Bayesian updates.
     *
     * @param count how many
     * @param steps the number of updates (network evaluations)
     * @param random the random source
     * @return class indices {@code [count][dim]}
     */
    public int[][] sampleDiscrete(int count, int steps, RandomGenerator random) {
        float[] theta = new float[count * dim * classes];
        java.util.Arrays.fill(theta, 1f / classes);
        for (int s = 1; s <= steps; s++) {
            float[] t = new float[count];
            java.util.Arrays.fill(t, (float) (s - 1) / steps);
            float[] p = classProbabilities(output(Var.of(Tensor.of(Shape.of(count, dim * classes), theta)), t))
                    .value().toArray();
            double alpha = precision * (2.0 * s - 1) / ((double) steps * steps);
            for (int i = 0; i < count * dim; i++) {
                int base = i * classes;
                int k = draw(p, base, random);
                double[] y = new double[classes];
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < classes; c++) {
                    y[c] = alpha * (classes * (c == k ? 1 : 0) - 1) + Math.sqrt(alpha * classes) * random.nextGaussian()
                            + Math.log(Math.max(theta[base + c], 1e-30));
                    max = Math.max(max, y[c]);
                }
                softmaxInto(y, max, theta, base);
            }
        }
        float[] t = new float[count];
        java.util.Arrays.fill(t, 1f);
        float[] p = classProbabilities(output(Var.of(Tensor.of(Shape.of(count, dim * classes), theta)), t)).value()
                .toArray();
        int[][] out = new int[count][dim];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < dim; d++) {
                out[i][d] = draw(p, (i * dim + d) * classes, random);
            }
        }
        return out;
    }

    private int draw(float[] p, int base, RandomGenerator random) {
        double u = random.nextDouble();
        double c = 0;
        for (int k = 0; k < classes; k++) {
            c += p[base + k];
            if (u < c) {
                return k;
            }
        }
        return classes - 1;
    }

    /** The clean estimate {@code mu / gamma - sqrt((1 - gamma) / gamma) eps_hat}, clipped to [-1, 1]. */
    private Var estimate(Var mu, float[] t) {
        int n = t.length;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            double gamma = Math.max(1e-6, 1 - Math.pow(precision, 2 * t[i]));
            a[i] = (float) (1 / gamma);
            b[i] = (float) Math.sqrt((1 - gamma) / gamma);
        }
        Var eps = output(mu, t);
        return mu.mul(GenerativeOps.column(a)).sub(eps.mul(GenerativeOps.column(b)));
    }

    /**
     * The continuous-time loss for real data, without the bound's time weighting.
     *
     * @param x data {@code [n, dim]}, roughly in [-1, 1]
     * @param random the source of times and observations
     * @return the scalar loss
     */
    public Var continuousLoss(Tensor x, RandomGenerator random) {
        int n = x.shape().dim(0);
        float[] t = new float[n];
        float[] mean = new float[n];
        float[] std = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) Math.min(random.nextDouble(), 1 - 1e-4);
            double gamma = 1 - Math.pow(precision, 2 * t[i]);
            mean[i] = (float) gamma;
            std[i] = (float) Math.sqrt(gamma * (1 - gamma));
        }
        Tensor mu = x.mul(GenerativeOps.column(mean)).add(Tensor.normal(x.shape(), 0, 1, random)
                .mul(GenerativeOps.column(std)));
        Var xhat = estimate(Var.of(mu), t);
        // the unweighted ("simple") form: the bound's sigma1^(-2t) weight puts nearly all of the
        // gradient on t near 1 and trained markedly worse on small budgets
        return xhat.sub(Var.of(x)).square().sum(1, true).mean();
    }

    /**
     * Generates real samples with {@code steps} Bayesian updates.
     *
     * @param count how many
     * @param steps the number of updates
     * @param random the random source
     * @return {@code [count, dim]}
     */
    public Tensor sampleContinuous(int count, int steps, RandomGenerator random) {
        Tensor mu = Tensor.zeros(Shape.of(count, dim));
        double rho = 1;
        for (int s = 1; s <= steps; s++) {
            float[] t = new float[count];
            java.util.Arrays.fill(t, (float) (s - 1) / steps);
            Tensor xhat = clip(estimate(Var.of(mu), t).value());
            double alpha = Math.pow(precision, -2.0 * s / steps) * (1 - Math.pow(precision, 2.0 / steps));
            Tensor y = xhat.add(Tensor.normal(xhat.shape(), 0, (float) (1 / Math.sqrt(alpha)), random));
            mu = mu.mul((float) (rho / (rho + alpha))).add(y.mul((float) (alpha / (rho + alpha))));
            rho += alpha;
        }
        float[] t = new float[count];
        java.util.Arrays.fill(t, 1f);
        return clip(estimate(Var.of(mu), t).value());
    }

    private static Tensor clip(Tensor x) {
        float[] v = x.toArray();
        for (int i = 0; i < v.length; i++) {
            v[i] = Math.max(-1, Math.min(1, v[i]));
        }
        return Tensor.of(x.shape(), v);
    }
}
