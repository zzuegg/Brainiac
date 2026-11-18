package io.github.zzuegg.brainiac.experimental.estimators;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.DirichletProcessMixture;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Estimators of the mutual information {@code I(X; Y)} from paired samples.
 *
 * <ul>
 *   <li>{@link #binned}: the plug-in estimate on an equal-frequency histogram, the classical
 *       baseline, heavily biased once the joint grid outgrows the sample;</li>
 *   <li>{@link #ksg}: the k-nearest-neighbour estimator of Kraskov, Stoegbauer and Grassberger
 *       (Phys. Rev. E 2004, estimator 1) with max-norm neighbourhoods;</li>
 *   <li>{@link #variational}: lower bounds maximised over a neural critic {@code T(x, y)}
 *       (Poole et al., ICML 2019, <a href="https://arxiv.org/abs/1905.06922">arXiv:1905.06922</a>):
 *       {@link Bound#MINE} (Donsker-Varadhan with a moving-average gradient correction, Belghazi
 *       et al. 2018), {@link Bound#NWJ} (Nguyen, Wainwright and Jordan 2010),
 *       {@link Bound#INFONCE} (van den Oord et al. 2018, at most {@code log B}) and
 *       {@link Bound#SMILE} (Song and Ermon, ICLR 2020: a Jensen-Shannon-trained critic read out
 *       through a clipped Donsker-Varadhan bound).</li>
 * </ul>
 * The critic is separable, {@code T(x, y) = g(x)^T h(y)} with two one-hidden-layer MLPs, so a
 * batch of {@code B} pairs yields all {@code B^2} scores with one matrix product: the diagonal
 * holds joint samples and the off-diagonal entries product-of-marginal samples.
 */
public final class MutualInformation {

    /** A variational lower bound on mutual information. */
    public enum Bound {
        /** Donsker-Varadhan with a moving-average denominator. */
        MINE,
        /** The f-divergence (NWJ) bound {@code E_p[T] - E_q[e^(T-1)]}. */
        NWJ,
        /** The contrastive bound {@code log B + E[T_ii - logsumexp_j T_ij]}. */
        INFONCE,
        /** Clipped Donsker-Varadhan readout of a Jensen-Shannon-trained critic. */
        SMILE
    }

    private MutualInformation() {
    }

    /**
     * The mutual information of a Gaussian pair with {@code d} independent coordinate pairs of
     * correlation {@code rho}: {@code -d/2 log(1 - rho^2)}.
     *
     * @param rho the per-coordinate correlation
     * @param d the number of coordinate pairs
     * @return the mutual information in nats
     */
    public static double gaussian(double rho, int d) {
        return -0.5 * d * Math.log(1 - rho * rho);
    }

    /**
     * The per-coordinate correlation that gives a target mutual information.
     *
     * @param mi the target mutual information in nats
     * @param d the number of coordinate pairs
     * @return {@code rho}
     */
    public static double correlationFor(double mi, int d) {
        return Math.sqrt(1 - Math.exp(-2 * mi / d));
    }

    /**
     * The plug-in estimate on an equal-frequency histogram with {@code bins} bins per coordinate.
     *
     * @param x samples of {@code X} {@code [n][dx]}
     * @param y samples of {@code Y} {@code [n][dy]}
     * @param bins bins per coordinate
     * @return {@code H(X) + H(Y) - H(X, Y)} in nats
     */
    public static double binned(double[][] x, double[][] y, int bins) {
        int n = x.length;
        long[] kx = cells(x, bins);
        long[] ky = cells(y, bins);
        long[] kxy = new long[n];
        long stride = 1;
        for (int j = 0; j < x[0].length; j++) {
            stride *= bins;
        }
        for (int i = 0; i < n; i++) {
            kxy[i] = kx[i] + stride * ky[i];
        }
        return entropy(kx) + entropy(ky) - entropy(kxy);
    }

    private static long[] cells(double[][] x, int bins) {
        int n = x.length;
        int d = x[0].length;
        long[] key = new long[n];
        long stride = 1;
        for (int j = 0; j < d; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = x[i][j];
            }
            double[] sorted = col.clone();
            Arrays.sort(sorted);
            for (int i = 0; i < n; i++) {
                int rank = Arrays.binarySearch(sorted, col[i]);
                int b = Math.min(bins - 1, (int) ((long) Math.max(rank, 0) * bins / n));
                key[i] += stride * b;
            }
            stride *= bins;
        }
        return key;
    }

    private static double entropy(long[] keys) {
        Map<Long, Integer> count = new HashMap<>();
        for (long k : keys) {
            count.merge(k, 1, Integer::sum);
        }
        double h = 0;
        for (int c : count.values()) {
            double p = (double) c / keys.length;
            h -= p * Math.log(p);
        }
        return h;
    }

    /**
     * The Kraskov-Stoegbauer-Grassberger estimator (algorithm 1).
     *
     * @param x samples of {@code X} {@code [n][dx]}
     * @param y samples of {@code Y} {@code [n][dy]}
     * @param k the neighbour count, for example 3
     * @return {@code psi(k) + psi(n) - <psi(n_x + 1) + psi(n_y + 1)>} in nats
     */
    public static double ksg(double[][] x, double[][] y, int k) {
        int n = x.length;
        double[][] dx = new double[n][n];
        double[][] dy = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                dx[i][j] = dx[j][i] = maxNorm(x[i], x[j]);
                dy[i][j] = dy[j][i] = maxNorm(y[i], y[j]);
            }
        }
        double sum = 0;
        double[] joint = new double[n - 1];
        for (int i = 0; i < n; i++) {
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    joint[c++] = Math.max(dx[i][j], dy[i][j]);
                }
            }
            double[] s = joint.clone();
            Arrays.sort(s);
            double eps = s[k - 1];
            int nx = 0;
            int ny = 0;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                if (dx[i][j] < eps) {
                    nx++;
                }
                if (dy[i][j] < eps) {
                    ny++;
                }
            }
            sum += DirichletProcessMixture.digamma(nx + 1) + DirichletProcessMixture.digamma(ny + 1);
        }
        return DirichletProcessMixture.digamma(k) + DirichletProcessMixture.digamma(n) - sum / n;
    }

    private static double maxNorm(double[] a, double[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    /**
     * A variational lower bound maximised over a separable MLP critic with Adam, then evaluated on
     * the whole sample in batches of the training size.
     *
     * @param bound the bound
     * @param xRaw samples of {@code X} {@code [n][dx]}, standardised per coordinate inside
     * @param yRaw samples of {@code Y} {@code [n][dy]}, standardised per coordinate inside
     * @param hidden hidden units of each critic tower
     * @param embedding the dimension of {@code g(x)} and {@code h(y)}
     * @param batch the batch size {@code B}
     * @param steps Adam steps
     * @param learningRate the Adam learning rate
     * @param clip SMILE's clipping threshold {@code tau}, for example 5 (ignored by other bounds)
     * @param seed the seed
     * @return the estimate in nats
     */
    public static double variational(Bound bound, double[][] xRaw, double[][] yRaw, int hidden, int embedding,
            int batch, int steps, double learningRate, double clip, long seed) {
        Objects.requireNonNull(bound, "bound");
        double[][] x = standardised(xRaw);
        double[][] y = standardised(yRaw);
        SplittableRandom r = new SplittableRandom(seed);
        int dx = x[0].length;
        int dy = y[0].length;
        Var wx = Var.param(Tensor.normal(Shape.of(dx, hidden), 0, (float) Math.sqrt(2.0 / dx), r));
        Var bx = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        Var vx = Var.param(Tensor.normal(Shape.of(hidden, embedding), 0, (float) (0.2 / Math.sqrt(hidden)), r));
        Var wy = Var.param(Tensor.normal(Shape.of(dy, hidden), 0, (float) Math.sqrt(2.0 / dy), r));
        Var by = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        Var vy = Var.param(Tensor.normal(Shape.of(hidden, embedding), 0, (float) (0.2 / Math.sqrt(hidden)), r));
        List<Var> ps = List.of(wx, bx, vx, wy, by, vy);
        Optimizer opt = Optimizer.adam((float) learningRate);
        float[] eye = new float[batch * batch];
        float[] off = new float[batch * batch];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < batch; j++) {
                eye[i * batch + j] = i == j ? 1 : 0;
                off[i * batch + j] = i == j ? 0 : 1;
            }
        }
        Tensor diagMask = Tensor.of(Shape.of(batch, batch), eye);
        Tensor offMask = Tensor.of(Shape.of(batch, batch), off);
        double offCount = batch * (batch - 1.0);
        double movingAverage = Double.NaN;
        int n = x.length;
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[batch];
            for (int i = 0; i < batch; i++) {
                idx[i] = r.nextInt(n);
            }
            Var t = scores(x, y, idx, wx, bx, vx, wy, by, vy);
            Var joint = t.mul(diagMask).sum().div(batch);
            Var objective = switch (bound) {
                case INFONCE -> joint.sub(t.logsumexp(1, false).mean());
                case NWJ -> joint.sub(t.sub(1).exp().mul(offMask).sum().div((float) offCount));
                case MINE -> {
                    Var marginal = t.exp().mul(offMask).sum().div((float) offCount);
                    double now = marginal.value().item();
                    movingAverage = Double.isNaN(movingAverage) ? now : 0.99 * movingAverage + 0.01 * now;
                    yield joint.sub(marginal.div((float) movingAverage));
                }
                case SMILE -> t.neg().softplus().neg().mul(diagMask).sum().div(batch)
                        .sub(t.softplus().mul(offMask).sum().div((float) offCount));
            };
            objective.neg().backward();
            opt.step(ps);
        }
        double total = 0;
        int batches = n / batch;
        for (int b = 0; b < batches; b++) {
            int[] idx = new int[batch];
            for (int i = 0; i < batch; i++) {
                idx[i] = b * batch + i;
            }
            double[] t = Var.noGrad(() -> scores(x, y, idx, wx, bx, vx, wy, by, vy)).value().toDoubleArray();
            total += readout(bound, t, batch, clip);
        }
        return total / batches;
    }

    private static Var scores(double[][] x, double[][] y, int[] idx, Var wx, Var bx, Var vx, Var wy, Var by,
            Var vy) {
        Var gx = Var.of(rows(x, idx)).matmul(wx).add(bx).relu().matmul(vx);
        Var hy = Var.of(rows(y, idx)).matmul(wy).add(by).relu().matmul(vy);
        return gx.matmul(hy.transpose());
    }

    private static double readout(Bound bound, double[] t, int b, double clip) {
        double joint = 0;
        double[] off = new double[b * (b - 1)];
        int c = 0;
        for (int i = 0; i < b; i++) {
            joint += t[i * b + i] / b;
            for (int j = 0; j < b; j++) {
                if (i != j) {
                    off[c++] = t[i * b + j];
                }
            }
        }
        switch (bound) {
            case INFONCE -> {
                double s = 0;
                for (int i = 0; i < b; i++) {
                    double m = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < b; j++) {
                        m = Math.max(m, t[i * b + j]);
                    }
                    double z = 0;
                    for (int j = 0; j < b; j++) {
                        z += Math.exp(t[i * b + j] - m);
                    }
                    s += (m + Math.log(z)) / b;
                }
                return Math.log(b) + joint - s;
            }
            case NWJ -> {
                double s = 0;
                for (double v : off) {
                    s += Math.exp(v - 1) / off.length;
                }
                return joint - s;
            }
            case MINE -> {
                return joint - logMeanExp(off);
            }
            default -> {
                double[] clipped = new double[off.length];
                for (int i = 0; i < off.length; i++) {
                    clipped[i] = Math.max(-clip, Math.min(clip, off[i]));
                }
                return joint - logMeanExp(clipped);
            }
        }
    }

    /** Per-coordinate z-scores; mutual information is invariant to them. */
    private static double[][] standardised(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] m = new double[d];
        double[] s = new double[d];
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                m[j] += r[j] / n;
            }
        }
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                s[j] += (r[j] - m[j]) * (r[j] - m[j]) / n;
            }
        }
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = (x[i][j] - m[j]) / Math.sqrt(s[j] + 1e-300);
            }
        }
        return out;
    }

    private static double logMeanExp(double[] v) {
        double m = Double.NEGATIVE_INFINITY;
        for (double a : v) {
            m = Math.max(m, a);
        }
        double s = 0;
        for (double a : v) {
            s += Math.exp(a - m);
        }
        return m + Math.log(s / v.length);
    }

    private static Tensor rows(double[][] x, int[] idx) {
        int d = x[0].length;
        float[] v = new float[idx.length * d];
        for (int i = 0; i < idx.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[idx[i]][j];
            }
        }
        return Tensor.of(Shape.of(idx.length, d), v);
    }
}
