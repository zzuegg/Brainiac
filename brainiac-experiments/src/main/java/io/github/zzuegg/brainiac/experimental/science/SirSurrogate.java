package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.sciml.PhysicsInformedNetwork;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * A parametric physics-informed surrogate of the SIR model for fast epidemic forecasting
 * (after Rama et al., 2025, <a href="https://arxiv.org/abs/2506.03897">arXiv:2506.03897</a>): one
 * network maps {@code (u, beta, gamma)}, with {@code u = t - tau0} the time since the epidemic's
 * start in weeks, to the susceptible fraction {@code s} and the log infectious fraction
 * {@code l = log i}. It is trained once over a prior box of parameters with the SIR equations
 * {@code s' = -beta s e^l} and {@code l' = beta s - gamma} as residuals (through
 * {@link PhysicsInformedNetwork}'s derivative machinery), the initial condition, and the
 * solutions of a few simulated epidemics; with {@code physicsWeight = 0} it is the plain MLP
 * surrogate trained on simulated curves only.
 *
 * <p>For a season it is never retrained: weekly counts {@code C_k = A (s(k - tau0) - s(k + 1 - tau0))}
 * with scale {@code A = N rho} enter a Poisson likelihood on a rolling window, and
 * {@code (beta, gamma, tau0, log A)} are sampled by random-walk Metropolis
 * ({@link HamiltonianMonteCarlo#randomWalkMetropolis}); the posterior predictive gives the
 * forecast. {@link #exactCurve} is the RK4 solution the surrogate replaces.
 */
public final class SirSurrogate {

    /** The latest start of an epidemic in the prior box, in weeks. */
    public static final double MAX_OFFSET = 10;
    private static final double EPS = 1e-3;
    private static final double SPAN = 30;

    private final PhysicsInformedNetwork net;
    private final double physicsWeight;

    /**
     * An untrained surrogate.
     *
     * @param width hidden width
     * @param depth hidden layers
     * @param physicsWeight the weight of the SIR residual (0 for the data-only MLP)
     * @param seed the seed
     */
    public SirSurrogate(int width, int depth, double physicsWeight, long seed) {
        this.net = PhysicsInformedNetwork.of(PhysicsInformedNetwork.tanhMlp(4, width, depth, 2, seed));
        this.physicsWeight = physicsWeight;
    }

    /**
     * The exact SIR solution {@code s(u)} at the given times, by RK4.
     *
     * @param beta transmission rate per week
     * @param gamma recovery rate per week
     * @param u times since the start (negative means before)
     * @return {@code s(u)}
     */
    public static double[] exactCurve(double beta, double gamma, double[] u) {
        double[] out = new double[u.length];
        double h = 0.05;
        for (int k = 0; k < u.length; k++) {
            if (u[k] <= 0) {
                out[k] = 1 - EPS;
                continue;
            }
            double s = 1 - EPS;
            double i = EPS;
            int steps = (int) Math.ceil(u[k] / h);
            double dt = u[k] / steps;
            for (int j = 0; j < steps; j++) {
                double[] k1 = {-beta * s * i, beta * s * i - gamma * i};
                double s2 = s + dt / 2 * k1[0];
                double i2 = i + dt / 2 * k1[1];
                double[] k2 = {-beta * s2 * i2, beta * s2 * i2 - gamma * i2};
                double s3 = s + dt / 2 * k2[0];
                double i3 = i + dt / 2 * k2[1];
                double[] k3 = {-beta * s3 * i3, beta * s3 * i3 - gamma * i3};
                double s4 = s + dt * k3[0];
                double i4 = i + dt * k3[1];
                double[] k4 = {-beta * s4 * i4, beta * s4 * i4 - gamma * i4};
                s += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
                i += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
            }
            out[k] = s;
        }
        return out;
    }

    // the fourth input, the early growth exponent (beta - gamma) u, makes the take-off time
    // nearly linear in the inputs
    private static float[] encode(double u, double beta, double gamma) {
        return new float[] {(float) (u / SPAN), (float) (beta - 2), (float) (gamma - 1),
            (float) ((beta - gamma) * u / GROWTH)};
    }

    private static final double GROWTH = 10;

    /**
     * Trains over the prior box {@code beta in [1.2, 3]}, {@code gamma in [0.6, 1.4]},
     * {@code u in [0, 30]} weeks.
     *
     * @param curves simulated epidemics used as data
     * @param steps Adam steps
     * @param batch points per loss term per step
     * @param seed the seed
     * @return the final loss
     */
    public double train(int curves, int steps, int batch, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        // simulated data: s(u) on a weekly grid for random parameters
        int perCurve = 31;
        float[] dx = new float[curves * perCurve * 4];
        float[] dy = new float[curves * perCurve];
        double[] grid = new double[perCurve];
        for (int k = 0; k < perCurve; k++) {
            grid[k] = k;
        }
        for (int c = 0; c < curves; c++) {
            double beta = 1.2 + 1.8 * r.nextDouble();
            double gamma = 0.6 + 0.8 * r.nextDouble();
            double[] s = exactCurve(beta, gamma, grid);
            for (int k = 0; k < perCurve; k++) {
                System.arraycopy(encode(grid[k], beta, gamma), 0, dx, (c * perCurve + k) * 4, 4);
                dy[c * perCurve + k] = (float) s[k];
            }
        }
        Optimizer opt = Optimizer.adam(1e-2f);
        double last = 0;
        int total = curves * perCurve;
        for (int step = 0; step < steps; step++) {
            // cosine decay from 1e-2 to 1e-4
            opt.learningRate((float) (1e-4 + 0.5 * (1e-2 - 1e-4) * (1 + Math.cos(Math.PI * step / steps))));
            float[] bx = new float[batch * 4];
            float[] by = new float[batch];
            for (int b = 0; b < batch; b++) {
                int i = r.nextInt(total);
                System.arraycopy(dx, i * 4, bx, b * 4, 4);
                by[b] = dy[i];
            }
            float[] cx = new float[batch * 4];
            float[] betas = new float[batch];
            float[] gammas = new float[batch];
            float[] ic = new float[batch * 4];
            for (int b = 0; b < batch; b++) {
                double beta = 1.2 + 1.8 * r.nextDouble();
                double gamma = 0.6 + 0.8 * r.nextDouble();
                System.arraycopy(encode(SPAN * r.nextDouble(), beta, gamma), 0, cx, b * 4, 4);
                betas[b] = (float) beta;
                gammas[b] = (float) gamma;
                System.arraycopy(encode(0, beta, gamma), 0, ic, b * 4, 4);
            }
            Tensor dataX = Tensor.of(Shape.of(batch, 4), bx);
            Tensor dataY = Tensor.of(Shape.of(batch, 1), by);
            Tensor colX = Tensor.of(Shape.of(batch, 4), cx);
            Tensor betaT = Tensor.of(Shape.of(batch, 1), betas);
            Tensor gammaT = Tensor.of(Shape.of(batch, 1), gammas);
            Tensor icX = Tensor.of(Shape.of(batch, 4), ic);
            Tensor growth = betaT.sub(gammaT).mul((float) (1 / GROWTH));
            Var l = opt.minimize(net.parameters(), () -> {
                Var out = net.forward(Var.of(dataX));
                Var loss = out.slice(1, 0, 1).sub(Var.of(dataY)).square().mean().mul(100f);
                Var init = net.forward(Var.of(icX));
                loss = loss.add(init.slice(1, 0, 1).sub((float) (1 - EPS)).square().mean().mul(100f))
                        .add(logI(init).sub((float) Math.log(EPS)).square().mean());
                if (physicsWeight > 0) {
                    Var res = net.residual(colX, d -> {
                        Var s = d.u(0);
                        Var li = d.u(1).mul(4f).sub(4f);
                        // chain rule through both time inputs
                        Var ds = d.first(0, 0).mul((float) (1 / SPAN)).add(d.first(3, 0).mul(growth));
                        Var dl = d.first(0, 1).mul((float) (1 / SPAN)).add(d.first(3, 1).mul(growth)).mul(4f);
                        Var bt = Var.of(betaT);
                        Var r1 = ds.add(bt.mul(s).mul(li.exp())).mul(10f);
                        Var r2 = dl.sub(bt.mul(s)).add(Var.of(gammaT));
                        return Var.concat(1, r1, r2);
                    });
                    loss = loss.add(res.square().mean().mul((float) physicsWeight));
                }
                return loss;
            });
            last = l.value().item();
        }
        return last;
    }

    private static Var logI(Var out) {
        return out.slice(1, 1, 2).mul(4f).sub(4f);
    }

    /**
     * The surrogate's {@code s(u)}.
     *
     * @param beta transmission rate
     * @param gamma recovery rate
     * @param u times since the start
     * @return {@code s(u)} (exactly {@code 1 - eps} for {@code u <= 0})
     */
    public double[] curve(double beta, double gamma, double[] u) {
        int n = u.length;
        float[] x = new float[n * 4];
        for (int k = 0; k < n; k++) {
            System.arraycopy(encode(Math.max(0, u[k]), beta, gamma), 0, x, k * 4, 4);
        }
        float[] v = Var.noGrad(() -> net.forward(Var.of(Tensor.of(Shape.of(n, 4), x)))).value().toArray();
        double[] s = new double[n];
        for (int k = 0; k < n; k++) {
            s[k] = u[k] <= 0 ? 1 - EPS : v[2 * k];
        }
        return s;
    }

    /** Computes {@code s(u)}. */
    @FunctionalInterface
    public interface Curve {
        /**
         * @param beta transmission rate
         * @param gamma recovery rate
         * @param u times
         * @return {@code s(u)}
         */
        double[] at(double beta, double gamma, double[] u);
    }

    /**
     * Expected weekly counts for weeks {@code first .. first + count - 1}.
     *
     * @param curve the solver (surrogate or exact)
     * @param theta {beta, gamma, tau0, log A}
     * @param first the first week
     * @param count weeks
     * @return expected counts
     */
    public static double[] expectedCounts(Curve curve, double[] theta, int first, int count) {
        double[] u = new double[count + 1];
        for (int k = 0; k <= count; k++) {
            u[k] = first + k - theta[2];
        }
        double[] s = curve.at(theta[0], theta[1], u);
        double a = Math.exp(theta[3]);
        double[] c = new double[count];
        for (int k = 0; k < count; k++) {
            c[k] = Math.max(1e-6, a * (s[k] - s[k + 1]));
        }
        return c;
    }

    /**
     * A rolling-window forecast.
     *
     * @param mean posterior predictive mean per horizon week
     * @param lower50 25% quantile
     * @param upper50 75% quantile
     * @param lower90 5% quantile
     * @param upper90 95% quantile
     * @param seconds wall time of the inference
     */
    public record Forecast(double[] mean, double[] lower50, double[] upper50, double[] lower90, double[] upper90,
            double seconds) {
    }

    /**
     * Fits a window of weekly counts by random-walk Metropolis and forecasts ahead.
     *
     * @param curve the solver
     * @param counts weekly counts
     * @param end the first week after the window
     * @param window weeks in the window
     * @param horizon weeks to forecast
     * @param samples kept MCMC samples
     * @param dispersion the negative-binomial dispersion of the counts ({@code <= 0} for Poisson)
     * @param seed the seed
     * @return the forecast
     */
    public static Forecast forecast(Curve curve, int[] counts, int end, int window, int horizon, int samples,
            double dispersion, long seed) {
        long t0 = System.nanoTime();
        int first = end - window;
        double scaleGuess = 0;
        for (int k = first; k < end; k++) {
            scaleGuess = Math.max(scaleGuess, counts[k]);
        }
        double logA0 = Math.log(Math.max(10, scaleGuess * 10));
        LogDensity post = (x, g) -> {
            if (x[0] < 1.2 || x[0] > 3 || x[1] < 0.6 || x[1] > 1.4 || x[2] < 0 || x[2] > MAX_OFFSET
                    || Math.abs(x[3] - logA0) > 6) {
                return Double.NEGATIVE_INFINITY;
            }
            double[] mu = expectedCounts(curve, x, first, window);
            int[] obs = Arrays.copyOfRange(counts, first, end);
            return -Epidemics.nll(obs, mu, dispersion);
        };
        SplittableRandom r = new SplittableRandom(seed);
        double[] start = {2.0, 1.0, Math.max(0, Math.min(MAX_OFFSET, first - 4)), logA0};
        // a few restarts pick the best starting point
        double best = post.value(start);
        for (int k = 0; k < 200; k++) {
            double[] c = {1.2 + 1.8 * r.nextDouble(), 0.6 + 0.8 * r.nextDouble(), MAX_OFFSET * r.nextDouble(),
                logA0 + 4 * (r.nextDouble() - 0.5)};
            double v = post.value(c);
            if (v > best) {
                best = v;
                start = c;
            }
        }
        HamiltonianMonteCarlo.Result res = HamiltonianMonteCarlo.randomWalkMetropolis(post, start, 0.05, samples,
                samples, r);
        double[][] pred = new double[samples][];
        for (int i = 0; i < samples; i++) {
            double[] mu = expectedCounts(curve, res.samples()[i], end, horizon);
            for (int h = 0; h < horizon; h++) {
                mu[h] = dispersion > 0 ? ScienceSims.negativeBinomial(mu[h], dispersion, r) : ScienceSims.poisson(mu[h], r);
            }
            pred[i] = mu;
        }
        double[] mean = new double[horizon];
        double[][] q = new double[4][horizon];
        double[] levels = {0.25, 0.75, 0.05, 0.95};
        for (int h = 0; h < horizon; h++) {
            double[] col = new double[samples];
            for (int i = 0; i < samples; i++) {
                col[i] = pred[i][h];
                mean[h] += col[i] / samples;
            }
            Arrays.sort(col);
            for (int j = 0; j < 4; j++) {
                q[j][h] = col[(int) Math.min(samples - 1, levels[j] * samples)];
            }
        }
        return new Forecast(mean, q[0], q[1], q[2], q[3], (System.nanoTime() - t0) / 1e9);
    }
}
