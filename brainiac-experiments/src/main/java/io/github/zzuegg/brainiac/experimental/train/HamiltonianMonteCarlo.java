package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Hamiltonian Monte Carlo and the No-U-Turn Sampler on any {@link LogDensity}, plus a
 * random-walk Metropolis baseline and an adapter that turns a network's weights and loss into a
 * weight posterior.
 *
 * <p>HMC (Neal 2011, <a href="https://arxiv.org/abs/1206.1901">arXiv:1206.1901</a>) draws a
 * Gaussian momentum, simulates Hamiltonian dynamics with leapfrog steps along the gradient of
 * {@code log p}, and accepts the end point with the Metropolis probability
 * {@code exp(-dH)}; long, gradient-guided moves give far less correlated draws than a random
 * walk. NUTS (Hoffman and Gelman 2014, <a href="https://arxiv.org/abs/1111.4246">
 * arXiv:1111.4246</a>, algorithm 6) removes the trajectory-length tuning: it doubles the
 * trajectory forwards or backwards until the ends start turning back on each other (a U-turn),
 * and picks a point from the trajectory by slice sampling. Both adapt the step size during
 * warm-up by dual averaging towards a target acceptance rate.
 *
 * <p>For a Bayesian neural network, {@link #parameterDensity} wraps the parameters and a
 * negative log posterior (the training loss times the dataset size plus the prior) built with
 * autograd; a temperature {@code T < 1} gives the "cold posterior" {@code p^(1/T)} of Wenzel
 * et al. 2020 (<a href="https://arxiv.org/abs/2002.02405">arXiv:2002.02405</a>). Full-batch
 * HMC on such a model is the reference posterior of Izmailov et al. 2021
 * (<a href="https://arxiv.org/abs/2104.14421">arXiv:2104.14421</a>).
 */
public final class HamiltonianMonteCarlo {

    /**
     * Samples from one chain.
     *
     * @param samples the draws after warm-up {@code [n][d]}
     * @param acceptance the mean acceptance statistic after warm-up
     * @param stepSize the adapted step size
     * @param gradientEvaluations the number of gradient evaluations in total
     */
    public record Result(double[][] samples, double acceptance, double stepSize, long gradientEvaluations) {
    }

    private static final double MAX_ENERGY_ERROR = 1000;

    private HamiltonianMonteCarlo() {
    }

    /** Counts gradient calls of a density. */
    private static final class Counted implements LogDensity {
        private final LogDensity inner;
        private long calls;

        Counted(LogDensity inner) {
            this.inner = inner;
        }

        @Override
        public double evaluate(double[] x, double[] gradient) {
            calls++;
            return inner.evaluate(x, gradient);
        }
    }

    /** Dual averaging of the log step size (Nesterov 2009; Hoffman and Gelman, section 3.2). */
    private static final class DualAveraging {
        private final double mu;
        private final double target;
        private double hBar;
        private double logEps;
        private double logEpsBar;
        private int t;

        DualAveraging(double eps, double target) {
            this.mu = Math.log(10 * eps);
            this.target = target;
            this.logEps = Math.log(eps);
        }

        double update(double acceptance) {
            t++;
            double w = 1.0 / (t + 10);
            hBar = (1 - w) * hBar + w * (target - acceptance);
            logEps = mu - Math.sqrt(t) / 0.05 * hBar;
            double eta = Math.pow(t, -0.75);
            logEpsBar = eta * logEps + (1 - eta) * logEpsBar;
            return Math.exp(logEps);
        }

        double finalStep() {
            return Math.exp(logEpsBar);
        }
    }

    /**
     * One leapfrog step in place.
     *
     * @param target the density
     * @param x the position, updated
     * @param p the momentum, updated
     * @param grad the gradient at {@code x} on entry, at the new {@code x} on exit
     * @param eps the step size (negative runs backwards)
     * @return {@code log p} at the new position
     */
    public static double leapfrog(LogDensity target, double[] x, double[] p, double[] grad, double eps) {
        for (int i = 0; i < x.length; i++) {
            p[i] += 0.5 * eps * grad[i];
            x[i] += eps * p[i];
        }
        double lp = target.evaluate(x, grad);
        for (int i = 0; i < x.length; i++) {
            p[i] += 0.5 * eps * grad[i];
        }
        return lp;
    }

    private static double kinetic(double[] p) {
        double s = 0;
        for (double v : p) {
            s += v * v;
        }
        return 0.5 * s;
    }

    private static double[] momentum(int d, SplittableRandom random) {
        double[] p = new double[d];
        for (int i = 0; i < d; i++) {
            p[i] = random.nextGaussian();
        }
        return p;
    }

    /**
     * One HMC transition: draws a momentum, runs {@code leapfrogSteps} steps and accepts or
     * rejects. AIS and SMC samplers use it as their MCMC move.
     *
     * @param target the density
     * @param x the current position (not modified)
     * @param stepSize the leapfrog step size
     * @param leapfrogSteps the trajectory length in steps
     * @param random the random source
     * @param acceptance receives the acceptance probability in element 0, or null
     * @return the next position (a new array, or a copy of {@code x} on rejection)
     */
    public static double[] transition(LogDensity target, double[] x, double stepSize, int leapfrogSteps,
            SplittableRandom random, double[] acceptance) {
        int d = x.length;
        double[] grad = new double[d];
        double lp0 = target.evaluate(x, grad);
        double[] p = momentum(d, random);
        double h0 = -lp0 + kinetic(p);
        double[] y = x.clone();
        double lp = lp0;
        for (int s = 0; s < leapfrogSteps; s++) {
            lp = leapfrog(target, y, p, grad, stepSize);
        }
        double h1 = -lp + kinetic(p);
        double a = Double.isFinite(h1) ? Math.min(1, Math.exp(h0 - h1)) : 0;
        if (acceptance != null) {
            acceptance[0] = a;
        }
        return random.nextDouble() < a ? y : x.clone();
    }

    /**
     * A reasonable first step size: doubles or halves it until one leapfrog step's acceptance
     * crosses one half (Hoffman and Gelman, algorithm 4).
     *
     * @param target the density
     * @param x a starting point
     * @param random the random source
     * @return the step size
     */
    public static double initialStepSize(LogDensity target, double[] x, SplittableRandom random) {
        int d = x.length;
        double eps = 0.1;
        double[] grad = new double[d];
        double lp0 = target.evaluate(x, grad);
        double[] p0 = momentum(d, random);
        double[] g0 = grad.clone();
        double logRatio = logAcceptAfterOneStep(target, x, p0, g0, lp0, eps);
        int direction = logRatio > Math.log(0.5) ? 1 : -1;
        for (int i = 0; i < 50; i++) {
            eps *= Math.pow(2, direction);
            logRatio = logAcceptAfterOneStep(target, x, p0, g0, lp0, eps);
            if (direction * logRatio <= direction * Math.log(0.5)) {
                break;
            }
        }
        return eps;
    }

    private static double logAcceptAfterOneStep(LogDensity target, double[] x, double[] p0, double[] g0, double lp0,
            double eps) {
        double[] y = x.clone();
        double[] p = p0.clone();
        double[] g = g0.clone();
        double lp = leapfrog(target, y, p, g, eps);
        double r = lp - kinetic(p) - lp0 + kinetic(p0);
        return Double.isFinite(r) ? r : Double.NEGATIVE_INFINITY;
    }

    /**
     * Runs HMC with a fixed number of leapfrog steps; the step size adapts by dual averaging
     * during warm-up when {@code targetAcceptance > 0}.
     *
     * @param target the density
     * @param start the starting point
     * @param stepSize the initial step size (0 picks one)
     * @param leapfrogSteps the steps per trajectory
     * @param warmup the warm-up transitions (discarded)
     * @param samples the kept transitions
     * @param targetAcceptance the adaptation target, for example 0.8, or 0 to keep the step
     * @param random the random source
     * @return the chain
     */
    public static Result hmc(LogDensity target, double[] start, double stepSize, int leapfrogSteps, int warmup,
            int samples, double targetAcceptance, SplittableRandom random) {
        Counted f = new Counted(target);
        double eps = stepSize > 0 ? stepSize : initialStepSize(f, start, random);
        DualAveraging da = new DualAveraging(eps, targetAcceptance);
        double[] x = start.clone();
        double[][] out = new double[samples][];
        double[] a = new double[1];
        double acc = 0;
        for (int it = 0; it < warmup + samples; it++) {
            x = transition(f, x, eps, leapfrogSteps, random, a);
            if (it < warmup) {
                if (targetAcceptance > 0) {
                    eps = it == warmup - 1 ? da.finalStep() : da.update(a[0]);
                }
            } else {
                out[it - warmup] = x;
                acc += a[0] / samples;
            }
        }
        return new Result(out, acc, eps, f.calls);
    }

    /** The state of a NUTS tree build. */
    private static final class Tree {
        double[] xMinus;
        double[] pMinus;
        double[] gMinus;
        double[] xPlus;
        double[] pPlus;
        double[] gPlus;
        double[] proposal;
        int n;
        boolean ok;
        double alpha;
        int nAlpha;
    }

    /**
     * Runs the No-U-Turn Sampler with dual-averaging step-size adaptation.
     *
     * @param target the density
     * @param start the starting point
     * @param warmup the warm-up iterations (adapt the step, discarded)
     * @param samples the kept iterations
     * @param targetAcceptance the adaptation target, for example 0.8
     * @param maxDepth the maximum tree depth, for example 10
     * @param random the random source
     * @return the chain
     */
    public static Result nuts(LogDensity target, double[] start, int warmup, int samples, double targetAcceptance,
            int maxDepth, SplittableRandom random) {
        Counted f = new Counted(target);
        int d = start.length;
        double eps = initialStepSize(f, start, random);
        DualAveraging da = new DualAveraging(eps, targetAcceptance);
        double[] x = start.clone();
        double[] grad = new double[d];
        double lp = f.evaluate(x, grad);
        double[][] out = new double[samples][];
        double acc = 0;
        for (int it = 0; it < warmup + samples; it++) {
            double[] p0 = momentum(d, random);
            double joint0 = lp - kinetic(p0);
            double logU = joint0 + Math.log(random.nextDouble());
            Tree t = new Tree();
            t.xMinus = x.clone();
            t.xPlus = x.clone();
            t.pMinus = p0.clone();
            t.pPlus = p0.clone();
            t.gMinus = grad.clone();
            t.gPlus = grad.clone();
            double[] chosen = x;
            int n = 1;
            boolean ok = true;
            double alpha = 0;
            int nAlpha = 0;
            for (int depth = 0; ok && depth < maxDepth; depth++) {
                int v = random.nextBoolean() ? 1 : -1;
                Tree sub;
                if (v == -1) {
                    sub = build(f, t.xMinus, t.pMinus, t.gMinus, logU, v, depth, eps, joint0, random);
                    t.xMinus = sub.xMinus;
                    t.pMinus = sub.pMinus;
                    t.gMinus = sub.gMinus;
                } else {
                    sub = build(f, t.xPlus, t.pPlus, t.gPlus, logU, v, depth, eps, joint0, random);
                    t.xPlus = sub.xPlus;
                    t.pPlus = sub.pPlus;
                    t.gPlus = sub.gPlus;
                }
                if (sub.ok && random.nextDouble() < (double) sub.n / n) {
                    chosen = sub.proposal;
                }
                n += sub.n;
                alpha = sub.alpha;
                nAlpha = sub.nAlpha;
                ok = sub.ok && noUTurn(t.xMinus, t.xPlus, t.pMinus, t.pPlus);
            }
            if (chosen != x) {
                x = chosen;
                lp = f.evaluate(x, grad);
            }
            double a = nAlpha > 0 ? alpha / nAlpha : 0;
            if (it < warmup) {
                eps = it == warmup - 1 ? da.finalStep() : da.update(a);
            } else {
                out[it - warmup] = x.clone();
                acc += a / samples;
            }
        }
        return new Result(out, acc, eps, f.calls);
    }

    private static boolean noUTurn(double[] xMinus, double[] xPlus, double[] pMinus, double[] pPlus) {
        double a = 0;
        double b = 0;
        for (int i = 0; i < xMinus.length; i++) {
            double dx = xPlus[i] - xMinus[i];
            a += dx * pMinus[i];
            b += dx * pPlus[i];
        }
        return a >= 0 && b >= 0;
    }

    private static Tree build(LogDensity f, double[] x, double[] p, double[] g, double logU, int v, int depth,
            double eps, double joint0, SplittableRandom random) {
        if (depth == 0) {
            double[] y = x.clone();
            double[] q = p.clone();
            double[] gg = g.clone();
            double lp = leapfrog(f, y, q, gg, v * eps);
            double joint = lp - kinetic(q);
            if (!Double.isFinite(joint)) {
                joint = Double.NEGATIVE_INFINITY;
            }
            Tree t = new Tree();
            t.xMinus = y;
            t.xPlus = y;
            t.pMinus = q;
            t.pPlus = q;
            t.gMinus = gg;
            t.gPlus = gg;
            t.proposal = y;
            t.n = logU <= joint ? 1 : 0;
            t.ok = joint > logU - MAX_ENERGY_ERROR;
            t.alpha = Double.isFinite(joint) ? Math.min(1, Math.exp(joint - joint0)) : 0;
            t.nAlpha = 1;
            return t;
        }
        Tree t = build(f, x, p, g, logU, v, depth - 1, eps, joint0, random);
        if (!t.ok) {
            return t;
        }
        Tree s;
        if (v == -1) {
            s = build(f, t.xMinus, t.pMinus, t.gMinus, logU, v, depth - 1, eps, joint0, random);
            t.xMinus = s.xMinus;
            t.pMinus = s.pMinus;
            t.gMinus = s.gMinus;
        } else {
            s = build(f, t.xPlus, t.pPlus, t.gPlus, logU, v, depth - 1, eps, joint0, random);
            t.xPlus = s.xPlus;
            t.pPlus = s.pPlus;
            t.gPlus = s.gPlus;
        }
        int total = t.n + s.n;
        if (total > 0 && random.nextDouble() < (double) s.n / total) {
            t.proposal = s.proposal;
        }
        t.alpha += s.alpha;
        t.nAlpha += s.nAlpha;
        t.ok = s.ok && noUTurn(t.xMinus, t.xPlus, t.pMinus, t.pPlus);
        t.n = total;
        return t;
    }

    /**
     * Random-walk Metropolis with an isotropic Gaussian proposal whose scale adapts towards an
     * acceptance of 0.234 during warm-up: the gradient-free baseline.
     *
     * @param target the density
     * @param start the starting point
     * @param scale the initial proposal standard deviation
     * @param warmup the warm-up iterations
     * @param samples the kept iterations
     * @param random the random source
     * @return the chain ({@code gradientEvaluations} counts density evaluations)
     */
    public static Result randomWalkMetropolis(LogDensity target, double[] start, double scale, int warmup,
            int samples, SplittableRandom random) {
        double[] x = start.clone();
        double lp = target.value(x);
        double s = scale;
        double[][] out = new double[samples][];
        double acc = 0;
        for (int it = 0; it < warmup + samples; it++) {
            double[] y = x.clone();
            for (int i = 0; i < y.length; i++) {
                y[i] += s * random.nextGaussian();
            }
            double ly = target.value(y);
            double a = Double.isFinite(ly) ? Math.min(1, Math.exp(ly - lp)) : 0;
            if (random.nextDouble() < a) {
                x = y;
                lp = ly;
            }
            if (it < warmup) {
                s *= Math.exp((a - 0.234) / Math.sqrt(it + 1.0));
            } else {
                out[it - warmup] = x.clone();
                acc += a / samples;
            }
        }
        return new Result(out, acc, s, warmup + samples + 1L);
    }

    /**
     * The weight posterior of a model as a log density over its flattened parameters:
     * {@code log p(w | D) = -energy(w) / temperature}, where {@code energy} builds the negative
     * log joint (summed data NLL plus the negative log prior) with autograd. Evaluating it
     * assigns the point to the parameters.
     *
     * @param parameters the model parameters
     * @param energy builds the negative log joint at the current parameters
     * @param temperature the posterior temperature, 1 for the Bayes posterior
     * @return the density
     */
    public static LogDensity parameterDensity(List<Var> parameters, Supplier<Var> energy, double temperature) {
        return (x, g) -> {
            assign(parameters, x);
            for (Var p : parameters) {
                p.zeroGrad();
            }
            Var e = energy.get();
            double value = -e.value().item() / temperature;
            if (g != null) {
                e.backward();
                int o = 0;
                for (Var p : parameters) {
                    Tensor gr = p.grad();
                    int size = (int) p.value().shape().size();
                    if (gr == null) {
                        java.util.Arrays.fill(g, o, o + size, 0);
                    } else {
                        double[] v = gr.toDoubleArray();
                        for (int i = 0; i < size; i++) {
                            g[o + i] = -v[i] / temperature;
                        }
                    }
                    o += size;
                    p.zeroGrad();
                }
            }
            return value;
        };
    }

    /**
     * Flattens parameter values into one vector.
     *
     * @param parameters the parameters
     * @return the concatenated values
     */
    public static double[] flatten(List<Var> parameters) {
        int total = 0;
        for (Var p : parameters) {
            total += (int) p.value().shape().size();
        }
        double[] x = new double[total];
        int o = 0;
        for (Var p : parameters) {
            double[] v = p.value().toDoubleArray();
            System.arraycopy(v, 0, x, o, v.length);
            o += v.length;
        }
        return x;
    }

    /**
     * Writes a flat vector back into parameters.
     *
     * @param parameters the parameters
     * @param x the concatenated values
     */
    public static void assign(List<Var> parameters, double[] x) {
        int o = 0;
        for (Var p : parameters) {
            int size = (int) p.value().shape().size();
            float[] v = new float[size];
            for (int i = 0; i < size; i++) {
                v[i] = (float) x[o + i];
            }
            p.assign(Tensor.of(p.value().shape(), v));
            o += size;
        }
    }
}
