package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Annealed importance sampling and SMC samplers for marginal likelihoods (Neal 2001,
 * <a href="https://arxiv.org/abs/physics/9803008">arXiv:physics/9803008</a>; Del Moral et al.
 * 2006, <a href="https://academic.oup.com/jrsssb/article/68/3/411/7110641">SMC samplers</a>).
 *
 * <p>To estimate {@code Z = integral f(x) dx} for an unnormalised target {@code f}, particles
 * start from a normalised prior {@code p0} and pass through the tempered densities
 * {@code p_b = p0^(1-b) f^b} for {@code 0 = b_0 < ... < b_K = 1}; at each stage the weight
 * gains {@code (b_k - b_(k-1)) (log f - log p0)} and the particle takes HMC moves that leave
 * {@code p_(b_k)} invariant. {@code mean(exp(w))} is an unbiased estimate of {@code Z}, and its
 * variance shrinks as the stages get finer, where plain importance sampling from the prior
 * collapses whenever the posterior is far narrower than the prior. The SMC-sampler variant
 * resamples when the weights' ESS drops below half, accumulating the evidence across stages.
 * Wu et al. 2017 (<a href="https://arxiv.org/abs/1611.04273">arXiv:1611.04273</a>) made AIS the
 * standard log-likelihood evaluation of decoder-based models.
 */
public final class AnnealedImportanceSampler {

    /**
     * An evidence estimate.
     *
     * @param logZ the estimate of {@code log Z}
     * @param ess the final importance ESS
     * @param particles the final particles, approximately from the target
     * @param gradientEvaluations the gradient evaluations spent
     */
    public record Estimate(double logZ, double ess, double[][] particles, long gradientEvaluations) {
    }

    private AnnealedImportanceSampler() {
    }

    /**
     * Plain importance sampling from the prior: {@code log mean(f(x) / p0(x))}.
     *
     * @param prior the normalised prior density
     * @param priorSampler draws from the prior
     * @param target the unnormalised target
     * @param n the number of samples
     * @param random the random source
     * @return the estimate (no gradients spent)
     */
    public static Estimate importance(LogDensity prior, Function<SplittableRandom, double[]> priorSampler,
            LogDensity target, int n, SplittableRandom random) {
        double[] lw = new double[n];
        double[][] xs = new double[n][];
        for (int i = 0; i < n; i++) {
            xs[i] = priorSampler.apply(random);
            lw[i] = target.value(xs[i]) - prior.value(xs[i]);
        }
        return new Estimate(McmcDiagnostics.logMeanExp(lw), McmcDiagnostics.importanceEss(lw), xs, 0);
    }

    /**
     * A sigmoidal annealing schedule, dense near both ends (Grosse et al. 2015).
     *
     * @param stages the number of stages {@code K}
     * @return {@code b_0 = 0, ..., b_K = 1}
     */
    public static double[] sigmoidSchedule(int stages) {
        double[] b = new double[stages + 1];
        double delta = 4;
        double lo = 1 / (1 + Math.exp(delta));
        double hi = 1 / (1 + Math.exp(-delta));
        for (int k = 0; k <= stages; k++) {
            double s = 1 / (1 + Math.exp(-(-delta + 2 * delta * k / stages)));
            b[k] = (s - lo) / (hi - lo);
        }
        b[0] = 0;
        b[stages] = 1;
        return b;
    }

    /**
     * Runs AIS (or, with {@code resample}, an SMC sampler) with HMC transitions whose step size
     * adapts between stages towards an acceptance of about 0.65.
     *
     * @param prior the normalised prior density
     * @param priorSampler draws from the prior
     * @param target the unnormalised target
     * @param particles the number of particles
     * @param stages the number of annealing stages
     * @param leapfrogSteps leapfrog steps per HMC move
     * @param stepSize the initial HMC step size
     * @param resample resample when the ESS drops below half (SMC sampler)
     * @param random the random source
     * @return the estimate
     */
    public static Estimate anneal(LogDensity prior, Function<SplittableRandom, double[]> priorSampler,
            LogDensity target, int particles, int stages, int leapfrogSteps, double stepSize, boolean resample,
            SplittableRandom random) {
        double[] beta = sigmoidSchedule(stages);
        double[][] x = new double[particles][];
        double[] lw = new double[particles];
        double[] logRatio = new double[particles];
        for (int i = 0; i < particles; i++) {
            x[i] = priorSampler.apply(random);
            logRatio[i] = target.value(x[i]) - prior.value(x[i]);
        }
        double logZ = 0;
        double eps = stepSize;
        long grads = 0;
        double[] acc = new double[1];
        for (int k = 1; k <= stages; k++) {
            for (int i = 0; i < particles; i++) {
                lw[i] += (beta[k] - beta[k - 1]) * logRatio[i];
            }
            if (resample && McmcDiagnostics.importanceEss(lw) < particles / 2.0 && k < stages) {
                logZ += McmcDiagnostics.logMeanExp(lw);
                int[] pick = ProbabilisticProgram.systematic(lw, random);
                double[][] nx = new double[particles][];
                double[] nr = new double[particles];
                for (int i = 0; i < particles; i++) {
                    nx[i] = x[pick[i]].clone();
                    nr[i] = logRatio[pick[i]];
                }
                x = nx;
                logRatio = nr;
                java.util.Arrays.fill(lw, 0);
            }
            LogDensity stage = LogDensity.tempered(prior, target, beta[k]);
            double meanAcc = 0;
            for (int i = 0; i < particles; i++) {
                x[i] = HamiltonianMonteCarlo.transition(stage, x[i], eps, leapfrogSteps, random, acc);
                grads += leapfrogSteps + 1;
                meanAcc += acc[0] / particles;
                logRatio[i] = target.value(x[i]) - prior.value(x[i]);
            }
            eps *= Math.exp(0.5 * (meanAcc - 0.65));
        }
        logZ += McmcDiagnostics.logMeanExp(lw);
        return new Estimate(logZ, McmcDiagnostics.importanceEss(lw), x, grads);
    }
}
