package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Score-based generative modelling through SDEs (Song et al.,
 * <a href="https://arxiv.org/abs/2011.13456">arXiv:2011.13456</a>): data diffuses along a
 * forward SDE {@code dx = f(x, t) dt + g(t) dW} on continuous time {@code t in [eps, 1]}, a
 * network learns the score {@code grad log p_t} by denoising score matching (here as a noise
 * predictor, {@code score = -eps_hat / std(t)}), and generation runs time backwards:
 * <ul>
 *   <li>{@link #samplePredictorCorrector}: reverse-diffusion steps of the reverse SDE
 *       (predictor, the paper's discretisation that stays stable at large steps), each
 *       followed by Langevin MCMC steps at the new time (corrector) with a step size set by
 *       a target signal-to-noise ratio;</li>
 *   <li>{@link #sampleOde}: the deterministic probability-flow ODE
 *       {@code dx/dt = f - g^2 score / 2}, which shares the SDE's marginals, so it can be
 *       solved with few Heun steps and gives exact likelihoods
 *       ({@link #probabilityFlow} plugs into {@link ContinuousFlow}).</li>
 * </ul>
 * Two SDEs: variance preserving ({@code VP}, the continuous DDPM, {@code f = -beta(t) x / 2},
 * linear {@code beta} from 0.1 to 20) and variance exploding ({@code VE}, SMLD,
 * {@code f = 0}, {@code sigma(t) = sigma_min (sigma_max / sigma_min)^t}).
 */
public final class ScoreSde {

    /** The forward SDE. */
    public enum Kind {
        /** Variance preserving: the continuous-time DDPM. */
        VP,
        /** Variance exploding: noise-conditional score networks (SMLD). */
        VE
    }

    private static final double BETA_MIN = 0.1;
    private static final double BETA_MAX = 20;
    private final Kind kind;
    private final Denoiser network;
    private final double sigmaMin;
    private final double sigmaMax;
    private final double epsilon = 1e-3;

    private ScoreSde(Kind kind, Denoiser network, double sigmaMin, double sigmaMax) {
        this.kind = kind;
        this.network = network;
        this.sigmaMin = sigmaMin;
        this.sigmaMax = sigmaMax;
    }

    /**
     * The variance-preserving SDE.
     *
     * @param noisePredictor {@code eps_hat(x_t, t)} with {@code t in [0, 1]} per row
     * @return the model
     */
    public static ScoreSde vp(Denoiser noisePredictor) {
        return new ScoreSde(Kind.VP, noisePredictor, 0, 0);
    }

    /**
     * The variance-exploding SDE.
     *
     * @param noisePredictor {@code eps_hat(x_t, t)} with {@code t in [0, 1]} per row
     * @param sigmaMin the noise at {@code t = 0}, such as 0.01
     * @param sigmaMax the noise at {@code t = 1}, about the largest data distance
     * @return the model
     */
    public static ScoreSde ve(Denoiser noisePredictor, double sigmaMin, double sigmaMax) {
        return new ScoreSde(Kind.VE, noisePredictor, sigmaMin, sigmaMax);
    }

    /** @return which SDE */
    public Kind kind() {
        return kind;
    }

    /**
     * The perturbation kernel {@code x_t = mean(t) x0 + std(t) n}.
     *
     * @param t the time
     * @return {@code {mean coefficient, standard deviation}}
     */
    public double[] marginal(double t) {
        if (kind == Kind.VP) {
            double integral = BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t * t;
            double mean = Math.exp(-0.5 * integral);
            return new double[] {mean, Math.sqrt(1 - mean * mean)};
        }
        return new double[] {1, sigmaMin * Math.pow(sigmaMax / sigmaMin, t)};
    }

    /**
     * The drift coefficient and squared diffusion at a time: {@code f(x, t) = drift * x},
     * {@code g(t)^2}.
     */
    private double[] coefficients(double t) {
        if (kind == Kind.VP) {
            double beta = BETA_MIN + t * (BETA_MAX - BETA_MIN);
            return new double[] {-0.5 * beta, beta};
        }
        double sigma = marginal(t)[1];
        return new double[] {0, sigma * sigma * 2 * Math.log(sigmaMax / sigmaMin)};
    }

    /**
     * Continuous-time denoising score matching: {@code t ~ U(eps, 1)}, and the noise predictor's
     * squared error (the score-matching loss with the likelihood-agnostic weight
     * {@code std(t)^2}).
     *
     * @param x0 clean data {@code [batch, dim]}
     * @param random the source of times and noise
     * @return the scalar loss
     */
    public Var loss(Tensor x0, RandomGenerator random) {
        int b = x0.shape().dim(0);
        float[] t = new float[b];
        float[] mean = new float[b];
        float[] std = new float[b];
        for (int i = 0; i < b; i++) {
            t[i] = (float) (epsilon + (1 - epsilon) * random.nextDouble());
            double[] m = marginal(t[i]);
            mean[i] = (float) m[0];
            std[i] = (float) m[1];
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor xt = x0.mul(GenerativeOps.column(mean)).add(noise.mul(GenerativeOps.column(std)));
        return network.apply(Var.of(xt), Tensor.of(Shape.of(b), t)).sub(Var.of(noise)).square().mean();
    }

    /**
     * The score {@code grad log p_t(x)}.
     *
     * @param x points {@code [batch, dim]}
     * @param t the time
     * @return the score
     */
    public Tensor score(Tensor x, double t) {
        int b = x.shape().dim(0);
        Tensor eps = network.apply(Var.of(x), Tensor.full(Shape.of(b), (float) t)).value();
        return eps.div((float) -marginal(t)[1]);
    }

    /**
     * A draw from the prior at {@code t = 1}.
     *
     * @param n how many
     * @param dim the dimension
     * @param random the noise source
     * @return {@code [n, dim]}
     */
    public Tensor prior(int n, int dim, RandomGenerator random) {
        return Tensor.normal(Shape.of(n, dim), 0, 1, random).mul((float) marginal(1)[1]);
    }

    /**
     * Predictor-corrector sampling: reverse-diffusion predictor steps, each followed by
     * Langevin corrector steps; the last step adds no noise.
     *
     * @param start the prior draw at {@code t = 1}
     * @param steps predictor steps
     * @param correctorSteps Langevin steps after each predictor step, 0 for none
     * @param snr the corrector's target signal-to-noise ratio, 0.16 in the paper
     * @param random the noise source
     * @return the samples; {@code steps * (1 + correctorSteps)} network evaluations
     */
    public Tensor samplePredictorCorrector(Tensor start, int steps, int correctorSteps, double snr,
            RandomGenerator random) {
        Tensor x = start;
        double dt = (1 - epsilon) / steps;
        for (int i = 0; i < steps; i++) {
            double t = 1 - i * dt;
            double next = t - dt;
            Tensor s = score(x, t);
            boolean last = i == steps - 1;
            // the reverse-diffusion (ancestral) predictor: the discretised reverse SDE that stays
            // stable at large steps, unlike plain Euler-Maruyama
            if (kind == Kind.VP) {
                double a = Math.pow(marginal(t)[0] / marginal(next)[0], 2);
                double beta = 1 - a;
                x = x.add(s.mul((float) beta)).div((float) Math.sqrt(a));
                if (!last) {
                    x = x.add(Tensor.normal(x.shape(), 0, 1, random).mul((float) Math.sqrt(beta)));
                }
            } else {
                double v = Math.pow(marginal(t)[1], 2) - Math.pow(marginal(next)[1], 2);
                x = x.add(s.mul((float) v));
                if (!last) {
                    x = x.add(Tensor.normal(x.shape(), 0, 1, random).mul((float) Math.sqrt(v)));
                }
            }
            if (!last) {
                for (int k = 0; k < correctorSteps; k++) {
                    Tensor g = score(x, next);
                    Tensor z = Tensor.normal(x.shape(), 0, 1, random);
                    double gNorm = Math.sqrt(g.square().sum().item() / x.shape().dim(0));
                    double zNorm = Math.sqrt(z.square().sum().item() / x.shape().dim(0));
                    double alpha = kind == Kind.VP ? Math.pow(marginal(t)[0] / marginal(next)[0], 2) : 1;
                    double step = 2 * alpha * Math.pow(snr * zNorm / Math.max(gNorm, 1e-8), 2);
                    x = x.add(g.mul((float) step)).add(z.mul((float) Math.sqrt(2 * step)));
                }
            }
        }
        return x;
    }

    /** The probability-flow velocity {@code dx/dt = f - g^2 score / 2} at a time. */
    private Tensor flow(Tensor x, double t) {
        double[] c = coefficients(t);
        return x.mul((float) c[0]).sub(score(x, t).mul((float) (0.5 * c[1])));
    }

    /**
     * Solves the probability-flow ODE from {@code t = 1} to {@code eps}.
     *
     * @param start the prior draw
     * @param steps the number of steps, uniform in time
     * @param heun Heun's method (two evaluations per step but the last) instead of Euler
     * @return the samples
     */
    public Tensor sampleOde(Tensor start, int steps, boolean heun) {
        Tensor x = start;
        double dt = (1 - epsilon) / steps;
        for (int i = 0; i < steps; i++) {
            double t = 1 - i * dt;
            Tensor v = flow(x, t);
            Tensor euler = x.sub(v.mul((float) dt));
            if (heun && i < steps - 1) {
                x = x.sub(v.add(flow(euler, t - dt)).mul((float) (dt / 2)));
            } else {
                x = euler;
            }
        }
        return x;
    }

    /**
     * The probability-flow ODE as a velocity field on {@code tau in [0, 1]} (data at 0, the
     * prior at 1), the time convention of core {@code FlowMatching} and {@link ContinuousFlow},
     * so {@code ContinuousFlow.of(sde.probabilityFlow(), ...)} gives exact log-likelihoods.
     * For VE the prior is not unit variance; {@link #priorLogScale} corrects for it.
     *
     * @return the field; gradients flow into the network
     */
    public Denoiser probabilityFlow() {
        return (x, tau) -> {
            float[] tv = tau.toArray();
            float[] time = new float[tv.length];
            float[] drift = new float[tv.length];
            float[] scoreScale = new float[tv.length];
            for (int i = 0; i < tv.length; i++) {
                double t = epsilon + (1 - epsilon) * tv[i];
                double[] c = coefficients(t);
                time[i] = (float) t;
                drift[i] = (float) (c[0] * (1 - epsilon));
                // -g^2/2 * score = g^2/2 * eps / std
                scoreScale[i] = (float) (0.5 * c[1] / marginal(t)[1] * (1 - epsilon));
            }
            Var eps = network.apply(x, Tensor.of(Shape.of(tv.length), time));
            return x.mul(GenerativeOps.column(drift)).add(eps.mul(GenerativeOps.column(scoreScale)));
        };
    }

    /**
     * The log-density correction for a prior that is {@code N(0, s^2 I)} instead of
     * {@code N(0, I)}: add {@code -dim * ln s} and rescale the latent, which this returns.
     *
     * @return {@code s}, the prior's standard deviation at {@code t = 1}
     */
    public double priorLogScale() {
        return marginal(1)[1];
    }
}
