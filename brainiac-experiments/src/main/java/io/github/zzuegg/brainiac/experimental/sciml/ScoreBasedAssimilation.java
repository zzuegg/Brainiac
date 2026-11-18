package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.VectorConditionedDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Score-based data assimilation (Rozet and Louppe, NeurIPS 2023): a diffusion model is trained
 * on short windows of trajectories (no equations needed), and whole posterior trajectory
 * windows are sampled given sparse, noisy observations by guiding the learned score with the
 * observation likelihood. At each reverse step the denoised estimate {@code x0^(x_t)} gives the
 * approximate likelihood
 *
 * <pre>
 * p(y | x_t) ~ N(y | H x0^(x_t), sigma_y^2 + gamma (1 - abar_t) / abar_t)
 * </pre>
 *
 * <p>(the SDA approximation, whose variance grows with the noise level), and its gradient with
 * respect to {@code x_t}, by autograd through the denoiser, is added to the score before a
 * DDIM step. The denoiser is the experimental {@link VectorConditionedDenoiser} trained with
 * the core {@link Diffusion} noise-prediction loss on a cosine {@link NoiseSchedule}.
 *
 * <p>Windows are flattened {@code [window * state]} and standardised per state component with
 * statistics from the training data.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2306.10574">arXiv:2306.10574</a>.
 */
public final class ScoreBasedAssimilation {

    /** Denoised estimates are clipped to this many standard deviations. */
    private static final float CLIP = 4f;

    private final int state;
    private final int window;
    private final VectorConditionedDenoiser network;
    private final Diffusion diffusion;
    private final double[] mean;
    private final double[] std;

    private ScoreBasedAssimilation(int state, int window, int width, int depth, int steps, double[] mean,
            double[] std, long seed) {
        this.state = state;
        this.window = window;
        this.network = VectorConditionedDenoiser.of(state * window, 0, width, depth, seed).timeScale(1f / steps);
        this.diffusion = new Diffusion(NoiseSchedule.cosine(steps), Diffusion.Prediction.NOISE);
        this.mean = mean.clone();
        this.std = std.clone();
    }

    /**
     * Creates the model with normalisation statistics from a training trajectory.
     *
     * @param trajectory states {@code [T][state]} used for the statistics
     * @param window the window length
     * @param width the denoiser width
     * @param depth the denoiser's residual layers
     * @param steps diffusion steps
     * @param seed the seed
     * @return the model
     */
    public static ScoreBasedAssimilation of(double[][] trajectory, int window, int width, int depth, int steps,
            long seed) {
        int d = trajectory[0].length;
        double[] m = new double[d];
        double[] s = new double[d];
        for (double[] x : trajectory) {
            for (int i = 0; i < d; i++) {
                m[i] += x[i] / trajectory.length;
            }
        }
        for (double[] x : trajectory) {
            for (int i = 0; i < d; i++) {
                s[i] += (x[i] - m[i]) * (x[i] - m[i]) / trajectory.length;
            }
        }
        for (int i = 0; i < d; i++) {
            s[i] = Math.sqrt(s[i]) + 1e-9;
        }
        return new ScoreBasedAssimilation(d, window, width, depth, steps, m, s, seed);
    }

    private Denoiser denoiser() {
        return (noisy, time) -> network.apply(noisy, time, null);
    }

    /** @return the denoiser's parameters */
    public List<Var> parameters() {
        return network.parameters();
    }

    /**
     * Random normalised training windows from a trajectory.
     *
     * @param trajectory states {@code [T][state]}
     * @param count the windows
     * @param random the source
     * @return {@code [count, window * state]}
     */
    public Tensor windows(double[][] trajectory, int count, SplittableRandom random) {
        float[] f = new float[count * window * state];
        for (int c = 0; c < count; c++) {
            int s = random.nextInt(trajectory.length - window + 1);
            for (int t = 0; t < window; t++) {
                for (int i = 0; i < state; i++) {
                    f[(c * window + t) * state + i] = (float) ((trajectory[s + t][i] - mean[i]) / std[i]);
                }
            }
        }
        return Tensor.of(Shape.of(count, window * state), f);
    }

    /**
     * The diffusion training loss on a batch of windows.
     *
     * @param windows {@code [b, window * state]} from {@link #windows}
     * @param random the source of steps and noise
     * @return the loss
     */
    public Var loss(Tensor windows, SplittableRandom random) {
        return diffusion.loss(denoiser(), windows, random);
    }

    /**
     * Samples posterior windows given observations.
     *
     * @param observations per window, observed values {@code [b][window][state]} (in data
     *     units; ignored where not observed)
     * @param observed which entries are observed {@code [window][state]}, shared by the batch
     * @param noise the observation noise standard deviation (data units)
     * @param gamma the SDA variance factor, for example 0.1 to 1
     * @param steps reverse DDIM steps
     * @param random the source
     * @return posterior samples {@code [b][window][state]} in data units
     */
    public double[][][] sample(double[][][] observations, boolean[][] observed, double noise, double gamma, int steps,
            SplittableRandom random) {
        int b = observations.length;
        int dim = window * state;
        float[] y = new float[b * dim];
        float[] mask = new float[b * dim];
        float[] obsVar = new float[dim];
        for (int t = 0; t < window; t++) {
            for (int i = 0; i < state; i++) {
                obsVar[t * state + i] = (float) (noise * noise / (std[i] * std[i]));
            }
        }
        for (int k = 0; k < b; k++) {
            for (int t = 0; t < window; t++) {
                for (int i = 0; i < state; i++) {
                    if (observed[t][i]) {
                        int idx = k * dim + t * state + i;
                        y[idx] = (float) ((observations[k][t][i] - mean[i]) / std[i]);
                        mask[idx] = 1;
                    }
                }
            }
        }
        Tensor yt = Tensor.of(Shape.of(b, dim), y);
        Tensor mt = Tensor.of(Shape.of(b, dim), mask);
        NoiseSchedule schedule = diffusion.schedule();
        int total = schedule.steps();
        Tensor x = Tensor.normal(Shape.of(b, dim), 0, 1, random);
        for (int s = steps - 1; s >= 0; s--) {
            int t = (int) Math.round((double) s * (total - 1) / Math.max(1, steps - 1));
            int prev = s == 0 ? -1 : (int) Math.round((double) (s - 1) * (total - 1) / Math.max(1, steps - 1));
            double ab = schedule.alphaBar(t);
            double abPrev = prev < 0 ? 1 : schedule.alphaBar(prev);
            Var xt = Var.param(x);
            Var eps = network.apply(xt, Tensor.full(Shape.of(b), t), null);
            Var x0 = xt.sub(eps.mul((float) Math.sqrt(1 - ab))).div((float) Math.sqrt(ab));
            float extra = (float) (gamma * (1 - ab) / ab);
            float[] var = new float[dim];
            for (int j = 0; j < dim; j++) {
                var[j] = obsVar[j] + extra;
            }
            Var r = x0.sub(yt).mul(mt);
            Var logLik = r.square().div(Var.of(Tensor.of(Shape.of(dim), var))).sum().mul(-0.5f);
            Tensor grad = Var.gradients(logLik, List.of(xt), false).get(0).value();
            // score of x_t is -eps / sqrt(1 - ab); adding the likelihood score shifts eps
            Tensor guided = eps.value().sub(grad.mul((float) Math.sqrt(1 - ab)));
            // clip the denoised estimate to the normalised data range (as DDPM/DDIM samplers do):
            // early steps divide by a tiny sqrt(alphaBar) and would otherwise blow up
            float[] x0v = x.sub(guided.mul((float) Math.sqrt(1 - ab))).div((float) Math.sqrt(ab)).toArray();
            for (int j = 0; j < x0v.length; j++) {
                x0v[j] = Math.clamp(x0v[j], -CLIP, CLIP);
            }
            Tensor x0g = Tensor.of(Shape.of(b, dim), x0v);
            Tensor epsC = x.sub(x0g.mul((float) Math.sqrt(ab))).div((float) Math.sqrt(Math.max(1e-12, 1 - ab)));
            x = x0g.mul((float) Math.sqrt(abPrev)).add(epsC.mul((float) Math.sqrt(1 - abPrev)));
        }
        float[] out = x.toArray();
        double[][][] res = new double[b][window][state];
        for (int k = 0; k < b; k++) {
            for (int t = 0; t < window; t++) {
                for (int i = 0; i < state; i++) {
                    res[k][t][i] = out[k * dim + t * state + i] * std[i] + mean[i];
                }
            }
        }
        return res;
    }

    /** @return the window length */
    public int window() {
        return window;
    }

    @Override
    public String toString() {
        return "ScoreBasedAssimilation(state=" + state + ", window=" + window + ")";
    }
}
