package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Equilibrium Matching (EqM, Wang and Du, <a href="https://arxiv.org/abs/2510.02300">2510.02300</a>):
 * a generative model that learns one <em>time-independent</em> gradient field {@code f(x)}
 * whose descent leads from noise to data, the gradient of an implicit energy landscape
 * with the data at its minima.
 *
 * <p>Training: {@code x = g x0 + (1 - g) eps} for {@code g ~ U(0, 1)}, and {@code f(x)} is
 * regressed onto {@code (eps - x0) c(g)}, where the scale {@code c} is constant
 * ({@code lambda}) up to {@code g = a} and then decays linearly to 0 at the data, so the
 * field vanishes at the data (an equilibrium). The network is never told {@code g}; it
 * receives time 0 (the core {@link Denoiser} signature keeps it a drop-in network).
 *
 * <p>Sampling is gradient descent, {@code x <- x - eta f(x)}, optionally with Nesterov
 * momentum, from noise. Because the field is time-free, the number of steps need not be
 * fixed in advance: {@link #sample} can stop each sample once {@code |f(x)|} falls below a
 * threshold (adaptive compute), and descent can start from any point, which also makes
 * denoising and inpainting natural.
 */
public final class EquilibriumMatching {
    private final float lambda;
    private final float decayStart;

    private EquilibriumMatching(float lambda, float decayStart) {
        this.lambda = lambda;
        this.decayStart = decayStart;
    }

    /**
     * Creates the objective with the paper's truncated-decay scale.
     *
     * @param lambda the gradient scale far from the data (4 in the paper)
     * @param decayStart where the decay starts, {@code a} (0.8 in the paper)
     * @return the objective
     */
    public static EquilibriumMatching of(float lambda, float decayStart) {
        return new EquilibriumMatching(lambda, decayStart);
    }

    /**
     * The target scale.
     *
     * @param g the interpolation weight of the data, 0 noise to 1 data
     * @return {@code c(g)}
     */
    public double scale(double g) {
        return g <= decayStart ? lambda : lambda * (1 - g) / (1 - decayStart);
    }

    /** @return lambda */
    public float lambda() {
        return lambda;
    }

    /**
     * The training loss for a batch.
     *
     * @param field the gradient field network (it receives time 0)
     * @param x0 clean data, batch first
     * @param random the source of interpolation weights and noise
     * @return the loss, a scalar
     */
    public Var loss(Denoiser field, Tensor x0, RandomGenerator random) {
        int batch = x0.shape().dim(0);
        float[] g = new float[batch];
        float[] keep = new float[batch];
        float[] c = new float[batch];
        for (int i = 0; i < batch; i++) {
            g[i] = (float) random.nextDouble();
            keep[i] = 1 - g[i];
            c[i] = (float) scale(g[i]);
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device());
        Tensor x = x0.mul(column(g, x0)).add(noise.mul(column(keep, x0)));
        Tensor target = noise.sub(x0).mul(column(c, x0));
        Var out = field.apply(Var.of(x), Tensor.zeros(Shape.of(batch)).to(x0.device()));
        return out.sub(Var.of(target)).square().mean();
    }

    private static Tensor column(float[] v, Tensor like) {
        int[] dims = new int[like.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = v.length;
        return Tensor.of(Shape.of(dims), v.clone(), like.device());
    }

    /** A sampling result: the samples and the mean number of field evaluations per sample. */
    public record Samples(Tensor samples, double meanSteps) {
    }

    /**
     * Gradient descent on the learned landscape.
     *
     * @param field the trained field
     * @param start the starting points (noise), batch first
     * @param maxSteps the most descent steps
     * @param stepSize eta
     * @param momentum Nesterov momentum, 0 for plain descent
     * @param threshold stop a sample once the field's mean squared value per dimension falls
     *     below this; 0 runs every step
     * @return the samples and the steps they took
     */
    public Samples sample(Denoiser field, Tensor start, int maxSteps, float stepSize, float momentum,
            float threshold) {
        int batch = start.shape().dim(0);
        int width = Math.toIntExact(start.size() / batch);
        float[] x = start.toArray();
        float[] prev = x.clone();
        boolean[] stopped = new boolean[batch];
        long evaluations = 0;
        Tensor time = Tensor.zeros(Shape.of(batch));
        for (int s = 0; s < maxSteps; s++) {
            float[] y = x.clone();
            if (momentum > 0) {
                for (int i = 0; i < y.length; i++) {
                    y[i] += momentum * (x[i] - prev[i]);
                }
            }
            float[] f = field.apply(Var.of(Tensor.of(start.shape(), y)), time).value().toArray();
            boolean any = false;
            for (int b = 0; b < batch; b++) {
                if (stopped[b]) {
                    continue;
                }
                evaluations++;
                double norm = 0;
                for (int k = 0; k < width; k++) {
                    norm += f[b * width + k] * f[b * width + k];
                }
                if (threshold > 0 && norm / width < threshold) {
                    stopped[b] = true;
                    continue;
                }
                any = true;
                for (int k = 0; k < width; k++) {
                    int i = b * width + k;
                    prev[i] = x[i];
                    x[i] = y[i] - stepSize * f[i];
                }
            }
            if (!any) {
                break;
            }
        }
        return new Samples(Tensor.of(start.shape(), x), (double) evaluations / batch);
    }
}
