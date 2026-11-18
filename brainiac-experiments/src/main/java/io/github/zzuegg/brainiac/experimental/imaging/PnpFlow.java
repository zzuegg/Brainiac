package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * PnP-Flow (Martin, Gagneux, Hagemann &amp; Steidl, ICLR 2025,
 * <a href="https://arxiv.org/abs/2410.02423">arXiv:2410.02423</a>): a flow-matching prior trained
 * with core {@link FlowMatching} (data at {@code t = 0}, noise at {@code t = 1}, velocity
 * {@code noise - x0}) used as a plug-and-play prior. Each iteration takes a gradient step on
 * {@code 1/2 |A x - y|^2}, moves the estimate back onto the flow path at time {@code t}
 * ({@code (1 - t) z + t noise}) and applies the flow's endpoint denoiser
 * {@code D(x_t, t) = x_t - t v(x_t, t)}, with {@code t} decreasing from 1 to 0; it never
 * differentiates through the ODE.
 *
 * <p>The same model serves as a variance-exploding diffusion prior for
 * {@link PosteriorSamplers} ({@link #score}: {@code sigma = t / (1 - t)}) and as a Gaussian
 * denoiser for {@link OperatorPlugAndPlay} ({@link #prior}).
 */
public final class PnpFlow {

    private final int side;
    private final ImagingCnn net;
    private final Denoiser model;

    /**
     * Creates an untrained prior: a CNN on the image plus a constant time channel.
     *
     * @param side the image side
     * @param hidden the hidden channels
     * @param depth the convolutions
     * @param seed the seed
     */
    public PnpFlow(int side, int hidden, int depth, long seed) {
        this.side = side;
        ImagingCnn cnn = new ImagingCnn(2, hidden, 1, depth, false, new SplittableRandom(seed));
        this.net = cnn;
        this.model = cnn::forward;
    }

    /**
     * Trains the velocity model with the flow-matching loss.
     *
     * @param images the training images
     * @param steps the Adam steps
     * @param batch the batch size
     * @param lr the learning rate
     * @param seed the seed
     */
    public void fit(List<double[]> images, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int n = side * side;
        for (int s = 0; s < steps; s++) {
            float[] xb = new float[batch * n];
            for (int i = 0; i < batch; i++) {
                double[] img = images.get(r.nextInt(images.size()));
                for (int j = 0; j < n; j++) {
                    xb[i * n + j] = (float) img[j];
                }
            }
            Var loss = FlowMatching.loss(model, Tensor.of(Shape.of(batch, 1, side, side), xb), r);
            loss.backward();
            opt.step(net.parameters());
        }
    }

    /**
     * The predicted clean image.
     *
     * @param xt the point on the path
     * @param t its time
     * @return {@code x_t - t v(x_t, t)}
     */
    public double[] endpoint(double[] xt, double t) {
        Tensor x = Tensor.of(Shape.of(1, 1, side, side), ImageMath.toFloat(xt));
        float[] v = Var.noGrad(() -> model.apply(Var.of(x), Tensor.of(Shape.of(1), new float[] {(float) t})))
                .value().toArray();
        double[] out = new double[xt.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = xt[i] - t * v[i];
        }
        return out;
    }

    /** @return the model as a variance-exploding denoiser {@code E[x0 | x0 + sigma e]} */
    public PosteriorSamplers.Score score() {
        return (x, sigma) -> {
            double t = sigma / (1 + sigma);
            double[] xt = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                xt[i] = (1 - t) * x[i];
            }
            return endpoint(xt, t);
        };
    }

    /** @return the model as a Gaussian denoiser for plug-and-play */
    public OperatorPlugAndPlay.Prior prior() {
        PosteriorSamplers.Score s = score();
        return s::denoise;
    }

    /**
     * Restores an image with PnP-Flow.
     *
     * @param a the forward model
     * @param y the measurements
     * @param iterations the iterations (times {@code 1 - k / iterations})
     * @param stepScale the data step as a share of {@code 1 / |A|^2}
     * @param r the noise source
     * @return the estimate
     */
    public double[] restore(LinearOperator a, double[] y, int iterations, double stepScale, SplittableRandom r) {
        double gamma = stepScale / a.normSquared(30);
        double[] x = a.adjoint(y);
        for (int k = 0; k < iterations; k++) {
            double t = 1 - k / (double) iterations;
            double[] res = ImageMath.axpy(a.apply(x), -1, y);
            double[] z = ImageMath.axpy(x, -gamma, a.adjoint(res));
            double[] zt = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                zt[i] = (1 - t) * z[i] + t * PosteriorSamplers.gaussian(r);
            }
            x = endpoint(zt, t);
        }
        return x;
    }

    /** @return the trainable values */
    public long parameterCount() {
        return net.parameterCount();
    }
}
