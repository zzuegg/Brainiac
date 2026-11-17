package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A differentiable 2-D Gaussian-splat rasterizer, the 2-D core of 3D Gaussian Splatting
 * (Kerbl et al., <a href="https://arxiv.org/abs/2308.04079">2308.04079</a>). A picture is a set
 * of {@code n} anisotropic Gaussians, each with a centre, two scales, a rotation, a colour
 * and an opacity. A pixel's colour is the front-to-back alpha composite
 * {@code C = sum_i c_i a_i T_i}, {@code T_i = prod_{j<i} (1 - a_j)}, with
 * {@code a_i = opacity_i * exp(-q_i / 2)} and {@code q_i} the Mahalanobis distance of the pixel
 * to Gaussian {@code i}. Every step is a {@link Var} operation (the transmittance product
 * is an exclusive cumulative sum of {@code log(1 - a)}, done as a matmul with a triangular
 * matrix), so fitting a picture is plain gradient descent on the splat parameters.
 *
 * <p>Rotations are stored as an unnormalized 2-vector {@code (a, b)}, so
 * {@code cos = a / |(a, b)|} and {@code sin = b / |(a, b)|} need no trigonometry.
 *
 * <p>As a {@link Layer} it maps pixel coordinates {@code [pixels, 2]} in [0, 1] to colours
 * {@code [pixels, channels]}.
 */
public final class GaussianSplats2d implements Layer {
    private final int count;
    private final int channels;
    private final Var centre;    // [1, n] x and y, stored as [2, n]
    private final Var logScale;  // [2, n]
    private final Var rotation;  // [2, n]
    private final Var colour;    // [n, channels]
    private final Var opacity;   // [1, n] logits
    private final Tensor upper;  // [n, n], 1 where row < column

    private GaussianSplats2d(int count, int channels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.count = count;
        this.channels = channels;
        centre = Var.param(Tensor.uniform(Shape.of(2, count), 0.05f, 0.95f, r));
        logScale = Var.param(Tensor.full(Shape.of(2, count), (float) Math.log(0.6 / Math.sqrt(count))));
        float[] rot = new float[2 * count];
        for (int i = 0; i < count; i++) {
            double a = r.nextDouble() * Math.PI;
            rot[i] = (float) Math.cos(a);
            rot[count + i] = (float) Math.sin(a);
        }
        rotation = Var.param(Tensor.of(Shape.of(2, count), rot));
        colour = Var.param(Tensor.uniform(Shape.of(count, channels), 0.3f, 0.7f, r));
        opacity = Var.param(Tensor.zeros(Shape.of(1, count)));
        float[] u = new float[count * count];
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                u[i * count + j] = 1;
            }
        }
        upper = Tensor.of(Shape.of(count, count), u);
    }

    /**
     * Creates randomly placed splats.
     *
     * @param count the number of Gaussians
     * @param channels colour channels (1 for grey, 3 for RGB)
     * @param seed the seed
     * @return the splats
     */
    public static GaussianSplats2d of(int count, int channels, long seed) {
        return new GaussianSplats2d(count, channels, seed);
    }

    /**
     * Pixel-centre coordinates of a {@code width x height} image, row-major, in [0, 1].
     *
     * @param width the width
     * @param height the height
     * @return {@code [width * height, 2]} as (x, y)
     */
    public static Tensor grid(int width, int height) {
        float[] g = new float[width * height * 2];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                g[(y * width + x) * 2] = (x + 0.5f) / width;
                g[(y * width + x) * 2 + 1] = (y + 0.5f) / height;
            }
        }
        return Tensor.of(Shape.of(width * height, 2), g);
    }

    @Override
    public Var forward(Var pixels) {
        if (pixels.shape().rank() != 2 || pixels.shape().dim(1) != 2) {
            throw new IllegalArgumentException("GaussianSplats2d expects [pixels, 2], got " + pixels.shape());
        }
        Var dx = pixels.slice(1, 0, 1).sub(centre.slice(0, 0, 1));   // [p, n]
        Var dy = pixels.slice(1, 1, 2).sub(centre.slice(0, 1, 2));
        Var a = rotation.slice(0, 0, 1);
        Var b = rotation.slice(0, 1, 2);
        Var norm = a.square().add(b.square()).add(1e-8f).sqrt();
        Var cos = a.div(norm);
        Var sin = b.div(norm);
        Var u = dx.mul(cos).add(dy.mul(sin)).div(logScale.slice(0, 0, 1).exp());    // rotate into the splat's frame
        Var v = dy.mul(cos).sub(dx.mul(sin)).div(logScale.slice(0, 1, 2).exp());
        Var q = u.square().add(v.square());
        Var alpha = q.mul(-0.5f).exp().mul(opacity.sigmoid()).mul(0.99f);   // [p, n], below 1 for the log
        Var logT = alpha.neg().add(1f).log().matmul(Var.of(upper.to(pixels.value().device())));   // exclusive cumsum
        Var weight = alpha.mul(logT.exp());
        return weight.matmul(colour);
    }

    @Override
    public List<Var> parameters() {
        return List.of(centre, logScale, rotation, colour, opacity);
    }

    @Override
    public int outputSize(int inputSize) {
        return channels;
    }

    /** @return the number of Gaussians */
    public int count() {
        return count;
    }
}
