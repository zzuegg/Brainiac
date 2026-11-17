package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * Gradient penalties for GAN critics, computed without second-order autograd:
 *
 * <ul>
 *   <li>{@link #wgan}: WGAN-GP's {@code E[(||grad_x D(x_hat)|| - 1)^2]} at random points
 *       {@code x_hat} between real and generated samples, which softly enforces the
 *       1-Lipschitz constraint of the Wasserstein critic. Gulrajani et al., "Improved
 *       Training of Wasserstein GANs", <a href="https://arxiv.org/abs/1704.00028">arXiv:1704.00028</a>.</li>
 *   <li>{@link #r1}: {@code E[||grad_x D(x)||^2]} on real data, the zero-centred penalty of
 *       StyleGAN. Mescheder et al., <a href="https://arxiv.org/abs/1801.04406">arXiv:1801.04406</a>.</li>
 * </ul>
 *
 * <p>Both need the gradient of {@code ||grad_x D||} with respect to the critic's weights
 * ("double backprop"). Brainiac's autograd is first order, so this class computes
 * {@code g = grad_x D} with an ordinary backward pass, fixes its direction {@code u = g / ||g||},
 * and measures the norm as a central-difference directional derivative,
 * {@code ||g|| = u . g ~ (D(x + h u) - D(x - h u)) / 2h}, built from ordinary forward passes,
 * so it is differentiable in the weights. Since {@code u} is a unit vector, the derivative of
 * {@code u} itself contributes nothing at first order, and the weight gradient is exact up to
 * {@code O(h^2)}.
 */
public final class GradientPenalty {

    private GradientPenalty() {
    }

    /**
     * The WGAN-GP penalty.
     *
     * @param critic the critic, mapping {@code [batch, ...]} to {@code [batch, 1]} or {@code [batch]}
     * @param criticParameters the critic's parameters, whose gradients this leaves unchanged
     * @param real real samples
     * @param fake generated samples, same shape
     * @param random the source of the interpolation weights
     * @return the penalty, differentiable in the critic's parameters
     */
    public static Var wgan(UnaryOperator<Var> critic, List<Var> criticParameters, Tensor real, Tensor fake,
            RandomGenerator random) {
        int b = real.shape().dim(0);
        int[] dims = real.shape().toArray();
        java.util.Arrays.fill(dims, 1, dims.length, 1);
        float[] eps = new float[b];
        for (int i = 0; i < b; i++) {
            eps[i] = random.nextFloat();
        }
        Tensor e = Tensor.of(Shape.of(dims), eps, real.device());
        Tensor xhat = real.mul(e).add(fake.mul(e.neg().add(1f)));
        return gradientNorm(critic, criticParameters, xhat, 1e-2f).sub(1f).square().mean();
    }

    /**
     * The R1 penalty {@code mean ||grad_x D(x)||^2} on real samples.
     *
     * @param critic the critic
     * @param criticParameters the critic's parameters
     * @param real real samples
     * @return the penalty
     */
    public static Var r1(UnaryOperator<Var> critic, List<Var> criticParameters, Tensor real) {
        return gradientNorm(critic, criticParameters, real, 1e-2f).square().mean();
    }

    /**
     * The per-sample input-gradient norm of a critic as a differentiable function of its
     * parameters.
     *
     * @param critic the critic
     * @param criticParameters the critic's parameters, whose gradients this leaves unchanged
     * @param x the points {@code [batch, ...]}
     * @param h the finite-difference step along the gradient direction
     * @return {@code [batch]} gradient norms
     */
    public static Var gradientNorm(UnaryOperator<Var> critic, List<Var> criticParameters, Tensor x, float h) {
        int b = x.shape().dim(0);
        Tensor[] saved = new Tensor[criticParameters.size()];
        for (int i = 0; i < saved.length; i++) {
            saved[i] = criticParameters.get(i).grad();
        }
        Var xv = Var.param(x);
        critic.apply(xv).sum().backward();
        Tensor g = xv.grad();
        for (int i = 0; i < saved.length; i++) {
            criticParameters.get(i).setGrad(saved[i]);
        }
        int[] dims = x.shape().toArray();
        int per = Math.toIntExact(x.size() / b);
        Tensor flat = g.reshape(b, per);
        Tensor norm = flat.square().sum(1, true).sqrt().add(1e-12f);
        Tensor u = flat.div(norm).reshape(dims).mul(h);
        Var plus = critic.apply(Var.of(x.add(u))).reshape(b);
        Var minus = critic.apply(Var.of(x.sub(u))).reshape(b);
        return plus.sub(minus).mul(1f / (2 * h));
    }
}
