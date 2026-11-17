package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Bridge matching between two arbitrary distributions, beside core {@code FlowMatching}
 * (which only goes between data and noise): stochastic interpolants (Albergo, Boffi and
 * Vanden-Eijnden, <a href="https://arxiv.org/abs/2303.08797">arXiv:2303.08797</a>) and the
 * diffusion Schrodinger bridge by iterative Markovian fitting (DSBM, Shi et al.,
 * <a href="https://arxiv.org/abs/2303.16852">arXiv:2303.16852</a>).
 *
 * <p>Given pairs {@code (x0, x1)} from a coupling of the source and target, the Brownian
 * bridge {@code x_t = (1 - t) x0 + t x1 + sigma sqrt(t (1 - t)) z} is pinned at both ends; a
 * drift network {@code b(x, t)} learns its conditional drift {@code (x1 - x_t) / (1 - t)}
 * by regression ({@link #loss}). Simulating {@code dX = b dt + sigma dW} from source samples
 * ({@link #transport}) then reaches the target distribution: this is the <em>Markovian
 * projection</em> of the bridge mixture. With {@code sigma = 0} it is rectified flow. DSBM
 * alternates that projection with the <em>reciprocal projection</em>: new pairs are made by
 * transporting fresh source samples with the current model ({@link #couple}) and the next
 * model is fitted to bridges between those pairs, which converges to the entropic optimal
 * transport (the Schrodinger bridge), so paths become short as well as correct. Rectified
 * flow's reflow is the {@code sigma = 0} special case. Time runs from the source (0) to the
 * target (1).
 */
public final class BridgeMatching {
    private static final float T_MAX = 0.99f;

    private BridgeMatching() {
    }

    /**
     * The bridge-matching loss on a batch of pairs.
     *
     * @param drift the drift network {@code b(x, t)}, times per row in [0, 1)
     * @param x0 source points {@code [n, dim]}
     * @param x1 their paired target points
     * @param sigma the bridge's diffusion coefficient, 0 for rectified flow
     * @param random the source of times and bridge noise
     * @return the scalar loss
     */
    public static Var loss(Denoiser drift, Tensor x0, Tensor x1, float sigma, RandomGenerator random) {
        int n = x0.shape().dim(0);
        float[] t = new float[n];
        float[] keep = new float[n];
        float[] spread = new float[n];
        float[] back = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) (random.nextDouble() * T_MAX);
            keep[i] = 1 - t[i];
            spread[i] = (float) (sigma * Math.sqrt(t[i] * (1 - t[i])));
            back[i] = (float) (sigma * Math.sqrt(t[i] / (1 - t[i])));
        }
        Tensor z = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor xt = x0.mul(GenerativeOps.column(keep)).add(x1.mul(GenerativeOps.column(t)))
                .add(z.mul(GenerativeOps.column(spread)));
        // (x1 - x_t) / (1 - t) = x1 - x0 - sigma sqrt(t / (1 - t)) z
        Tensor target = x1.sub(x0).sub(z.mul(GenerativeOps.column(back)));
        return drift.apply(Var.of(xt), Tensor.of(Shape.of(n), t)).sub(Var.of(target)).square().mean();
    }

    /**
     * Transports source points with Euler-Maruyama steps of {@code dX = b dt + sigma dW}; the
     * last step adds no noise.
     *
     * @param drift the trained drift
     * @param x0 source points
     * @param steps the number of steps
     * @param sigma the diffusion coefficient used in training
     * @param random the noise source
     * @return the transported points
     */
    public static Tensor transport(Denoiser drift, Tensor x0, int steps, float sigma, RandomGenerator random) {
        int n = x0.shape().dim(0);
        Tensor x = x0;
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            float t = Math.min(i * dt, T_MAX);
            Tensor b = drift.apply(Var.of(x), Tensor.full(Shape.of(n), t)).value();
            x = x.add(b.mul(dt));
            if (sigma > 0 && i < steps - 1) {
                x = x.add(Tensor.normal(x.shape(), 0, (float) (sigma * Math.sqrt(dt)), random));
            }
        }
        return x;
    }

    /**
     * The reciprocal step of iterative Markovian fitting (and reflow): pairs each source point
     * with where the current model transports it.
     *
     * @param drift the current drift
     * @param x0 fresh source points
     * @param steps transport steps
     * @param sigma the diffusion coefficient
     * @param random the noise source
     * @return {@code {x0, x1}}, the new coupling
     */
    public static Tensor[] couple(Denoiser drift, Tensor x0, int steps, float sigma, RandomGenerator random) {
        return new Tensor[] {x0, transport(drift, x0, steps, sigma, random)};
    }
}
