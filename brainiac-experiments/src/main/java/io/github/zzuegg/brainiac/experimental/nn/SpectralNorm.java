package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Spectral normalization: divides a weight matrix by an estimate of its largest singular
 * value {@code sigma}, so the layer is (about) 1-Lipschitz. The estimate comes from power
 * iteration that keeps its vector {@code u} between calls, so one iteration per training
 * step suffices. The gradient flows through {@code sigma = u^T W v} with {@code u, v} held
 * fixed, as in the paper. The standard way to stabilize GAN discriminators and to bound a
 * network's sensitivity to its input.
 *
 * <p>Use one instance per weight: {@code Var w = sn.normalize(weight)} in the forward pass.
 * Weights of any rank work: they are flattened to {@code [dim(0), rest]}.
 *
 * <p>Paper: Miyato et al., "Spectral Normalization for Generative Adversarial Networks"
 * (2018), <a href="https://arxiv.org/abs/1802.05957">arXiv:1802.05957</a>.
 */
public final class SpectralNorm {

    private final int iterations;
    private Tensor u;
    private float lastSigma = Float.NaN;

    /**
     * Creates the normalizer.
     *
     * @param rows the first dimension of the weight it will normalize
     * @param iterations power iterations per call, usually 1
     * @param random the random source of the starting vector
     */
    public SpectralNorm(int rows, int iterations, RandomGenerator random) {
        if (iterations < 1) {
            throw new IllegalArgumentException("need at least one iteration, got " + iterations);
        }
        this.iterations = iterations;
        this.u = unit(Tensor.normal(Shape.of(1, rows), 0, 1, random));
    }

    private static Tensor unit(Tensor v) {
        return v.div(v.square().sum().sqrt().add(1e-12f));
    }

    /**
     * Returns {@code weight / sigma(weight)}.
     *
     * @param weight the weight, {@code [rows, ...]}
     * @return the normalized weight, same shape, differentiable in {@code weight}
     */
    public Var normalize(Var weight) {
        int rows = weight.shape().dim(0);
        int cols = Math.toIntExact(weight.shape().size() / rows);
        Var w = weight.reshape(rows, cols);
        Tensor wt = w.value();
        Tensor v = null;
        Tensor uu = u.to(wt.device());
        for (int i = 0; i < iterations; i++) {
            v = unit(uu.matmul(wt));                      // [1, cols]
            uu = unit(v.matmul(wt.transpose()));          // [1, rows]
        }
        u = uu;
        Var sigma = Var.of(uu).matmul(w).matmul(Var.of(v.transpose()));   // [1, 1]
        lastSigma = sigma.value().item();
        return w.div(sigma.reshape(1)).reshape(weight.shape().toArray());
    }

    /** @return the singular value estimated by the last {@link #normalize} call */
    public float lastSigma() {
        return lastSigma;
    }
}
