package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Approximates the polar factor {@code U V^T} of a matrix {@code G = U S V^T}, the step
 * direction of Muon-family optimizers. The core {@code Muon} hard-codes Keller Jordan's
 * five-step Newton-Schulz quintic; this interface lets {@link OrthogonalMuon},
 * {@link NorMuon} and {@link AdaptiveMuon} take any orthogonaliser, such as the
 * minimax-optimal {@link PolarExpress}.
 */
@FunctionalInterface
public interface Orthogonalizer {

    /**
     * @param g a matrix {@code [rows, cols]}
     * @return its approximate polar factor, same shape
     */
    Tensor orthogonalize(Tensor g);

    /**
     * Muon's Newton-Schulz iteration with the fixed coefficients {@code (3.4445, -4.775,
     * 2.0315)}, which push singular values quickly into about {@code [0.7, 1.2]} but never
     * converge to 1.
     *
     * @param steps iterations, 5 in Muon
     * @return the orthogonaliser
     */
    static Orthogonalizer newtonSchulz(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive");
        }
        return g -> Scion.orthogonalize(g, steps);
    }

    /**
     * Polar Express with the paper's settings (lower bound {@code 1e-3}, safety factor 1.01).
     *
     * @param steps iterations
     * @return the orthogonaliser
     */
    static Orthogonalizer polarExpress(int steps) {
        return PolarExpress.of(steps);
    }

    /**
     * How far a matrix is from having orthonormal rows (or columns, if tall):
     * {@code ||X X^T - I||_F / sqrt(k)} on the shorter side {@code k}, the root mean
     * square deviation of the squared singular values from 1.
     *
     * @param x a matrix
     * @return the error, 0 for an exact polar factor
     */
    static float error(Tensor x) {
        boolean tall = x.shape().dim(0) > x.shape().dim(1);
        Tensor w = tall ? x.transpose() : x;
        int k = w.shape().dim(0);
        Tensor gram = w.matmul(w.transpose());
        Tensor eye = Tensor.arange(k).reshape(k, 1).equalTo(Tensor.arange(k).reshape(1, k));
        return (float) (gram.sub(eye).square().sum().sqrt().item() / Math.sqrt(k));
    }

    /**
     * Normalises a matrix so its spectral norm is at most {@code 1 / safety}, by its
     * Frobenius norm.
     *
     * @param x a matrix
     * @param safety a factor at least 1
     * @return the scaled matrix
     */
    static Tensor normalize(Tensor x, float safety) {
        Tensor norm = x.square().sum().sqrt().mul(safety);
        return x.div(norm.maximum(Tensor.full(Shape.of(), 1e-7f)));
    }
}
