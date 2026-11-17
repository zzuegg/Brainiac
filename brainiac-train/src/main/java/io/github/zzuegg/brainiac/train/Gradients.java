package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Operations on the gradients of a set of parameters, for training loops written by hand.
 * Nothing is read back to the host: on a GPU the step does not wait.
 */
public final class Gradients {

    private Gradients() {
    }

    /**
     * The global L2 norm of the gradients, over every parameter that has one.
     *
     * @param parameters the parameters
     * @return a float32 scalar, or {@code null} if no parameter has a gradient
     */
    public static Tensor norm(List<Var> parameters) {
        Tensor total = null;
        for (Var p : parameters) {
            if (p.grad() != null) {
                Tensor sq = p.grad().to(DType.FLOAT32).square().sum();
                total = total == null ? sq : total.add(sq);
            }
        }
        return total == null ? null : total.sqrt();
    }

    /**
     * Scales all gradients together so their global L2 norm is at most {@code maxNorm}
     * (Pascanu et al., 2013), as {@code torch.nn.utils.clip_grad_norm_}.
     *
     * @param parameters the parameters whose gradients are replaced
     * @param maxNorm the largest allowed norm, positive
     * @return the norm before clipping, a float32 scalar, or {@code null} if no parameter has a gradient
     */
    public static Tensor clipNorm(List<Var> parameters, float maxNorm) {
        if (!(maxNorm > 0)) {
            throw new IllegalArgumentException("maxNorm must be positive, got " + maxNorm);
        }
        Tensor norm = norm(parameters);
        if (norm == null) {
            return null;
        }
        Tensor limit = Tensor.full(Shape.of(), maxNorm, norm.dtype(), norm.device());
        Tensor scale = limit.div(norm.maximum(limit));   // exactly 1 below the limit
        Map<DType, Tensor> scales = new EnumMap<>(DType.class);
        scales.put(scale.dtype(), scale);
        for (Var p : parameters) {
            if (p.grad() != null) {
                Tensor g = p.grad();
                p.setGrad(g.mul(scales.computeIfAbsent(g.dtype(), scale::to)));
            }
        }
        return norm;
    }

    /**
     * Clamps every gradient value to {@code [-limit, limit]}.
     *
     * @param parameters the parameters whose gradients are replaced
     * @param limit the largest allowed magnitude, positive
     */
    public static void clipValue(List<Var> parameters, float limit) {
        if (!(limit > 0)) {
            throw new IllegalArgumentException("limit must be positive, got " + limit);
        }
        for (Var p : parameters) {
            if (p.grad() != null) {
                Tensor g = p.grad();
                Tensor hi = Tensor.full(Shape.of(), limit, g.dtype(), g.device());
                p.setGrad(g.minimum(hi).maximum(hi.neg()));
            }
        }
    }
}
