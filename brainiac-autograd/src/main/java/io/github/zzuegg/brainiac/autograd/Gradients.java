package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;

/**
 * Helpers on the gradients of a set of parameters, after {@link Var#backward()} and before an
 * optimizer step.
 */
public final class Gradients {

    private Gradients() {
    }

    /**
     * The L2 norm of all the gradients together, as one vector; parameters without a gradient
     * are left out. It stays on the device: nothing is read back.
     *
     * @param params the parameters
     * @return a scalar tensor, or null if none has a gradient
     */
    public static Tensor globalNorm(List<Var> params) {
        Tensor sum = null;
        for (Var p : params) {
            if (p.grad() != null) {
                Tensor s = p.grad().to(DType.FLOAT32).square().sum();
                sum = sum == null ? s : sum.add(s);
            }
        }
        return sum == null ? null : sum.sqrt();
    }

    /**
     * Scales all the gradients by {@code min(1, maxNorm / norm)}, so their global norm is at
     * most {@code maxNorm}, as {@code torch.nn.utils.clip_grad_norm_}. Nothing is read back
     * from the device.
     *
     * @param params the parameters
     * @param maxNorm the largest global norm, positive
     * @return the norm before clipping, as a scalar tensor, or null if none has a gradient
     */
    public static Tensor clipByGlobalNorm(List<Var> params, float maxNorm) {
        if (!(maxNorm > 0)) {
            throw new IllegalArgumentException("maxNorm must be positive, got " + maxNorm);
        }
        Tensor norm = globalNorm(params);
        if (norm == null) {
            return null;
        }
        Tensor scale = norm.add(1e-6f).pow(-1).mul(maxNorm);
        scale = scale.minimum(Tensor.full(Shape.of(), 1, scale.dtype(), scale.device()));
        for (Var p : params) {
            if (p.grad() != null) {
                p.setGrad(p.grad().mul(scale.to(p.grad().dtype())));
            }
        }
        return norm;
    }

    /**
     * Limits each gradient value to {@code [-limit, limit]}, as {@code clip_grad_value_}.
     *
     * @param params the parameters
     * @param limit the largest magnitude, positive
     */
    public static void clipByValue(List<Var> params, float limit) {
        if (!(limit > 0)) {
            throw new IllegalArgumentException("limit must be positive, got " + limit);
        }
        for (Var p : params) {
            if (p.grad() != null) {
                p.setGrad(p.grad().clamp(-limit, limit));
            }
        }
    }
}
