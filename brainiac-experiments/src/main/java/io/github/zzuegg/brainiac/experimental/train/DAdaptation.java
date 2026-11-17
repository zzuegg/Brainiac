package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Map;

/**
 * D-Adaptation for SGD (Defazio and Mishchenko 2023,
 * <a href="https://arxiv.org/abs/2301.07733">arXiv:2301.07733</a>): dual-averaging SGD whose
 * step size {@code d * lr / |g_0|} is set from a running lower bound {@code d} on the
 * distance to the solution, {@code d_hat = 2 sum_k lambda_k <g_k, s_k> / |s|}, where
 * {@code s} is the weighted sum of past gradients. No learning-rate tuning: leave
 * {@code lr = 1}.
 */
public final class DAdaptation extends SlotOptimizer {

    private final double growth;
    private double d;
    private double numerator;
    private double g0;
    private double sSquared;
    private double stepScale;

    /**
     * @param learningRate a multiplier on the adapted step, normally 1
     * @param initialD the starting estimate, for example {@code 1e-6}
     * @param growth the largest factor {@code d} may grow by per step ({@code
     *     Double.POSITIVE_INFINITY} for no limit)
     */
    public DAdaptation(float learningRate, double initialD, double growth) {
        super(learningRate, "s", "x0");
        this.d = initialD;
        this.growth = growth;
    }

    /** @return D-Adapted SGD with lr 1, {@code d0 = 1e-6} and unlimited growth */
    public static DAdaptation sgd() {
        return new DAdaptation(1, 1e-6, Double.POSITIVE_INFINITY);
    }

    /** @return the current distance estimate */
    public double distance() {
        return d;
    }

    @Override
    protected void beforeStep(List<Var> withGrad, int t) {
        if (g0 == 0) {
            double sq = 0;
            for (Var p : withGrad) {
                sq += p.grad().square().sum().item();
            }
            g0 = Math.sqrt(sq);
        }
        stepScale = g0 > 0 ? d * learningRate() / g0 : 0;
        sSquared = 0;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor x0 = s[1] == null ? p.value() : s[1];
        Tensor sum = s[0] == null ? Tensor.zeros(g.shape(), g.dtype(), g.device()) : s[0];
        numerator += stepScale * g.mul(sum).sum().item();
        sum = sum.add(g.mul((float) stepScale));
        sSquared += sum.square().sum().item();
        p.assign(x0.sub(sum));
        return new Tensor[] {sum, x0};
    }

    @Override
    protected void afterStep(List<Var> withGrad, int t) {
        if (sSquared > 0) {
            double dHat = 2 * numerator / Math.sqrt(sSquared);
            d = Math.max(d, Math.min(dHat, d * growth));
        }
    }

    @Override
    protected Map<String, Tensor> extraState() {
        return Map.of("d", Tensor.of(Shape.of(1), (float) d), "numerator", Tensor.of(Shape.of(1), (float) numerator),
                "g0", Tensor.of(Shape.of(1), (float) g0));
    }

    @Override
    protected void loadExtraState(Map<String, Tensor> saved) {
        if (saved.containsKey("d")) {
            d = saved.get("d").item();
            numerator = saved.get("numerator").item();
            g0 = saved.get("g0").item();
        }
    }

    @Override
    public String toString() {
        return "DAdaptSGD(d=" + d + ")";
    }
}
