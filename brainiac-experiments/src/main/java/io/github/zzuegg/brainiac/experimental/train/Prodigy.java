package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Map;

/**
 * Prodigy (Mishchenko and Defazio 2023, <a href="https://arxiv.org/abs/2306.06101">arXiv:2306.06101</a>):
 * Adam whose step size {@code d} estimates the distance to the solution online, so the
 * learning rate needs no tuning (leave it at 1). It keeps a weighted sum of gradients
 * {@code s} and of the progress {@code <g, x0 - x>}; their ratio is a lower bound on the
 * distance {@code |x0 - x*|}, and {@code d} grows to the largest bound seen:
 *
 * <pre>
 *   m = b1 m + (1 - b1) d g;   v = b2 v + (1 - b2) d^2 g^2
 *   r = sqrt(b2) r + (1 - sqrt(b2)) lr d^2 &lt;g, x0 - x&gt;;   s = sqrt(b2) s + (1 - sqrt(b2)) lr d^2 g
 *   d = max(d, r / |s|_1);   x -= lr d m / (sqrt(v) + d eps)
 * </pre>
 */
public final class Prodigy extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private double d;
    private double r;
    private double sNorm;

    /**
     * @param learningRate a multiplier on the adapted step, normally 1
     * @param weightDecay the decoupled weight decay
     * @param initialD the starting distance estimate, for example {@code 1e-6}
     */
    public Prodigy(float learningRate, float weightDecay, double initialD) {
        super(learningRate, "m", "v", "s", "x0");
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.weightDecay = weightDecay;
        this.d = initialD;
    }

    /** @return Prodigy with learning rate 1, no weight decay and {@code d0 = 1e-6} */
    public static Prodigy of() {
        return new Prodigy(1, 0, 1e-6);
    }

    /** @return the current adapted step size {@code d} */
    public double distance() {
        return d;
    }

    @Override
    protected void beforeStep(List<Var> withGrad, int t) {
        r *= Math.sqrt(beta2);
        sNorm = 0;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float lr = learningRate();
        float dk = (float) d;
        Tensor x0 = s[3] == null ? p.value() : s[3];
        Tensor m = s[0] == null ? g.mul((1 - beta1) * dk) : s[0].mul(beta1).add(g.mul((1 - beta1) * dk));
        Tensor v = s[1] == null ? g.square().mul((1 - beta2) * dk * dk)
                : s[1].mul(beta2).add(g.square().mul((1 - beta2) * dk * dk));
        float sb = (float) Math.sqrt(beta2);
        float coef = (1 - sb) * lr * dk * dk;
        r += coef * g.mul(x0.sub(p.value())).sum().item();
        Tensor sum = s[2] == null ? g.mul(coef) : s[2].mul(sb).add(g.mul(coef));
        sNorm += sum.abs().sum().item();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * dk * weightDecay) : p.value();
        p.assign(w.sub(m.div(v.sqrt().add(dk * 1e-8f)).mul(lr * dk)));
        return new Tensor[] {m, v, sum, x0};
    }

    @Override
    protected void afterStep(List<Var> withGrad, int t) {
        if (sNorm > 0) {
            d = Math.max(d, r / sNorm);
        }
    }

    @Override
    protected Map<String, Tensor> extraState() {
        return Map.of("d", Tensor.of(Shape.of(1), (float) d), "r", Tensor.of(Shape.of(1), (float) r));
    }

    @Override
    protected void loadExtraState(Map<String, Tensor> saved) {
        if (saved.containsKey("d")) {
            d = saved.get("d").item();
            r = saved.get("r").item();
        }
    }

    @Override
    public String toString() {
        return "Prodigy(d=" + d + ")";
    }
}
