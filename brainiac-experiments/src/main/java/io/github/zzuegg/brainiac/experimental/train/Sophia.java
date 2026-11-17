package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Sophia (Liu et al. 2023, <a href="https://arxiv.org/abs/2305.14342">arXiv:2305.14342</a>):
 * a second-order-ish optimizer that divides the gradient momentum by a cheap running
 * estimate of the Hessian diagonal {@code h} and clips every coordinate of the result to
 * {@code [-1, 1]}: {@code w -= lr * clip(m / max(gamma h, eps), 1)}. Clipping keeps it safe
 * where the curvature estimate is wrong or negative.
 *
 * <p>Refresh the Hessian estimate every few steps (the paper: every 10) in one of two ways:
 * <ul>
 *   <li>{@link #updateHessianGnb(List, int)} (Sophia-G): after a backward pass of the loss
 *       on labels <em>sampled from the model's own softmax</em> (see
 *       {@link #sampleLabels}), the Gauss-Newton-Bartlett estimate is
 *       {@code batch * g_hat^2}.</li>
 *   <li>{@link #updateHessianHutchinson(List, Supplier, long)} (Sophia-H): Hutchinson's
 *       {@code u * (H u)} with a random sign vector {@code u} and the exact Hessian-vector
 *       product of double backpropagation.</li>
 * </ul>
 */
public final class Sophia extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float gamma;
    private final float weightDecay;
    private final Map<Var, Tensor> hessian = new IdentityHashMap<>();

    /**
     * @param learningRate the step size, for example {@code 1e-3f}
     * @param beta1 momentum decay, 0.965
     * @param beta2 Hessian EMA decay, 0.99
     * @param gamma the clipping scale, for example 0.01
     * @param weightDecay the decoupled weight decay
     */
    public Sophia(float learningRate, float beta1, float beta2, float gamma, float weightDecay) {
        super(learningRate, "m");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.gamma = gamma;
        this.weightDecay = weightDecay;
    }

    /**
     * @param learningRate the step size
     * @return Sophia with the paper's defaults ({@code beta1 = 0.965, beta2 = 0.99, gamma = 0.01})
     */
    public static Sophia of(float learningRate) {
        return new Sophia(learningRate, 0.965f, 0.99f, 0.01f, 0);
    }

    /**
     * Class ids sampled from the softmax of logits, for the Gauss-Newton-Bartlett estimate.
     *
     * @param logits {@code [batch, classes]}
     * @param random the random source
     * @return {@code [batch]} sampled ids
     */
    public static Tensor sampleLabels(Tensor logits, SplittableRandom random) {
        float[] p = LearningMath.softmax(logits).toArray();
        int classes = logits.shape().dim(1);
        int batch = logits.shape().dim(0);
        float[] ids = new float[batch];
        for (int i = 0; i < batch; i++) {
            double u = random.nextDouble();
            int c = 0;
            double acc = p[i * classes];
            while (acc < u && c < classes - 1) {
                c++;
                acc += p[i * classes + c];
            }
            ids[i] = c;
        }
        return Tensor.of(Shape.of(batch), ids);
    }

    /**
     * Updates the Hessian estimate from the current gradients, which must come from the loss
     * on sampled labels, and clears them.
     *
     * @param parameters the parameters
     * @param batch the batch size
     */
    public void updateHessianGnb(List<Var> parameters, int batch) {
        for (Var p : parameters) {
            if (p.grad() != null) {
                blend(p, p.grad().square().mul(batch));
                p.zeroGrad();
            }
        }
    }

    /**
     * Updates the Hessian estimate with Hutchinson's estimator {@code u * Hu}, with the exact
     * Hessian-vector product {@code Hu} from core nested autograd
     * ({@link Var#gradients(Var, List, boolean)}: the gradient of {@code grad . u}). Leaves the
     * parameters' gradients untouched.
     *
     * @param parameters the parameters
     * @param loss computes the loss on a batch; it need not run {@code backward()}
     * @param seed the seed of the sign vector
     */
    public void updateHessianHutchinson(List<Var> parameters, Supplier<Var> loss, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> grads = Var.gradients(loss.get(), parameters, true);
        List<Tensor> u = new ArrayList<>(parameters.size());
        Var gu = null;
        for (int i = 0; i < parameters.size(); i++) {
            Var p = parameters.get(i);
            float[] signs = new float[Math.toIntExact(p.shape().size())];
            for (int j = 0; j < signs.length; j++) {
                signs[j] = r.nextBoolean() ? 1 : -1;
            }
            Tensor ui = Tensor.of(p.shape(), signs).to(p.value().device());
            u.add(ui);
            Var term = grads.get(i).mul(Var.of(ui)).sum();
            gu = gu == null ? term : gu.add(term);
        }
        if (gu == null) {
            return;
        }
        List<Var> hu = Var.gradients(gu, parameters, false);
        for (int i = 0; i < parameters.size(); i++) {
            blend(parameters.get(i), u.get(i).mul(hu.get(i).value()));
        }
    }

    @Override
    public void forget(Var parameter) {
        super.forget(parameter);
        hessian.remove(parameter);
    }

    private void blend(Var p, Tensor estimate) {
        Tensor h = hessian.get(p);
        hessian.put(p, h == null ? estimate.mul(1 - beta2) : h.mul(beta2).add(estimate.mul(1 - beta2)));
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor h = hessian.get(p);
        Tensor denom = h == null ? Tensor.full(Shape.of(), 1e-12f)
                : h.mul(gamma).maximum(Tensor.full(Shape.of(), 1e-12f));
        Tensor ratio = m.div(denom).maximum(Tensor.full(Shape.of(), -1f)).minimum(Tensor.full(Shape.of(), 1f));
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(ratio.mul(learningRate())));
        return new Tensor[] {m};
    }

    @Override
    public String toString() {
        return "Sophia(lr=" + learningRate() + ")";
    }
}
