package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Map;

/**
 * Schedule-Free SGD and AdamW (Defazio et al. 2024,
 * <a href="https://arxiv.org/abs/2405.15682">arXiv:2405.15682</a>): no learning-rate decay
 * schedule and no need to know the training length, yet as good as a tuned cosine
 * schedule. Three sequences are kept: the base iterate {@code z} (plain SGD or Adam steps),
 * the running average {@code x} of all {@code z} (weighted by the squared warm-up rate),
 * and the point {@code y = (1 - beta) z + beta x} where gradients are taken:
 *
 * <pre>
 *   z = z - lr * step(grad(y));   x = (1 - c) x + c z,  c = lr_t^2 / sum lr_i^2;   y = (1 - beta) z + beta x
 * </pre>
 *
 * <p>The parameters hold {@code y} while training. Evaluate the model at {@code x}: call
 * {@link #evalMode} before evaluating or saving and {@link #trainMode} before training
 * again.
 */
public final class ScheduleFree extends SlotOptimizer {

    private final boolean adam;
    private final float beta;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final int warmup;
    private double weightSum;
    private boolean eval;

    private ScheduleFree(boolean adam, float learningRate, float beta, float beta2, float weightDecay, int warmup) {
        super(learningRate, "z", "x", "v");
        this.adam = adam;
        this.beta = beta;
        this.beta2 = beta2;
        this.epsilon = 1e-8f;
        this.weightDecay = weightDecay;
        this.warmup = warmup;
    }

    /**
     * Schedule-free SGD.
     *
     * @param learningRate the constant step size
     * @param beta the interpolation momentum, 0.9
     * @param warmup linear warm-up steps, 0 for none
     * @return the optimizer
     */
    public static ScheduleFree sgd(float learningRate, float beta, int warmup) {
        return new ScheduleFree(false, learningRate, beta, 0, 0, warmup);
    }

    /**
     * Schedule-free AdamW.
     *
     * @param learningRate the constant step size, for example {@code 2.5e-3f}
     * @param weightDecay the weight decay, applied at {@code y}
     * @param warmup linear warm-up steps, 0 for none
     * @return the optimizer ({@code beta1 = 0.9, beta2 = 0.999})
     */
    public static ScheduleFree adamW(float learningRate, float weightDecay, int warmup) {
        return new ScheduleFree(true, learningRate, 0.9f, 0.999f, weightDecay, warmup);
    }

    private float rate(int t) {
        return warmup > 0 ? learningRate() * Math.min(1f, (float) t / warmup) : learningRate();
    }

    @Override
    protected void beforeStep(List<Var> withGrad, int t) {
        if (eval) {
            throw new IllegalStateException("call trainMode(parameters) before training");
        }
        float lr = rate(t);
        weightSum += (double) lr * lr;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor y = p.value();
        Tensor z = s[0] == null ? y : s[0];
        Tensor x = s[1] == null ? y : s[1];
        Tensor v = null;
        float lr = rate(t);
        Tensor step = g;
        if (adam) {
            v = s[2] == null ? g.square().mul(1 - beta2) : s[2].mul(beta2).add(g.square().mul(1 - beta2));
            step = g.div(v.div(1 - (float) Math.pow(beta2, t)).sqrt().add(epsilon));
        }
        if (weightDecay > 0) {
            step = step.add(y.mul(weightDecay));
        }
        z = z.sub(step.mul(lr));
        float c = (float) (lr * lr / weightSum);
        x = x.mul(1 - c).add(z.mul(c));
        p.assign(z.mul(1 - beta).add(x.mul(beta)));
        return new Tensor[] {z, x, v};
    }

    /**
     * Puts the averaged weights {@code x} into the parameters, for evaluation.
     *
     * @param parameters the parameters this optimizer trains
     */
    public void evalMode(List<Var> parameters) {
        if (eval) {
            return;
        }
        for (Var p : parameters) {
            Tensor[] s = slots(p);
            if (s != null) {
                p.assign(s[1]);
            }
        }
        eval = true;
    }

    /**
     * Puts the gradient points {@code y} back into the parameters.
     *
     * @param parameters the parameters
     */
    public void trainMode(List<Var> parameters) {
        if (!eval) {
            return;
        }
        for (Var p : parameters) {
            Tensor[] s = slots(p);
            if (s != null) {
                p.assign(s[0].mul(1 - beta).add(s[1].mul(beta)));
            }
        }
        eval = false;
    }

    @Override
    protected Map<String, Tensor> extraState() {
        return Map.of("weightSum", Tensor.of(Shape.of(1), (float) weightSum));
    }

    @Override
    protected void loadExtraState(Map<String, Tensor> saved) {
        Tensor w = saved.get("weightSum");
        weightSum = w == null ? 0 : w.item();
    }

    @Override
    public String toString() {
        return (adam ? "ScheduleFreeAdamW" : "ScheduleFreeSGD") + "(lr=" + learningRate() + ")";
    }
}
