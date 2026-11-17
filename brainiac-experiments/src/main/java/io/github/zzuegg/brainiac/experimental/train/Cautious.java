package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Cautious optimizers (Liang et al. 2024, <a href="https://arxiv.org/abs/2411.16085">arXiv:2411.16085</a>):
 * a one-line change to any momentum optimizer. Each update component whose sign disagrees
 * with the current gradient (it would, to first order, <em>raise</em> the loss on this
 * batch) is masked out, and the rest is rescaled by {@code n / (mask.sum() + 1)} so that the
 * average step size stays the same. C-AdamW and C-Lion train faster than AdamW and Lion at
 * the same settings and keep their convergence guarantees.
 *
 * <p>This wrapper works with any {@link Optimizer}: it remembers the weights and gradient,
 * lets {@code inner} step, and masks the resulting update {@code u = w_new - w_old} where
 * {@code u * g > 0}. The inner optimizer's state (momentum, second moments) is left
 * untouched, as in the paper. Give the inner optimizer no weight decay and pass it here
 * instead: the decay is then applied unmasked, as C-AdamW does.
 */
public final class Cautious implements Optimizer {

    private final Optimizer inner;
    private final float weightDecay;

    /**
     * @param inner the optimizer whose updates are masked, without weight decay
     * @param weightDecay the decoupled weight decay applied after the masked update
     */
    public Cautious(Optimizer inner, float weightDecay) {
        if (weightDecay < 0) {
            throw new IllegalArgumentException("weightDecay must not be negative");
        }
        this.inner = inner;
        this.weightDecay = weightDecay;
    }

    /**
     * @param inner the optimizer whose updates are masked
     * @return the cautious variant, no extra weight decay
     */
    public static Cautious wrap(Optimizer inner) {
        return new Cautious(inner, 0);
    }

    /**
     * C-AdamW.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return cautious AdamW
     */
    public static Cautious adamW(float learningRate, float weightDecay) {
        return new Cautious(Optimizer.adam(learningRate), weightDecay);
    }

    /**
     * C-Lion.
     *
     * @param learningRate the step size
     * @param weightDecay the decoupled weight decay
     * @return cautious Lion
     */
    public static Cautious lion(float learningRate, float weightDecay) {
        return new Cautious(Optimizer.lion(learningRate, 0), weightDecay);
    }

    /**
     * The mask itself: 1 where the update agrees with the descent direction, rescaled to
     * keep the mean step size.
     *
     * @param update the proposed change of the weights
     * @param grad the gradient
     * @return the scaled 0/1 mask, same shape
     */
    public static Tensor mask(Tensor update, Tensor grad) {
        Tensor keep = update.mul(grad).neg().greater(Tensor.full(Shape.of(), 0f));
        float n = update.size();
        return keep.div(keep.mean().add(1f / n));
    }

    @Override
    public void step(List<Var> parameters) {
        List<Var> live = new ArrayList<>();
        List<Tensor> before = new ArrayList<>();
        List<Tensor> grads = new ArrayList<>();
        for (Var p : parameters) {
            if (p.grad() != null) {
                live.add(p);
                before.add(p.value());
                grads.add(p.grad().add(0f));   // a copy: the inner step may release the gradient
            }
        }
        Tensor.evaluate(grads);
        inner.step(parameters);
        List<Tensor> touched = new ArrayList<>();
        float decay = 1 - inner.learningRate() * weightDecay;
        for (int i = 0; i < live.size(); i++) {
            Var p = live.get(i);
            Tensor w0 = before.get(i);
            Tensor u = p.value().sub(w0);
            Tensor base = weightDecay > 0 ? w0.mul(decay) : w0;
            p.assign(base.add(u.mul(mask(u, grads.get(i)))));
            touched.add(p.value());
        }
        Tensor.evaluate(touched);
    }

    @Override
    public void resetState(Var parameter, Tensor keep) {
        inner.resetState(parameter, keep);
    }

    @Override
    public void forget(Var parameter) {
        inner.forget(parameter);
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        return inner.state(parameters);
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        inner.loadState(parameters, state);
    }

    @Override
    public String toString() {
        return "Cautious(" + inner + (weightDecay > 0 ? ", weightDecay=" + weightDecay : "") + ")";
    }
}
