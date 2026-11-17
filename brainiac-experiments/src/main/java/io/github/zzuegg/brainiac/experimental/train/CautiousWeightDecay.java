package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Cautious weight decay (Chen et al. 2025, <a href="https://arxiv.org/abs/2510.12402">arXiv:2510.12402</a>):
 * decoupled weight decay applied only to the coordinates where the optimizer's update and
 * the parameter have the same sign, {@code w <- w - lr (u + wd * 1[u * w >= 0] * w)}, so
 * decay only helps an update that already moves the weight toward zero and never fights
 * one that moves it away. Plain decoupled decay optimises a regularised objective; the
 * cautious version keeps the original objective's stationary points (it is a sliding-mode
 * method that prefers small-norm solutions among them), so it reaches lower loss with
 * a similar weight norm.
 *
 * <p>It wraps any optimizer given <em>without</em> its own weight decay: after the inner
 * step it reads the update {@code u} from the change of the weights and decays the
 * agreeing entries with the current learning rate. {@link #adamW} gives cautious-decay
 * AdamW; wrap {@link Cautious} for cautious masking plus cautious decay, or use
 * {@link OrthogonalMuon}'s option for Muon. Vectors (biases, norms) can be excluded with
 * {@code Optimizer.groups} as usual.
 */
public final class CautiousWeightDecay implements Optimizer {

    private final Optimizer inner;
    private final float weightDecay;
    private long decayed;
    private long seen;

    /**
     * @param inner the optimizer, without weight decay of its own
     * @param weightDecay the decoupled weight decay
     */
    public CautiousWeightDecay(Optimizer inner, float weightDecay) {
        if (weightDecay < 0) {
            throw new IllegalArgumentException("weightDecay must not be negative");
        }
        this.inner = inner;
        this.weightDecay = weightDecay;
    }

    /**
     * AdamW with cautious weight decay.
     *
     * @param learningRate the step size
     * @param weightDecay the weight decay
     * @return the optimizer
     */
    public static CautiousWeightDecay adamW(float learningRate, float weightDecay) {
        return new CautiousWeightDecay(Optimizer.adam(learningRate), weightDecay);
    }

    /**
     * The decay mask: 1 where the update {@code u} (subtracted from the weights) and the
     * weight agree in sign.
     *
     * @param update the update direction {@code u}
     * @param weight the weights before the step
     * @return the 0/1 mask, same shape
     */
    public static Tensor mask(Tensor update, Tensor weight) {
        return update.mul(weight).greater(Tensor.full(Shape.of(), -1e-30f));
    }

    @Override
    public void step(List<Var> parameters) {
        List<Var> live = new ArrayList<>();
        List<Tensor> before = new ArrayList<>();
        for (Var p : parameters) {
            if (p.grad() != null) {
                live.add(p);
                before.add(p.value());
            }
        }
        inner.step(parameters);
        if (weightDecay == 0) {
            return;
        }
        float lr = inner.learningRate();
        List<Tensor> touched = new ArrayList<>();
        for (int i = 0; i < live.size(); i++) {
            Var p = live.get(i);
            Tensor w0 = before.get(i);
            Tensor mask = mask(w0.sub(p.value()), w0);   // u = (w0 - w1) / lr, same sign
            decayed += (long) mask.sum().item();
            seen += mask.size();
            p.assign(p.value().sub(w0.mul(mask).mul(lr * weightDecay)));
            touched.add(p.value());
        }
        Tensor.evaluate(touched);
    }

    /** @return the share of weight entries decayed so far, over all steps */
    public double decayedShare() {
        return seen == 0 ? 0 : (double) decayed / seen;
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
        return "CautiousWeightDecay(" + inner + ", weightDecay=" + weightDecay + ")";
    }
}
