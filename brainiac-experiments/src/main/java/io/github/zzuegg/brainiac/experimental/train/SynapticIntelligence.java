package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Synaptic Intelligence (Zenke, Poole and Ganguli 2017, <a href="https://arxiv.org/abs/1703.04200">arXiv:1703.04200</a>):
 * like EWC, but the importance of every weight is measured <em>online</em> along the
 * training path: each step adds {@code -g * dw}, the weight's share of the loss decrease,
 * to its running contribution {@code omega}. At the end of a task the importance becomes
 * {@code Omega += omega / (total change^2 + xi)} and the penalty
 * {@code c sum Omega (w - w*)^2} protects it.
 *
 * <p>{@link #track(Optimizer)} returns an optimizer that records the path while it steps;
 * use it in place of the original one, and {@link #wrap(Loss)} adds the penalty for a core
 * {@code Trainer}.
 */
public final class SynapticIntelligence {

    private final List<Var> parameters;
    private final float strength;
    private final float xi;
    private final Map<Var, Tensor> omega = new IdentityHashMap<>();
    private final Map<Var, Tensor> importance = new IdentityHashMap<>();
    private final Map<Var, Tensor> start = new IdentityHashMap<>();

    /**
     * @param parameters the parameters to protect
     * @param strength the penalty strength {@code c}, for example 1
     * @param xi the damping, for example {@code 0.1}
     */
    public SynapticIntelligence(List<Var> parameters, float strength, float xi) {
        this.parameters = List.copyOf(parameters);
        this.strength = strength;
        this.xi = xi;
        for (Var p : parameters) {
            start.put(p, p.value());
        }
    }

    /**
     * @param inner the optimizer that trains the parameters
     * @return an optimizer that steps with {@code inner} and records each weight's
     *     contribution to the loss decrease
     */
    public Optimizer track(Optimizer inner) {
        return new Optimizer() {
            @Override
            public void step(List<Var> ps) {
                Map<Var, Tensor[]> before = new IdentityHashMap<>();
                for (Var p : parameters) {
                    if (p.grad() != null) {
                        before.put(p, new Tensor[] {p.value(), p.grad()});
                    }
                }
                inner.step(ps);
                before.forEach((p, wg) -> omega.merge(p, wg[1].mul(p.value().sub(wg[0])).neg(), Tensor::add));
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
            public Map<String, Tensor> state(List<Var> ps) {
                return inner.state(ps);
            }

            @Override
            public void loadState(List<Var> ps, Map<String, Tensor> state) {
                inner.loadState(ps, state);
            }
        };
    }

    /** Ends a task: turns the path contributions into importances and re-anchors. */
    public void consolidate() {
        for (Var p : parameters) {
            Tensor w = omega.remove(p);
            if (w != null) {
                Tensor delta = p.value().sub(start.get(p));
                Tensor add = w.div(delta.square().add(xi)).maximum(Tensor.scalar(0));
                importance.merge(p, add, Tensor::add);
            }
            start.put(p, p.value());
        }
    }

    /**
     * @param p a parameter
     * @return its accumulated importance, or {@code null} before consolidation
     */
    public Tensor importance(Var p) {
        return importance.get(p);
    }

    /** @return {@code c sum Omega (w - w*)^2} */
    public Var penalty() {
        Var total = null;
        for (Var p : parameters) {
            Tensor o = importance.get(p);
            if (o != null) {
                Var term = p.sub(start.get(p)).square().mul(o).sum();
                total = total == null ? term : total.add(term);
            }
        }
        return total == null ? Var.of(Tensor.scalar(0)) : total.mul(strength);
    }

    /**
     * @param base the task loss
     * @return the loss plus the SI penalty
     */
    public Loss wrap(Loss base) {
        return (prediction, target) -> base.compute(prediction, target).add(penalty());
    }
}
