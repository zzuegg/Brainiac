package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Muon (Jordan et al. 2024) with a pluggable {@link Orthogonalizer} and an optional
 * cautious weight decay: the same update as the core {@code Optimizer.muon} (Nesterov
 * momentum, orthogonalised, scaled by {@code sqrt(max(1, rows / cols))}, decoupled weight
 * decay), so with {@code Orthogonalizer.newtonSchulz(5)} it matches the core exactly and
 * with {@link PolarExpress} it becomes Muon with Polar Express
 * (<a href="https://arxiv.org/abs/2505.16932">arXiv:2505.16932</a>).
 *
 * <p>With {@code cautiousDecay} the decay only acts on entries whose update moves the weight
 * toward zero (the sign of the update agrees with the sign of the weight), after "Cautious
 * Weight Decay" (Chen et al. 2025, <a href="https://arxiv.org/abs/2510.12402">arXiv:2510.12402</a>);
 * see {@link CautiousWeightDecay}. Matrices only; give the rest to AdamW with
 * {@code Optimizer.groups}.
 */
public final class OrthogonalMuon extends SlotOptimizer {

    private final float weightDecay;
    private final float momentum;
    private final Orthogonalizer orthogonalizer;
    private final boolean cautiousDecay;

    /**
     * @param learningRate the step size, for example 0.02
     * @param weightDecay the decoupled weight decay
     * @param momentum the momentum, 0.95
     * @param orthogonalizer how the momentum is orthogonalised
     * @param cautiousDecay whether to decay only where the update shrinks the weight
     */
    public OrthogonalMuon(float learningRate, float weightDecay, float momentum, Orthogonalizer orthogonalizer,
            boolean cautiousDecay) {
        super(learningRate, "momentum");
        if (weightDecay < 0 || momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("bad Muon settings");
        }
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.orthogonalizer = orthogonalizer;
        this.cautiousDecay = cautiousDecay;
    }

    /**
     * Muon with Polar Express and no weight decay.
     *
     * @param learningRate the step size
     * @param steps Polar Express iterations, for example 5
     * @return the optimizer
     */
    public static OrthogonalMuon polarExpress(float learningRate, int steps) {
        return new OrthogonalMuon(learningRate, 0, 0.95f, PolarExpress.of(steps), false);
    }

    /**
     * Nesterov momentum as the core Muon uses it.
     *
     * @param g the gradient
     * @param buf the momentum buffer, or {@code null}
     * @param momentum the momentum
     * @return {@code {new buffer, Nesterov direction}}
     */
    static Tensor[] nesterov(Tensor g, Tensor buf, float momentum) {
        Tensor b = buf == null ? g.mul(1 - momentum) : buf.mul(momentum).add(g.mul(1 - momentum));
        return new Tensor[] {b, g.mul(1 - momentum).add(b.mul(momentum))};
    }

    /**
     * Muon's shape factor {@code sqrt(max(1, rows / cols))}.
     *
     * @param g a matrix
     * @return the factor
     */
    static float shapeScale(Tensor g) {
        return (float) Math.sqrt(Math.max(1.0, g.shape().dim(0) / (double) g.shape().dim(1)));
    }

    /**
     * Applies an update {@code w <- w - lr * update - lr * decay * w} with plain or
     * cautious decoupled decay.
     *
     * @param p the parameter
     * @param update the update direction (subtracted)
     * @param lr the learning rate
     * @param weightDecay the decay
     * @param cautious whether to decay only where {@code update * w >= 0}
     */
    static void apply(Var p, Tensor update, float lr, float weightDecay, boolean cautious) {
        Tensor w = p.value();
        Tensor next = w.sub(update.mul(lr));
        if (weightDecay > 0) {
            Tensor decay = w.mul(lr * weightDecay);
            if (cautious) {
                decay = decay.mul(update.mul(w).greater(Tensor.full(Shape.of(), -1e-30f)));
            }
            next = next.sub(decay);
        }
        p.assign(next);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        if (g.rank() != 2) {
            throw new IllegalArgumentException("Muon updates matrices only, got " + g.shape());
        }
        Tensor[] m = nesterov(g, s[0], momentum);
        Tensor u = orthogonalizer.orthogonalize(m[1]).mul(shapeScale(g));
        apply(p, u, learningRate(), weightDecay, cautiousDecay);
        return new Tensor[] {m[0]};
    }

    @Override
    public String toString() {
        return "OrthogonalMuon(lr=" + learningRate() + ", " + orthogonalizer + (cautiousDecay ? ", cautious" : "")
                + ")";
    }
}
