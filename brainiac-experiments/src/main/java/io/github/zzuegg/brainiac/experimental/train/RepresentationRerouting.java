package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;

/**
 * Circuit breakers by representation rerouting (RR): instead of teaching a model to refuse,
 * train the internal representations of harmful processes away from where they were, so a
 * harmful generation that has started anyway (after a jailbreak or a prefilled answer)
 * runs into representations that no longer lead anywhere useful and breaks down. A frozen
 * copy of the original model gives the reference activations {@code h0} at one layer;
 * each step minimises
 *
 * <pre>c_rr(t) mean relu(cos(h(x_harm), h0(x_harm))) + c_ret(t) mean ||h(x_ok) - h0(x_ok)||^2</pre>
 *
 * <p>so harmful rows become orthogonal to (or opposite of) their original direction while
 * benign rows stay put; as in the paper, the rerouting weight decays from {@code alpha} to
 * {@code alpha / 2} and the retain weight grows from 0 to {@code alpha / 2} over training.
 * Harmful rows should be prompts <em>together with harmful completions</em>, so the breaker
 * trips inside the harmful generation itself.
 *
 * <p>Zou et al. 2024, "Improving Alignment and Robustness with Circuit Breakers",
 * <a href="https://arxiv.org/abs/2406.04313">arXiv:2406.04313</a>.
 */
public final class RepresentationRerouting {

    private final Model model;
    private final Model frozen;
    private final int layer;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private float alpha = 4;
    private long totalSteps = 100;
    private long steps;

    private RepresentationRerouting(Model model, int layer) {
        this.model = Objects.requireNonNull(model);
        this.frozen = model.copy(model.device());
        this.layer = layer;
    }

    /**
     * Sets up rerouting; the model's current weights become the frozen reference.
     *
     * @param model the model, changed in place
     * @param layer the step (layer of a sequential model) whose output is rerouted
     * @return the trainer
     */
    public static RepresentationRerouting of(Model model, int layer) {
        return new RepresentationRerouting(model, layer);
    }

    /**
     * @param o the optimizer of the model's parameters
     * @return this
     */
    public RepresentationRerouting optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param a the overall loss weight
     * @return this
     */
    public RepresentationRerouting alpha(float a) {
        this.alpha = a;
        return this;
    }

    /**
     * @param n the planned number of steps (for the weight schedule)
     * @return this
     */
    public RepresentationRerouting totalSteps(long n) {
        this.totalSteps = Math.max(1, n);
        return this;
    }

    private Var hidden(Tensor x) {
        Var[] out = new Var[1];
        model.forward(Var.of(x), true, (step, value) -> {
            if (step.index() == layer) {
                out[0] = value;
            }
            return value;
        });
        return out[0];
    }

    /**
     * Mean cosine similarity between rows of two activation matrices.
     *
     * @param a {@code [rows, width]}
     * @param b reference {@code [rows, width]}
     * @return the per-row cosine, {@code [rows]}
     */
    static Var cosine(Var a, Tensor b) {
        Var dot = a.mul(b).sum(1, false);
        Var na = a.square().sum(1, false).add(1e-8f).sqrt();
        Tensor nb = b.square().sum(1, false).add(1e-8f).sqrt();
        return dot.div(na.mul(nb));
    }

    /**
     * One training step.
     *
     * @param harmful inputs of harmful processes (prompt plus harmful completion prefixes)
     * @param benign inputs whose representations must stay
     * @return the rerouting term's mean cosine before the step
     */
    public float step(Tensor harmful, Tensor benign) {
        double t = Math.min(1, steps / (double) totalSteps);
        float cRr = (float) (alpha * (1 - t / 2));
        float cRet = (float) (alpha * t / 2);
        steps++;
        Tensor refHarm = LayerHooks.trace(frozen, Var.of(harmful)).get(layer).value();
        Tensor refOk = LayerHooks.trace(frozen, Var.of(benign)).get(layer).value();
        Var cos = cosine(hidden(harmful), refHarm);
        Var loss = cos.relu().mean().mul(cRr);
        if (cRet > 0) {
            loss = loss.add(hidden(benign).sub(refOk).square().sum(1, false).mean().mul(cRet));
        }
        loss.backward();
        optimizer.step(model.parameters());
        return cos.mean().value().item();
    }

    /**
     * The mean cosine between the current and the original representations of some inputs.
     *
     * @param x inputs
     * @return the mean cosine; near 1 means unchanged
     */
    public double similarity(Tensor x) {
        Tensor h = LayerHooks.trace(model, Var.of(x)).get(layer).value();
        Tensor h0 = LayerHooks.trace(frozen, Var.of(x)).get(layer).value();
        return cosine(Var.of(h), h0).mean().value().item();
    }

    /** @return the frozen reference model's parameters (never updated) */
    public List<Var> referenceParameters() {
        return frozen.parameters();
    }
}
