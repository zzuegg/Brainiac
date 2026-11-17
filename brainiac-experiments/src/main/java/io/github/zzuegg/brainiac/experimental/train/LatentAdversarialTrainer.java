package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;

/**
 * Targeted latent adversarial training (LAT): fine-tuning in which an adversary perturbs a
 * hidden layer's activations, not the inputs, to bring out an unwanted behaviour, and the
 * model is trained to behave well anyway. Unwanted behaviours that no input found during
 * training can trigger (backdoors, jailbreak-only capabilities) still live in some region of
 * activation space; an attacker in the latent space can reach those regions directly, so
 * training against it removes behaviours that input-space adversarial training and plain
 * fine-tuning leave in place.
 *
 * <p>Each {@link #step}: the adversary starts from zero and takes {@code attackSteps}
 * normalised gradient steps on a per-row perturbation {@code delta} added to layer
 * {@code layer}'s output, minimising the loss towards the <em>unwanted</em> targets and
 * projected back onto the L2 ball of radius {@code epsilon}; then the model takes one
 * optimizer step on the loss towards the <em>wanted</em> targets under that perturbation,
 * plus the ordinary loss on an unperturbed retain batch.
 *
 * <p>The model must output logits; targets are class ids (next tokens for a language
 * model). Casper et al. 2024, "Defending Against Unforeseen Failure Modes with Latent
 * Adversarial Training", <a href="https://arxiv.org/abs/2403.05030">arXiv:2403.05030</a>;
 * Sheshadri et al. 2024, "Latent Adversarial Training Improves Robustness to Persistent
 * Harmful Behaviors in LLMs", <a href="https://arxiv.org/abs/2407.15549">arXiv:2407.15549</a>.
 */
public final class LatentAdversarialTrainer {

    private final Model model;
    private final int layer;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private float epsilon = 1;
    private int attackSteps = 8;
    private float attackRate = 0.3f;
    private float retainWeight = 1;

    private LatentAdversarialTrainer(Model model, int layer) {
        this.model = Objects.requireNonNull(model);
        this.layer = layer;
    }

    /**
     * Sets up latent adversarial training.
     *
     * @param model the model, changed in place; it outputs logits
     * @param layer the step (layer of a sequential model) whose output is attacked
     * @return the trainer
     */
    public static LatentAdversarialTrainer of(Model model, int layer) {
        return new LatentAdversarialTrainer(model, layer);
    }

    /**
     * @param o the optimizer of the model's parameters
     * @return this
     */
    public LatentAdversarialTrainer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param e the L2 radius of the perturbation per row
     * @return this
     */
    public LatentAdversarialTrainer epsilon(float e) {
        this.epsilon = e;
        return this;
    }

    /**
     * @param n the adversary's steps per training step
     * @return this
     */
    public LatentAdversarialTrainer attackSteps(int n) {
        this.attackSteps = n;
        return this;
    }

    /**
     * @param rate the adversary's step length as a share of {@code epsilon}
     * @return this
     */
    public LatentAdversarialTrainer attackRate(float rate) {
        this.attackRate = rate;
        return this;
    }

    /**
     * @param w the weight of the retain loss
     * @return this
     */
    public LatentAdversarialTrainer retainWeight(float w) {
        this.retainWeight = w;
        return this;
    }

    private Model.Hook add(Var delta) {
        return (step, out) -> step.index() == layer ? out.add(delta) : out;
    }

    /**
     * Finds the worst perturbation for a batch: the one, within the radius, that makes the
     * unwanted targets most likely.
     *
     * @param x the inputs
     * @param unwanted the unwanted target ids
     * @return the perturbation {@code [rows, width]}
     */
    public Tensor attack(Tensor x, Tensor unwanted) {
        int rows = x.shape().dim(0);
        Tensor clean = LayerHooks.trace(model, Var.of(x)).get(layer).value();
        Var delta = Var.param(Tensor.zeros(clean.shape()));
        for (int s = 0; s < attackSteps; s++) {
            Var loss = Functional.crossEntropy(model.forward(Var.of(x), false, add(delta)), unwanted);
            Tensor g = Var.gradients(loss, List.of(delta), false).get(0).value();
            float[] gv = g.to(DType.FLOAT32).toArray();
            float[] d = delta.value().to(DType.FLOAT32).toArray();
            int width = gv.length / rows;
            for (int r = 0; r < rows; r++) {
                double n = 0;
                for (int j = 0; j < width; j++) {
                    n += gv[r * width + j] * gv[r * width + j];
                }
                n = Math.sqrt(n) + 1e-12;
                double dn = 0;
                for (int j = 0; j < width; j++) {
                    d[r * width + j] -= (float) (attackRate * epsilon * gv[r * width + j] / n);
                    dn += d[r * width + j] * d[r * width + j];
                }
                dn = Math.sqrt(dn);
                if (dn > epsilon) {
                    for (int j = 0; j < width; j++) {
                        d[r * width + j] *= (float) (epsilon / dn);
                    }
                }
            }
            delta.assign(Tensor.of(delta.shape(), d));
        }
        return delta.value();
    }

    /**
     * One training step.
     *
     * @param x inputs the adversary attacks
     * @param wanted the behaviour to keep under attack (target ids)
     * @param unwanted the behaviour the adversary tries to bring out (target ids)
     * @param retainX inputs whose ordinary behaviour must be kept, or null
     * @param retainTargets their target ids, or null
     * @return the loss towards the wanted targets under attack
     */
    public float step(Tensor x, Tensor wanted, Tensor unwanted, Tensor retainX, Tensor retainTargets) {
        Var delta = Var.of(attack(x, unwanted));
        Var adv = Functional.crossEntropy(model.forward(Var.of(x), true, add(delta)), wanted);
        Var loss = adv;
        if (retainX != null) {
            loss = loss.add(Functional.crossEntropy(model.forward(Var.of(retainX), true), retainTargets)
                    .mul(retainWeight));
        }
        loss.backward();
        optimizer.step(model.parameters());
        return adv.value().item();
    }
}
