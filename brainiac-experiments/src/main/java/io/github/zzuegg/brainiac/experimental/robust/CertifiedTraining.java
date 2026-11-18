package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;

/**
 * Certified training of a ReLU network on bounds from {@link BoundPropagation}: the loss is
 * {@code κ CE(z(x)) + (1 - κ) CE(z_worst)}, where the worst-case logits are the negated
 * margin lower bounds over the {@code ℓ∞} ball. The radius ramps from 0 to the target
 * {@code ε} after a warm-up of clean steps, and {@code κ} from 1 to 0.5, the schedule that makes
 * IBP training stable (Gowal et al. 2018, <a href="https://arxiv.org/abs/1810.12715">arXiv:1810.12715</a>).
 * CROWN-IBP mixes the margins as {@code β CROWN-IBP + (1 - β) IBP} with {@code β} falling from 1
 * to 0 over the ramp (Zhang et al., ICLR 2020, <a href="https://arxiv.org/abs/1906.06316">arXiv:1906.06316</a>).
 */
public final class CertifiedTraining {

    /** The bound trained on. */
    public enum Method {
        /** interval bounds */
        IBP,
        /** CROWN-IBP mixed with IBP */
        CROWN_IBP
    }

    private final List<BoundPropagation.Affine> layers;
    private final List<Var> parameters;
    private final Optimizer optimizer;
    private final Method method;
    private float epsilon = 0.1f;
    private int warmup = 50;
    private int ramp = 200;
    private int step;

    private CertifiedTraining(List<BoundPropagation.Affine> layers, List<Var> parameters, Optimizer optimizer,
            Method method) {
        this.layers = List.copyOf(layers);
        this.parameters = List.copyOf(parameters);
        this.optimizer = Objects.requireNonNull(optimizer);
        this.method = Objects.requireNonNull(method);
    }

    /**
     * Certified training of a network.
     *
     * @param layers the network's affine layers
     * @param parameters its parameters
     * @param optimizer the optimizer
     * @param method the bound
     * @return the settings
     */
    public static CertifiedTraining of(List<BoundPropagation.Affine> layers, List<Var> parameters,
            Optimizer optimizer, Method method) {
        return new CertifiedTraining(layers, parameters, optimizer, method);
    }

    /**
     * @param e the target {@code ℓ∞} radius
     * @param warmupSteps clean steps first
     * @param rampSteps steps over which {@code ε}, {@code κ} and {@code β} move to their targets
     * @return this
     */
    public CertifiedTraining schedule(float e, int warmupSteps, int rampSteps) {
        this.epsilon = e;
        this.warmup = warmupSteps;
        this.ramp = rampSteps;
        return this;
    }

    /** @return the radius of the current step */
    public float currentEpsilon() {
        return epsilon * progress();
    }

    private float progress() {
        return step < warmup ? 0 : Math.min(1f, (step - warmup) / (float) Math.max(1, ramp));
    }

    /**
     * One training step.
     *
     * @param x the inputs {@code [n, in]}
     * @param labels the classes
     * @return the loss
     */
    public float step(Tensor x, int[] labels) {
        float t = progress();
        float kappa = 1 - 0.5f * t;
        float e = epsilon * t;
        float[] ids = new float[labels.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = labels[i];
        }
        Tensor y = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(ids.length), ids);
        Var in = Var.of(x);
        Var loss = Functional.crossEntropy(BoundPropagation.forward(layers, in), y).mul(kappa);
        if (e > 0) {
            Var margins = BoundPropagation.ibpMargins(layers, in, e, labels);
            if (method == Method.CROWN_IBP) {
                float beta = 1 - t;
                if (beta > 0) {
                    margins = BoundPropagation.crownIbp(layers, in, e, labels).mul(beta).add(margins.mul(1 - beta));
                }
            }
            loss = loss.add(Functional.crossEntropy(BoundPropagation.worstCaseLogits(margins), y).mul(1 - kappa));
        }
        float value = loss.value().item();
        for (Var p : parameters) {
            p.zeroGrad();
        }
        loss.backward();
        optimizer.step(parameters);
        step++;
        return value;
    }
}
