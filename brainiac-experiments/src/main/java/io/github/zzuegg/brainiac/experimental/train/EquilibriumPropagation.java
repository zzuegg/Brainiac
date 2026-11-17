package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Equilibrium Propagation (Scellier and Bengio 2017,
 * <a href="https://arxiv.org/abs/1602.05179">arXiv:1602.05179</a>) on a layered Hopfield-like
 * network with symmetric weights and hard-sigmoid units {@code rho(s) = clamp(s, 0, 1)}.
 *
 * <ol>
 *   <li><b>Free phase</b>: with the input clamped, the state settles to a minimum of the
 *       energy {@code E = 1/2 sum s^2 - sum rho(s_l) W_l rho(s_{l+1}) - sum b rho(s)}, here
 *       by iterating {@code s_l = rho(s_{l-1} W_{l-1} + s_{l+1} W_l^T + b_l)}.</li>
 *   <li><b>Nudged phases</b>: the output is pulled towards the target with strength
 *       {@code beta} (and, symmetrically, away with {@code -beta}, which removes the
 *       estimate's first-order bias, Laborieux et al. 2021) and the state settles again.</li>
 *   <li><b>Learning</b>: {@code dW_l = (rho_l^+ T rho_{l+1}^+ - rho_l^- T rho_{l+1}^-) / (2 beta)},
 *       a contrastive Hebbian rule that needs only the two equilibria at each synapse.</li>
 * </ol>
 *
 * <p>The same physical dynamics compute both the prediction and the gradient, which is why
 * this is the main learning-rule candidate for analog hardware.
 */
public final class EquilibriumPropagation implements LocalLearner {

    private final int[] sizes;
    private final Var[] weights;
    private final Var[] biases;
    private final Optimizer optimizer;
    private final float beta;
    private final int freeSteps;
    private final int nudgedSteps;

    /**
     * @param optimizer the optimizer for the weights
     * @param beta the nudging strength, for example 0.5
     * @param freeSteps iterations of the free phase, for example 20
     * @param nudgedSteps iterations of each nudged phase, for example 8
     * @param seed the seed
     * @param sizes layer sizes, input first, output (classes) last
     */
    public EquilibriumPropagation(Optimizer optimizer, float beta, int freeSteps, int nudgedSteps, long seed,
            int... sizes) {
        this.sizes = sizes.clone();
        this.optimizer = optimizer;
        this.beta = beta;
        this.freeSteps = freeSteps;
        this.nudgedSteps = nudgedSteps;
        SplittableRandom r = new SplittableRandom(seed);
        weights = new Var[sizes.length - 1];
        biases = new Var[sizes.length - 1];
        for (int l = 0; l < weights.length; l++) {
            float a = (float) Math.sqrt(3.0 / sizes[l]);
            weights[l] = Var.param(Tensor.uniform(Shape.of(sizes[l], sizes[l + 1]), -a, a, r));
            biases[l] = Var.param(Tensor.zeros(Shape.of(1, sizes[l + 1])));
        }
    }

    /** @return the weights and biases */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weights));
        p.addAll(List.of(biases));
        return p;
    }

    private static Tensor rho(Tensor s) {
        return s.maximum(Tensor.full(Shape.of(), 0f)).minimum(Tensor.full(Shape.of(), 1f));
    }

    /** Settles the state from {@code s} (index 0 is the clamped input). */
    private Tensor[] settle(Tensor[] s, int steps, Tensor target, float nudge) {
        int n = s.length;
        for (int t = 0; t < steps; t++) {
            Tensor[] next = s.clone();
            for (int l = 1; l < n; l++) {
                Tensor drive = next[l - 1].matmul(weights[l - 1].value()).add(biases[l - 1].value());
                if (l < n - 1) {
                    drive = drive.add(s[l + 1].matmul(weights[l].value().transpose()));
                } else if (nudge != 0) {
                    drive = drive.add(target.sub(s[l]).mul(nudge));
                }
                next[l] = rho(drive);
            }
            Tensor.evaluate(List.of(next));
            s = next;
        }
        return s;
    }

    private Tensor[] start(Tensor inputs) {
        Tensor[] s = new Tensor[sizes.length];
        s[0] = inputs;
        for (int l = 1; l < s.length; l++) {
            s[l] = Tensor.zeros(Shape.of(inputs.shape().dim(0), sizes[l]));
        }
        return s;
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int batch = inputs.shape().dim(0);
        Tensor target = LearningMath.oneHot(labels, sizes[sizes.length - 1]);
        Tensor[] free = settle(start(inputs), freeSteps, null, 0);
        Tensor[] plus = settle(free.clone(), nudgedSteps, target, beta);
        Tensor[] minus = settle(free.clone(), nudgedSteps, target, -beta);
        for (int l = 0; l < weights.length; l++) {
            Tensor dw = plus[l].transpose().matmul(plus[l + 1]).sub(minus[l].transpose().matmul(minus[l + 1]))
                    .div(2 * beta * batch);
            Tensor db = plus[l + 1].sub(minus[l + 1]).sum(0, true).div(2 * beta * batch);
            weights[l].setGrad(dw.neg());
            biases[l].setGrad(db.neg());
        }
        optimizer.step(parameters());
        return free[free.length - 1].sub(target).square().sum().item() / (2 * batch);
    }

    @Override
    public Tensor predict(Tensor inputs) {
        Tensor[] s = settle(start(inputs), freeSteps, null, 0);
        return s[s.length - 1];
    }
}
