package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The Intrinsic Curiosity Module (Pathak et al., <a href="https://arxiv.org/abs/1705.05363">1705.05363</a>):
 * curiosity as the error of a forward model that predicts the next state's features from
 * the current features and the action. The features come from an encoder trained by an
 * inverse model (predict the action from two consecutive feature vectors), so they keep
 * only what the agent can affect and ignore uncontrollable noise. The bonus is
 * {@code eta/2 |f(phi(s), a) - phi(s')|^2}, divided by its running standard deviation.
 */
public final class IntrinsicCuriosityModule implements IntrinsicReward {

    private final int actions;
    private final Model encoder;
    private final Model inverse;
    private final Model forward;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private float beta = 0.2f;
    private final AgentKit.Running errors = new AgentKit.Running();

    private IntrinsicCuriosityModule(int observations, int actions, int features, int hidden, long seed) {
        this.actions = actions;
        this.encoder = Model.sequential().input(observations).dense(hidden, Activation.TANH)
                .dense(features, Activation.TANH).seed(seed).build();
        this.inverse = AgentKit.mlp(2 * features, actions, seed + 1, hidden);
        this.forward = AgentKit.mlp(features + actions, features, seed + 2, hidden);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param features feature size
     * @param hidden hidden width
     * @param seed the seed
     * @return the module
     */
    public static IntrinsicCuriosityModule of(int observations, int actions, int features, int hidden, long seed) {
        return new IntrinsicCuriosityModule(observations, actions, features, hidden, seed);
    }

    /**
     * @param o the optimizer of all three networks
     * @return this module
     */
    public IntrinsicCuriosityModule optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param weight the weight of the forward loss against the inverse loss (the paper's
     *     {@code beta})
     * @return this module
     */
    public IntrinsicCuriosityModule forwardWeight(float weight) {
        this.beta = weight;
        return this;
    }

    @Override
    public float observe(float[] observation, float[] action, float[] next) {
        int a = Math.round(action[0]);
        Var phi = encoder.forward(Var.of(AgentKit.row(observation)), true);
        Var phiNext = encoder.forward(Var.of(AgentKit.row(next)), true);
        Var logits = inverse.forward(Var.concat(1, phi, phiNext), true);
        Var inverseLoss = Functional.crossEntropy(logits, Tensor.of(Shape.of(1), (float) a));
        Var predicted = forward.forward(Var.concat(1, phi.detach(), Var.of(AgentKit.row(AgentKit.oneHot(a, actions)))),
                true);
        Var forwardLoss = predicted.sub(phiNext.detach()).square().mean();
        float error = forwardLoss.value().item();
        Var loss = inverseLoss.mul(1 - beta).add(forwardLoss.mul(beta));
        loss.backward();
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.addAll(inverse.parameters());
        params.addAll(forward.parameters());
        optimizer.step(params);
        errors.add(error);
        return (float) (error / errors.std());
    }
}
