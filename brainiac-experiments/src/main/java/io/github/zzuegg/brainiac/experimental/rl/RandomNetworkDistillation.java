package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.Objects;

/**
 * Random Network Distillation (Burda et al., <a href="https://arxiv.org/abs/1810.12894">1810.12894</a>):
 * curiosity as the error of predicting a fixed, randomly initialized network. A predictor
 * network is trained to match the random target's output on every observation the agent
 * sees; on familiar observations it matches well, on novel ones it does not, so the
 * prediction error is a novelty bonus. Unlike forward-model curiosity it is not fooled by
 * stochastic transitions (the target is deterministic). The bonus is divided by a running
 * standard deviation of the errors, as in the paper.
 */
public final class RandomNetworkDistillation implements IntrinsicReward {

    private final Model target;
    private final Model predictor;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private final AgentKit.Running errors = new AgentKit.Running();
    private boolean normalize = true;

    private RandomNetworkDistillation(Model target, Model predictor) {
        this.target = Objects.requireNonNull(target);
        this.predictor = Objects.requireNonNull(predictor);
    }

    /**
     * @param observations observation size
     * @param features output size of the target network
     * @param hidden hidden width
     * @param seed the seed (target and predictor get different initializations)
     * @return the bonus
     */
    public static RandomNetworkDistillation of(int observations, int features, int hidden, long seed) {
        return new RandomNetworkDistillation(AgentKit.mlp(observations, features, seed, hidden),
                AgentKit.mlp(observations, features, seed + 7919, hidden, hidden));
    }

    /**
     * @param o the predictor's optimizer
     * @return this bonus
     */
    public RandomNetworkDistillation optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param on divide the error by its running standard deviation
     * @return this bonus
     */
    public RandomNetworkDistillation normalize(boolean on) {
        this.normalize = on;
        return this;
    }

    /**
     * @param observation an observation
     * @return the current prediction error, without learning
     */
    public float error(float[] observation) {
        float[] t = AgentKit.predict(target, observation);
        float[] p = AgentKit.predict(predictor, observation);
        double e = 0;
        for (int i = 0; i < t.length; i++) {
            e += (t[i] - p[i]) * (t[i] - p[i]);
        }
        return (float) (e / t.length);
    }

    @Override
    public float observe(float[] observation, float[] action, float[] next) {
        Var x = Var.of(AgentKit.row(next));
        Var loss = predictor.forward(x, true).sub(target.forward(x, false).value()).square().mean();
        float e = loss.value().item();
        loss.backward();
        optimizer.step(predictor.parameters());
        errors.add(e);
        return normalize ? (float) (e / errors.std()) : e;
    }
}
