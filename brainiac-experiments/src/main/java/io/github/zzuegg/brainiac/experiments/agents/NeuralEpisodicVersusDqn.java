package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.NeuralEpisodicControl;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;

/**
 * Neural Episodic Control against DQN on a small budget of cart-pole steps: the episodic
 * memory turns each return into a usable estimate at once, where DQN's network needs many
 * gradient steps to absorb it.
 */
public final class NeuralEpisodicVersusDqn {

    /**
     * @param nec the greedy return of Neural Episodic Control
     * @param dqn the greedy return of DQN after the same steps
     * @param entries the entries in NEC's dictionaries
     */
    public record Result(double nec, double dqn, int entries) {
    }

    private NeuralEpisodicVersusDqn() {
    }

    /**
     * @param steps environment steps for each learner
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(long steps, long seed) {
        Environment env = new CartPole(200);
        NeuralEpisodicControl nec = NeuralEpisodicControl.of(4, 2, 8, 32, 2000, seed).epsilon(0.1).neighbours(16);
        nec.fit(env, steps);
        double necReturn = Rl.evaluate(nec, env, 10, seed);
        Dqn dqn = Dqn.of(() -> Nets.mlp(4, 2, Activation.RELU, seed, 64, 64), 2).learningStarts(500)
                .epsilon(Schedule.linear(1, 0.05, steps / 2)).targetUpdateEvery(250).nSteps(3).seed(seed);
        dqn.fit(env, steps);
        return new Result(necReturn, Rl.evaluate(dqn, env, 10, seed), nec.size());
    }
}
