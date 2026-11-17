package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.RlResult;
import org.junit.jupiter.api.Test;

class ContinuousControlTest {

    @Test
    void sacLearnsToSwingThePendulum() {
        Pendulum env = new Pendulum(100);
        double random = Rl.evaluate(Agent.random(env.actionSpace(), 3), env, 5, 9);
        Sac sac = Sac.of(3, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32))
                .learningStarts(300).learningRate(3e-3f).seed(1);
        long t = System.nanoTime();
        RlResult r = sac.fit(env, 4000);
        double trained = Rl.evaluate(sac, env, 5, 9);
        System.out.println("SAC random " + random + " trained " + trained + " alpha " + sac.alpha() + " eps "
                + r.episodes().size() + " " + (System.nanoTime() - t) / 1e9 + "s");
        assertTrue(trained > random + 150, trained + " vs " + random);
    }

    @Test
    void td3LearnsToSwingThePendulum() {
        Pendulum env = new Pendulum(100);
        double random = Rl.evaluate(Agent.random(env.actionSpace(), 3), env, 5, 9);
        Td3 td3 = Td3.td3(3, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32)).learningStarts(300)
                .learningRate(3e-3f).seed(1);
        td3.fit(env, 4000);
        double trained = Rl.evaluate(td3, env, 5, 9);
        System.out.println("TD3 random " + random + " trained " + trained);
        assertTrue(trained > random + 150, trained + " vs " + random);
    }

    @Test
    void ddpgLearnsToSwingThePendulum() {
        Pendulum env = new Pendulum(100);
        double random = Rl.evaluate(Agent.random(env.actionSpace(), 3), env, 5, 9);
        Td3 ddpg = Td3.ddpg(3, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32)).learningStarts(300)
                .learningRate(3e-3f).seed(1);
        ddpg.fit(env, 4000);
        double trained = Rl.evaluate(ddpg, env, 5, 9);
        System.out.println("DDPG random " + random + " trained " + trained);
        assertTrue(trained > random + 150, trained + " vs " + random);
    }

    @Test
    void cemPlanningWithALearnedModelSwingsThePendulum() {
        Pendulum env = new Pendulum(100);
        double random = Rl.evaluate(Agent.random(env.actionSpace(), 3), env, 2, 9);
        CemPlanner.Dynamics model = CemPlanner.learn(env,
                io.github.zzuegg.brainiac.rl.Nets.mlp(4, 3, Activation.RELU, 1, 64, 64), 3000, 1500, 1);
        CemPlanner.Reward reward = (o, a) -> {
            double angle = Math.atan2(o[1], o[0]);
            double u = Math.max(-2, Math.min(2, a[0]));
            return -(angle * angle + 0.1 * o[2] * o[2] + 0.001 * u * u);
        };
        CemPlanner mpc = CemPlanner.of(model, reward, (io.github.zzuegg.brainiac.rl.Space.Box) env.actionSpace())
                .horizon(25).search(100, 10, 4).seed(1);
        long t = System.nanoTime();
        double planned = Rl.evaluate(mpc, env, 2, 9);
        System.out.println("CEM-MPC random " + random + " planned " + planned + " " + (System.nanoTime() - t) / 1e9
                + "s");
        assertTrue(planned > random + 300, planned + " vs " + random);
    }
}
