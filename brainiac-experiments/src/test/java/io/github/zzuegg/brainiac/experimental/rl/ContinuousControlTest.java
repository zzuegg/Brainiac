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
}
