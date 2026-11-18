package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Td3;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Pendulum;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModernOffPolicyEvolutionaryTest {

    @Test
    void evolutionaryAtEqualBudget() {
        Map<String, double[]> r = ModernOffPolicy.evolutionary(ModernOffPolicy.Task.PENDULUM, 2_000, 1, 32);
        System.out.printf("12.6.17 pendulum greedy return/steps used at 2k steps: %s%n",
                TabularExplorationTest.table(r));
        assertTrue(r.values().stream().allMatch(v -> v[0] < 0 && v[1] <= 2_400), "every row stays within the budget");
    }

    @Test
    void stepwiseDriver() {
        Td3 td3 = Td3.td3(3, 1, RlNetworkFactory.mlp(Activation.RELU, 16, 16)).learningStarts(50).seed(3);
        Environment env = new Pendulum(200);
        td3.begin(env);
        float[] obs = env.reset(new SplittableRandom(1));
        int updates = 0;
        for (int t = 0; t < 120; t++) {
            float[] u = td3.actUnit(obs);
            assertTrue(Math.abs(u[0]) <= 1, "unit actions");
            Environment.Step s = env.step(td3.toEnvironment(u));
            td3.observe(obs, u, s);
            obs = s.done() ? env.reset(new SplittableRandom(t)) : s.observation();
            updates += td3.updateOnce() ? 1 : 0;
        }
        assertEquals(120, td3.stepsObserved());
        assertTrue(updates > 0 && updates <= 71, "updates start after the warm-up");
    }
}
