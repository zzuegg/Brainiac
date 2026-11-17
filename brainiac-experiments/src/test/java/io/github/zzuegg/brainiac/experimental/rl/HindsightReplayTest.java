package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Transition;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

class HindsightReplayTest {

    static double successRate(Agent agent, BitFlip env, int episodes) {
        SplittableRandom r = new SplittableRandom(99);
        int wins = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(r.split());
            Environment.Step s;
            do {
                s = env.step(agent.act(obs, false));
                obs = s.observation();
            } while (!s.done());
            wins += s.terminated() ? 1 : 0;
        }
        return wins / (double) episodes;
    }

    @Test
    void relabelledTransitionsReachTheirGoal() {
        BitFlip env = new BitFlip(3);
        float[] o = env.reset(new SplittableRandom(1));
        Environment.Step s = env.step(0);
        Transition t = new Transition(o, new float[] {0}, s.reward(), s.observation(), s.terminated(), 0.9f);
        List<Transition> hindsight = HindsightReplay.relabel(env, List.of(t), 2, new SplittableRandom(1));
        assertEquals(2, hindsight.size());
        assertEquals(0f, hindsight.getFirst().reward());
        assertTrue(hindsight.getFirst().terminated());
    }

    @Test
    void herSolvesBitFlippingWherePlainDqnFails() {
        int n = 7;
        BitFlip env = new BitFlip(n);
        HindsightReplay her = HindsightReplay.dqn(() -> Nets.mlp(2 * n, n, Activation.RELU, 1, 64), n).seed(1);
        HindsightReplay plain = HindsightReplay.dqn(() -> Nets.mlp(2 * n, n, Activation.RELU, 1, 64), n)
                .relabelCount(0).seed(1);
        her.fit(env, 4000);
        plain.fit(env, 4000);
        double withHer = successRate(her, env, 100);
        double without = successRate(plain, env, 100);
        System.out.println("bit flip " + n + ": HER " + withHer + " plain " + without);
        assertTrue(withHer > 0.6 && withHer > without + 0.3, withHer + " vs " + without);
    }
}
