package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Rl;
import org.junit.jupiter.api.Test;

class ImitationTest {

    /** A 4x4 grid with the goal in the bottom-right corner. */
    static TabularMdp world() {
        return TabularMdp.grid(4, 4, 3, 3, 0.1, 16);
    }

    static OfflineDataset demos() {
        TabularMdp mdp = world();
        Agent expert = TabularMdp.greedy(mdp.optimalQ(0.95, 50));
        return OfflineDataset.collect(mdp, expert, 30, 1);
    }

    @Test
    void maxEntIrlRecoversAGoalSeekingReward() {
        TabularMdp mdp = world();
        MaxEntIrl irl = MaxEntIrl.of(mdp).learningRate(0.2);
        double gap = irl.fit(demos(), 150);
        double[] r = irl.reward();
        int best = RlTensors.argmax(r);
        Agent recovered = TabularMdp.greedy(mdp.withReward(r).optimalQ(0.95, 50));
        double score = Rl.evaluate(recovered, mdp, 20, 5);
        double random = Rl.evaluate(Agent.random(mdp.actionSpace(), 1), mdp, 20, 5);
        System.out.printf("MaxEnt IRL: gap %.3f, best state %d, recovered policy %.3f, random %.3f%n", gap, best,
                score, random);
        assertTrue(best == 15, "the goal should get the highest learned reward, got state " + best);
        assertTrue(score > 0.85 && score > random + 0.5);
    }

    @Test
    void gailImitatesTheExpertWithoutSeeingRewards() {
        TabularMdp mdp = world();
        double random = Rl.evaluate(Agent.random(mdp.actionSpace(), 1), mdp, 20, 5);
        Gail gail = Gail.of(demos(), 16, 4, RlNetworkFactory.mlp(Activation.TANH, 32)).seed(2);
        gail.fit(mdp, 20000);
        double score = Rl.evaluate(gail, mdp, 20, 5);
        System.out.printf("GAIL: %.3f vs random %.3f%n", score, random);
        assertTrue(score > 0.8 && score > random + 0.5, "GAIL " + score);
    }
}
