package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.BitFlip;
import io.github.zzuegg.brainiac.experimental.rl.ContrastiveGoalRl;
import io.github.zzuegg.brainiac.experimental.rl.HindsightReplay;
import io.github.zzuegg.brainiac.experimental.rl.ResidualRlNetworks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Goal reaching without rewards: contrastive RL ({@link ContrastiveGoalRl}) with deep
 * residual encoders of several depths, against hindsight experience replay on DQN
 * ({@link HindsightReplay}, as in {@link GoalsImitationAndTeams#hindsight}), on bit flipping
 * ({@link BitFlip}): success rate after the same number of environment steps.
 */
@SuppressWarnings("exports")
public final class ContrastiveGoals {
    private ContrastiveGoals() {
    }

    /**
     * The share of greedy episodes that reach the goal.
     *
     * @param agent the policy
     * @param env the task
     * @param episodes how many
     * @param seed the seed
     * @return the success rate
     */
    public static double successRate(Agent agent, BitFlip env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
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

    /**
     * Runs the comparison.
     *
     * @param bits the bit-flipping size
     * @param steps environment steps per learner
     * @param depths the encoder depths for contrastive RL (multiples of 4)
     * @param seed the seed
     * @return success rate per learner
     */
    public static Map<String, Double> run(int bits, int steps, int[] depths, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        BitFlip env = new BitFlip(bits);
        HindsightReplay her = HindsightReplay.dqn(() -> Nets.mlp(2 * bits, bits, Activation.RELU, seed, 64), bits)
                .relabelCount(4).seed(seed);
        her.fit(env, steps);
        out.put("HER + DQN", successRate(her, env, 100, seed));
        for (int depth : depths) {
            ContrastiveGoalRl crl = ContrastiveGoalRl.of(bits, bits, bits, ResidualRlNetworks.deepResidual(32, depth),
                    16, seed);
            crl.fit(new BitFlip(bits), steps);
            out.put("contrastive RL, depth " + depth, successRate(crl, env, 100, seed));
        }
        return out;
    }
}
