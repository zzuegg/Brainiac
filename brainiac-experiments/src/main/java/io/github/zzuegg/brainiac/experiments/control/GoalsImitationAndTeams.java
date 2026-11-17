package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.BitFlip;
import io.github.zzuegg.brainiac.experimental.rl.Gail;
import io.github.zzuegg.brainiac.experimental.rl.HindsightReplay;
import io.github.zzuegg.brainiac.experimental.rl.MaxEntIrl;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.TabularMdp;
import io.github.zzuegg.brainiac.experimental.rl.TwoStepGame;
import io.github.zzuegg.brainiac.experimental.rl.ValueDecomposition;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Three small studies of learning signals beyond a dense reward.
 * <ul>
 *   <li>{@link #hindsight}: success rate on bit flipping of growing size with and without
 *       hindsight experience replay (Andrychowicz et al.,
 *       <a href="https://arxiv.org/abs/1707.01495">arXiv:1707.01495</a>);</li>
 *   <li>{@link #imitation}: behaviour cloning, GAIL and maximum-entropy IRL from a handful of
 *       expert demonstrations on a slippery grid, where cloning suffers from states the
 *       expert never visited (Ho and Ermon, <a href="https://arxiv.org/abs/1606.03476">arXiv:1606.03476</a>;
 *       Ziebart et al., AAAI 2008);</li>
 *   <li>{@link #teams}: independent learners, VDN and QMIX on the two-step coordination
 *       game (Rashid et al., <a href="https://arxiv.org/abs/1803.11485">arXiv:1803.11485</a>).</li>
 * </ul>
 */
public final class GoalsImitationAndTeams {

    private GoalsImitationAndTeams() {
    }

    private static double successRate(Agent agent, BitFlip env, int episodes, long seed) {
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
     * @param sizes bit counts to try
     * @param steps training steps per run
     * @param seed the seed
     * @return "n bits, HER" / "n bits, plain" to greedy success rate
     */
    public static Map<String, Double> hindsight(int[] sizes, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (int n : sizes) {
            for (boolean her : new boolean[] {true, false}) {
                BitFlip env = new BitFlip(n);
                HindsightReplay learner = HindsightReplay.dqn(() -> Nets.mlp(2 * n, n, Activation.RELU, seed, 64), n)
                        .relabelCount(her ? 4 : 0).seed(seed);
                learner.fit(env, steps);
                out.put(n + " bits, " + (her ? "HER" : "plain"), successRate(learner, env, 100, seed));
            }
        }
        return out;
    }

    /**
     * @param demonstrations expert episodes given to every method
     * @param gailSteps environment steps of GAIL training
     * @param seed the seed
     * @return method to mean evaluation return; "expert" and "random" for reference
     */
    public static Map<String, Double> imitation(int demonstrations, int gailSteps, long seed) {
        TabularMdp mdp = TabularMdp.grid(5, 5, 4, 4, 0.2, 20);
        Agent expert = TabularMdp.greedy(mdp.optimalQ(0.95, 60));
        OfflineDataset demos = OfflineDataset.collect(mdp, expert, demonstrations, seed);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("expert", Rl.evaluate(expert, mdp, 40, seed));
        out.put("random", Rl.evaluate(Agent.random(mdp.actionSpace(), seed), mdp, 40, seed));
        out.put("BC", Rl.evaluate(BehaviorCloning.discrete(Nets.mlp(25, 4, Activation.RELU, seed, 32), 4)
                .seed(seed).train(demos, 300), mdp, 40, seed));
        Gail gail = Gail.of(demos, 25, 4, RlNetworkFactory.mlp(Activation.TANH, 32)).seed(seed);
        gail.fit(mdp, gailSteps);
        out.put("GAIL", Rl.evaluate(gail, mdp, 40, seed));
        MaxEntIrl irl = MaxEntIrl.of(mdp).learningRate(0.2);
        irl.fit(demos, 150);
        out.put("MaxEnt IRL", Rl.evaluate(TabularMdp.greedy(mdp.withReward(irl.reward()).optimalQ(0.95, 60)), mdp, 40,
                seed));
        return out;
    }

    /**
     * @param steps joint steps of training each
     * @param seed the seed
     * @return mixer to greedy team return (optimum 8)
     */
    public static Map<String, Double> teams(int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (ValueDecomposition.Mixer m : ValueDecomposition.Mixer.values()) {
            TwoStepGame game = new TwoStepGame();
            ValueDecomposition learner = ValueDecomposition.of(m, game, RlNetworkFactory.mlp(Activation.RELU, 32))
                    .epsilon(Schedule.constant(1)).seed(seed);
            learner.fit(game, steps);
            out.put(m.toString(), learner.evaluate(game, seed));
        }
        return out;
    }
}
