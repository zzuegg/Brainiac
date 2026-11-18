package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.AdversarialImitation;
import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.BitFlip;
import io.github.zzuegg.brainiac.experimental.rl.CommunicatingAgents;
import io.github.zzuegg.brainiac.experimental.rl.CooperativeNavigation;
import io.github.zzuegg.brainiac.experimental.rl.CounterfactualMultiAgent;
import io.github.zzuegg.brainiac.experimental.rl.Gail;
import io.github.zzuegg.brainiac.experimental.rl.HiddenGoalGame;
import io.github.zzuegg.brainiac.experimental.rl.HindsightReplay;
import io.github.zzuegg.brainiac.experimental.rl.Maddpg;
import io.github.zzuegg.brainiac.experimental.rl.MaxEntIrl;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEvaluation;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentPpo;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.TabularMdp;
import io.github.zzuegg.brainiac.experimental.rl.TwoStepGame;
import io.github.zzuegg.brainiac.experimental.rl.ValueDecomposition;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.CartPole;
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
 *   <li>{@link #navigation}: cooperative navigation ({@link CooperativeNavigation}) with
 *       centralised critics, MADDPG and COMA (W7 5.6), IPPO and MAPPO (W7 5.7), against
 *       independent Q-learners, VDN and QMIX, at equal environment steps.</li>
 *   <li>{@link #communication}: learned communication on the hidden-goal game
 *       ({@link HiddenGoalGame}): CommNet and TarMAC against no channel, a discrete
 *       Gumbel channel and a fully observed oracle (W7 5.9).</li>
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

    /**
     * A cooperative-navigation result.
     *
     * @param teamReturn mean greedy team return per episode (higher is better)
     * @param coverage mean fraction of landmarks covered at the end of an episode
     * @param seconds training time
     */
    public record TeamScore(double teamReturn, double coverage, double seconds) {
    }

    /**
     * Trains every multi-agent learner on cooperative navigation at the same number of joint
     * environment steps.
     *
     * @param agents agents (and landmarks)
     * @param steps joint environment steps per learner
     * @param seed the seed
     * @param methods which learners: "random", "IPPO", "MAPPO", "COMA", "MADDPG", "independent
     *     DQN", "VDN", "QMIX"
     * @return method to its score
     */
    public static Map<String, TeamScore> navigation(int agents, long steps, long seed, String... methods) {
        Map<String, TeamScore> out = new LinkedHashMap<>();
        CooperativeNavigation eval = new CooperativeNavigation(agents, 25);
        for (String m : methods) {
            long t0 = System.nanoTime();
            MultiAgentEvaluation.JointPolicy policy;
            switch (m) {
                case "random" -> {
                    SplittableRandom r = new SplittableRandom(seed);
                    policy = (obs, explore) -> {
                        int[] a = new int[obs.length];
                        for (int i = 0; i < a.length; i++) {
                            a[i] = r.nextInt(5);
                        }
                        return a;
                    };
                }
                case "IPPO" -> policy = MultiAgentPpo.ippo(() -> new CooperativeNavigation(agents, 25), 32, seed)
                        .fit(steps)::act;
                case "MAPPO" -> policy = MultiAgentPpo.mappo(() -> new CooperativeNavigation(agents, 25), 32, seed)
                        .fit(steps)::act;
                case "COMA" -> policy = CounterfactualMultiAgent.of(() -> new CooperativeNavigation(agents, 25), 32,
                        seed).fit(steps)::act;
                case "MADDPG" -> {
                    Maddpg maddpg = Maddpg.of(() -> new CooperativeNavigation(agents, 25), 32, seed).fit(steps);
                    double seconds = (System.nanoTime() - t0) / 1e9;
                    double[] r = maddpg.evaluate(20, seed + 100);
                    out.put(m, new TeamScore(r[0], r[1], seconds));
                    continue;
                }
                case "independent DQN", "VDN", "QMIX" -> {
                    ValueDecomposition.Mixer mixer = m.equals("VDN") ? ValueDecomposition.Mixer.VDN
                            : m.equals("QMIX") ? ValueDecomposition.Mixer.QMIX : ValueDecomposition.Mixer.INDEPENDENT;
                    ValueDecomposition vd = ValueDecomposition.of(mixer, eval, RlNetworkFactory.mlp(Activation.RELU, 32))
                            .epsilon(Schedule.linear(1, 0.05, (int) (steps / 2))).seed(seed);
                    vd.fit(new CooperativeNavigation(agents, 25), steps);
                    policy = vd::act;
                }
                default -> throw new IllegalArgumentException(m);
            }
            double seconds = (System.nanoTime() - t0) / 1e9;
            out.put(m, new TeamScore(MultiAgentEvaluation.teamReturn(eval, policy, 20, seed + 100),
                    MultiAgentEvaluation.coverage(eval, policy, 20, seed + 100), seconds));
        }
        return out;
    }

    /**
     * Learned communication on the hidden-goal game.
     *
     * @param agents agents
     * @param goals goals
     * @param iterations gradient steps of 64 episodes
     * @param seed the seed
     * @return channel name to {@code {mean team reward, success rate}}; "TarMAC attention" holds
     *     {@code {attention weight on the informed agent, attention entropy in nats}}
     */
    public static Map<String, double[]> communication(int agents, int goals, int iterations, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        HiddenGoalGame game = new HiddenGoalGame(agents, goals, false);
        String[] names = {"no communication", "CommNet", "TarMAC", "Gumbel channel"};
        CommunicatingAgents.Channel[] channels = CommunicatingAgents.Channel.values();
        for (int i = 0; i < channels.length; i++) {
            CommunicatingAgents c = CommunicatingAgents.of(channels[i], game, 32, 2, seed)
                    .fit(() -> new HiddenGoalGame(agents, goals, false), iterations, 64);
            out.put(names[i], c.evaluate(game, 500, seed + 1));
            if (channels[i] == CommunicatingAgents.Channel.TARMAC) {
                out.put("TarMAC attention", c.attentionOnInformed(game, 200, seed + 2));
            }
        }
        HiddenGoalGame full = new HiddenGoalGame(agents, goals, true);
        out.put("full-observation oracle", CommunicatingAgents.of(CommunicatingAgents.Channel.NONE, full, 32, 2, seed)
                .fit(() -> new HiddenGoalGame(agents, goals, true), iterations, 64).evaluate(full, 500, seed + 1));
        return out;
    }

    /**
     * A cart-pole expert: push towards the side the pole leans to, with a little position and
     * velocity feedback.
     *
     * @return the controller
     */
    public static Agent cartPoleExpert() {
        return (o, explore) -> new float[] {o[2] + 0.5f * o[3] + 0.01f * o[0] + 0.1f * o[1] > 0 ? 1 : 0};
    }

    private static Map<String, Double> imitationRow(Environment env, Agent expert, int obs, int actions,
            OfflineDataset demos, int steps, long seed, TabularMdp mdp) {
        Map<String, Double> out = new LinkedHashMap<>();
        int eval = 20;
        out.put("random", Rl.evaluate(Agent.random(env.actionSpace(), seed), env, eval, seed));
        out.put("BC", Rl.evaluate(BehaviorCloning.discrete(Nets.mlp(obs, actions, Activation.RELU, seed, 32), actions)
                .seed(seed).train(demos, 300), env, eval, seed));
        Gail gail = Gail.of(demos, obs, actions, RlNetworkFactory.mlp(Activation.TANH, 32)).seed(seed);
        gail.fit(env, steps);
        out.put("GAIL (wave 1)", Rl.evaluate(gail, env, eval, seed));
        for (AdversarialImitation.Method m : AdversarialImitation.Method.values()) {
            AdversarialImitation a = AdversarialImitation.of(m, demos, obs, actions,
                    RlNetworkFactory.mlp(Activation.TANH, 32)).seed(seed);
            a.fit(env, steps);
            out.put(a.toString(), Rl.evaluate(a, env, eval, seed));
        }
        AdversarialImitation gp = AdversarialImitation.of(AdversarialImitation.Method.GAIL, demos, obs, actions,
                RlNetworkFactory.mlp(Activation.TANH, 32)).absorbingStates(false).seed(seed);
        gp.fit(env, steps);
        out.put(gp.toString(), Rl.evaluate(gp, env, eval, seed));
        if (mdp != null) {
            MaxEntIrl irl = MaxEntIrl.of(mdp).learningRate(0.2);
            irl.fit(demos, 150);
            out.put("MaxEnt IRL", Rl.evaluate(TabularMdp.greedy(mdp.withReward(irl.reward()).optimalQ(0.95, 60)), mdp,
                    eval, seed));
        }
        return out;
    }

    /**
     * W10 4.13: regularised GAIL (gradient penalty, absorbing states), AIRL and IQ-Learn against
     * wave 1's GAIL, behaviour cloning, maximum-entropy IRL and random on wave 1's slippery
     * grid, with growing numbers of demonstrations. Wave 1's {@link #imitation} rows are
     * unchanged.
     *
     * @param demonstrations demonstration counts to try
     * @param steps environment steps of each online learner
     * @param seed the seed
     * @return "n demos: method" to mean return; "expert" for reference
     */
    public static Map<String, Double> imitationFollowUp(int[] demonstrations, int steps, long seed) {
        TabularMdp mdp = TabularMdp.grid(5, 5, 4, 4, 0.2, 20);
        Agent expert = TabularMdp.greedy(mdp.optimalQ(0.95, 60));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("expert", Rl.evaluate(expert, mdp, 20, seed));
        for (int n : demonstrations) {
            OfflineDataset demos = OfflineDataset.collect(mdp, expert, n, seed);
            imitationRow(mdp, expert, 25, 4, demos, steps, seed, mdp).forEach((k, v) -> out.put(n + " demos: " + k, v));
        }
        return out;
    }

    /**
     * W10 4.13 on cart-pole (200 steps; the expert is {@link #cartPoleExpert}). Maximum-entropy
     * IRL is tabular and is left out.
     *
     * @param demonstrations demonstration counts to try
     * @param steps environment steps of each online learner
     * @param seed the seed
     * @return "n demos: method" to mean return; "expert" for reference
     */
    public static Map<String, Double> imitationFollowUpCartPole(int[] demonstrations, int steps, long seed) {
        CartPole env = new CartPole(200);
        Agent expert = cartPoleExpert();
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("expert", Rl.evaluate(expert, env, 20, seed));
        for (int n : demonstrations) {
            OfflineDataset demos = OfflineDataset.collect(env, expert, n, seed);
            imitationRow(env, expert, 4, 2, demos, steps, seed, null).forEach((k, v) -> out.put(n + " demos: " + k, v));
        }
        return out;
    }

    /** The grid with every move rotated one step clockwise: the same states and goal, other dynamics. */
    static TabularMdp rotated(TabularMdp mdp) {
        int n = mdp.states();
        int a = mdp.actions();
        double[][][] p = new double[n][a][];
        boolean[] terminal = new boolean[n];
        for (int s = 0; s < n; s++) {
            terminal[s] = mdp.terminal(s);
            for (int k = 0; k < a; k++) {
                p[s][k] = mdp.transition(s, (k + 1) % a);
            }
        }
        return TabularMdp.of(p, mdp.reward(), mdp.start(), terminal, mdp.maxSteps());
    }

    /** Value iteration on a state-action reward; after a termination the absorbing reward repeats. */
    static Agent solve(TabularMdp mdp, double[][] reward, double absorbingReward, double gamma) {
        int n = mdp.states();
        int a = mdp.actions();
        double after = absorbingReward / (1 - gamma);
        double[][] qv = new double[n][a];
        for (int it = 0; it < 100; it++) {
            double[] v = new double[n];
            for (int s = 0; s < n; s++) {
                v[s] = mdp.terminal(s) ? after : java.util.Arrays.stream(qv[s]).max().orElse(0);
            }
            for (int s = 0; s < n; s++) {
                for (int k = 0; k < a; k++) {
                    double[] p = mdp.transition(s, k);
                    double next = 0;
                    for (int t = 0; t < n; t++) {
                        next += p[t] * v[t];
                    }
                    qv[s][k] = reward[s][k] + gamma * next;
                }
            }
        }
        return TabularMdp.greedy(qv);
    }

    private static float[] cell(int s, int n) {
        float[] x = new float[n];
        x[s] = 1;
        return x;
    }

    /** The expected recovered reward of each state and action under the training dynamics. */
    static double[][] recovered(AdversarialImitation learner, TabularMdp mdp) {
        int n = mdp.states();
        double[][] r = new double[n][mdp.actions()];
        for (int s = 0; s < n; s++) {
            for (int k = 0; k < mdp.actions(); k++) {
                double[] p = mdp.transition(s, k);
                for (int t = 0; t < n; t++) {
                    if (p[t] > 0) {
                        r[s][k] += p[t] * learner.reward(cell(s, n), k, cell(t, n), mdp.terminal(t));
                    }
                }
            }
        }
        return r;
    }

    /**
     * W10 4.13 transfer: each method's recovered reward from the slippery grid, re-optimised by
     * value iteration on the grid with rotated moves and scored with the true reward there.
     * Copying the expert's actions (the expert itself, or BC) fails under the new dynamics; a
     * reward that describes the goal and not the actions should transfer.
     *
     * @param demonstrations expert episodes
     * @param steps environment steps of each online learner
     * @param seed the seed
     * @return method to mean return on the rotated grid
     */
    public static Map<String, Double> imitationTransfer(int demonstrations, int steps, long seed) {
        TabularMdp mdp = TabularMdp.grid(5, 5, 4, 4, 0.2, 20);
        TabularMdp moved = rotated(mdp);
        Agent expert = TabularMdp.greedy(mdp.optimalQ(0.95, 60));
        OfflineDataset demos = OfflineDataset.collect(mdp, expert, demonstrations, seed);
        Map<String, Double> out = new LinkedHashMap<>();
        int eval = 20;
        out.put("oracle (true reward re-solved)", Rl.evaluate(TabularMdp.greedy(moved.optimalQ(0.95, 60)), moved, eval,
                seed));
        out.put("expert policy copied", Rl.evaluate(expert, moved, eval, seed));
        out.put("random", Rl.evaluate(Agent.random(moved.actionSpace(), seed), moved, eval, seed));
        MaxEntIrl irl = MaxEntIrl.of(mdp).learningRate(0.2);
        irl.fit(demos, 150);
        out.put("MaxEnt IRL reward", Rl.evaluate(TabularMdp.greedy(moved.withReward(irl.reward()).optimalQ(0.95, 60)),
                moved, eval, seed));
        for (boolean absorbing : new boolean[] {true, false}) {
            for (AdversarialImitation.Method m : AdversarialImitation.Method.values()) {
                AdversarialImitation a = AdversarialImitation.of(m, demos, 25, 4,
                        RlNetworkFactory.mlp(Activation.TANH, 32)).absorbingStates(absorbing).seed(seed);
                a.fit(mdp, steps);
                String name = m + (absorbing ? " (absorbing)" : " (no absorbing)");
                double[][] r = recovered(a, mdp);
                out.put(name + " policy itself", Rl.evaluate(a, moved, eval, seed));
                out.put(name + " reward re-solved on the training grid", Rl.evaluate(solve(mdp, r,
                        a.absorbingReward(), 0.95), mdp, eval, seed));
                out.put(name + " reward", Rl.evaluate(solve(moved, r, a.absorbingReward(), 0.95), moved, eval, seed));
            }
        }
        return out;
    }

    /**
     * Runs the W10 4.13 tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println("grid: " + imitationFollowUp(new int[] {1, 5, 20}, 20_000, 1));
        System.out.println("cart-pole: " + imitationFollowUpCartPole(new int[] {1, 5, 20}, 50_000, 1));
        System.out.println("transfer: " + imitationTransfer(5, 20_000, 1));
    }
}
