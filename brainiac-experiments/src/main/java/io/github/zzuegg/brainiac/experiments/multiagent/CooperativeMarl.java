package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.marl.CompleteFactorization;
import io.github.zzuegg.brainiac.experimental.marl.CooperativeMatrixGame;
import io.github.zzuegg.brainiac.experimental.marl.Happo;
import io.github.zzuegg.brainiac.experimental.marl.IsingCoordination;
import io.github.zzuegg.brainiac.experimental.marl.MeanFieldActorCritic;
import io.github.zzuegg.brainiac.experimental.marl.MeanFieldQ;
import io.github.zzuegg.brainiac.experimental.marl.OtherPlay;
import io.github.zzuegg.brainiac.experimental.marl.ScoutCarrier;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentPpo;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.TwoStepGame;
import io.github.zzuegg.brainiac.experimental.rl.ValueDecomposition;
import io.github.zzuegg.brainiac.experiments.agents.ReferentialGame;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Schedule;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Cooperative multi-agent RL beyond VDN and QMIX (wave 11, 11.6.13-11.6.16).
 *
 * <ul>
 *   <li>{@link #factorisation}: greedy return and the share of seeds at the optimal joint
 *       action for QTRAN, QPLEX and QFIX ({@link CompleteFactorization}) against
 *       {@link ValueDecomposition}'s INDEPENDENT, VDN and QMIX and MAPPO on the climbing game,
 *       the penalty game and {@link TwoStepGame};</li>
 *   <li>{@link #meanField}: team return and order parameter on an Ising lattice for
 *       {@link MeanFieldQ} and {@link MeanFieldActorCritic} against independent Q-learners and
 *       MAPPO with parameter sharing (the ground state scores 1 per step);</li>
 *   <li>{@link #otherPlay}: self-play and cross-play between independent seeds for self-play
 *       and {@link OtherPlay} on the lever game and a signalling game with permutable messages,
 *       plus MAPPO seeds on the lever game and {@link ReferentialGame}'s trained pair;</li>
 *   <li>{@link #heterogeneous}: return and its spread over seeds for {@link Happo}'s HAPPO,
 *       HATRPO and simultaneous updates against MAPPO (shared), IPPO and QPLEX on the
 *       scout-and-carrier task and the climbing game.</li>
 * </ul>
 */
public final class CooperativeMarl {

    private CooperativeMarl() {
    }

    /**
     * A score over seeds.
     *
     * @param mean the mean return
     * @param std the standard deviation over seeds
     * @param optimal the share of seeds within 1e-3 of the optimum
     */
    public record Score(double mean, double std, double optimal) {
        @Override
        public String toString() {
            return String.format(Locale.ROOT, "%.2f±%.2f(opt %.2f)", mean, std, optimal);
        }
    }

    private static Score score(double[] returns, double optimum) {
        double mean = 0;
        int hit = 0;
        for (double r : returns) {
            mean += r;
            if (r >= optimum - 1e-3) {
                hit++;
            }
        }
        mean /= returns.length;
        double var = 0;
        for (double r : returns) {
            var += (r - mean) * (r - mean);
        }
        return new Score(mean, Math.sqrt(var / returns.length), hit / (double) returns.length);
    }

    /** The factorisation tasks. */
    public enum Task {
        /** the climbing game, optimum 11 */
        CLIMBING,
        /** the penalty game with k = -100, optimum 10 */
        PENALTY,
        /** {@link TwoStepGame}, optimum 8 */
        TWO_STEP
    }

    private static MultiAgentEnvironment task(Task t) {
        return switch (t) {
            case CLIMBING -> CooperativeMatrixGame.climbing();
            case PENALTY -> CooperativeMatrixGame.penalty(-100);
            case TWO_STEP -> new TwoStepGame();
        };
    }

    private static double optimum(Task t) {
        return switch (t) {
            case CLIMBING -> 11;
            case PENALTY -> 10;
            case TWO_STEP -> 8;
        };
    }

    /**
     * 11.6.13: every factorisation at the same number of joint steps with uniform exploration.
     *
     * @param t the task
     * @param steps joint steps per learner
     * @param seeds seeds
     * @return learner to score
     */
    public static Map<String, Score> factorisation(Task t, int steps, int seeds) {
        Map<String, Score> out = new LinkedHashMap<>();
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, 32);
        double opt = optimum(t);
        for (ValueDecomposition.Mixer m : ValueDecomposition.Mixer.values()) {
            double[] r = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                MultiAgentEnvironment env = task(t);
                ValueDecomposition learner = ValueDecomposition.of(m, env, nets).epsilon(Schedule.constant(1))
                        .seed(s);
                learner.fit(env, steps);
                r[s] = learner.evaluate(env, s);
            }
            out.put(m.toString(), score(r, opt));
        }
        for (CompleteFactorization.Method m : CompleteFactorization.Method.values()) {
            if (m == CompleteFactorization.Method.VDN || m == CompleteFactorization.Method.QMIX) {
                continue;
            }
            double[] r = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                MultiAgentEnvironment env = task(t);
                CompleteFactorization learner = CompleteFactorization.of(m, env, nets, 61 + s)
                        .epsilon(Schedule.constant(1)).seed(s);
                learner.fit(env, steps);
                r[s] = learner.evaluate(env, s);
            }
            out.put(m.toString(), score(r, opt));
        }
        double[] r = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            MultiAgentPpo ppo = MultiAgentPpo.mappo(() -> task(t), 32, s).fit(steps);
            r[s] = ppo.evaluate(1, s);
        }
        out.put("MAPPO", score(r, opt));
        return out;
    }

    /**
     * A lattice score.
     *
     * @param teamReturn mean greedy team return per episode (the ground state scores the
     *     episode length)
     * @param order mean order parameter at the end of the episode (1 at the ground state)
     */
    public record Lattice(double teamReturn, double order) {
        @Override
        public String toString() {
            return String.format(Locale.ROOT, "%.2f/%.2f", teamReturn, order);
        }
    }

    private interface JointActor {
        int[] act(float[][] obs);
    }

    private static Lattice rollout(IsingCoordination env, JointActor actor, Runnable begin, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        double order = 0;
        for (int e = 0; e < episodes; e++) {
            float[][] obs = env.reset(r.split());
            begin.run();
            MultiAgentEnvironment.Step s;
            do {
                s = env.step(actor.act(obs));
                total += s.reward();
                obs = s.observations();
            } while (!s.done());
            order += env.orderParameter();
        }
        return new Lattice(total / episodes, order / episodes);
    }

    /**
     * 11.6.14: mean-field learners on a {@code side x side} Ising lattice.
     *
     * @param side lattice side
     * @param coupling +1 ferromagnetic, -1 antiferromagnetic
     * @param steps joint steps per learner
     * @param seed the seed
     * @return learner to lattice score (episodes of 10 steps)
     */
    public static Map<String, Lattice> meanField(int side, float coupling, int steps, long seed) {
        Supplier<IsingCoordination> env = () -> new IsingCoordination(side, 10, coupling);
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.TANH, 32);
        Map<String, Lattice> out = new LinkedHashMap<>();
        IsingCoordination e = env.get();
        MeanFieldQ mfq = MeanFieldQ.of(e, nets, seed);
        mfq.fit(e, steps);
        out.put("MF-Q", rollout(e, o -> mfq.act(o, false), mfq::beginEpisode, 5, seed));
        MeanFieldActorCritic mfac = MeanFieldActorCritic.of(e, nets, seed);
        mfac.fit(e, steps);
        out.put("MF-AC", rollout(e, o -> mfac.act(o, false), mfac::beginEpisode, 5, seed));
        MeanFieldQ iql = MeanFieldQ.of(e, nets, seed).independent();
        iql.fit(e, steps);
        out.put("IQL", rollout(e, o -> iql.act(o, false), iql::beginEpisode, 5, seed));
        MultiAgentPpo ppo = MultiAgentPpo.mappo(env, 32, seed).fit(steps);
        out.put("MAPPO", rollout(e, o -> ppo.act(o, false), () -> { }, 5, seed));
        SplittableRandom coin = new SplittableRandom(seed);
        out.put("random", rollout(e, o -> random(o.length, coin), () -> { }, 5, seed));
        return out;
    }

    private static int[] random(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextInt(2);
        }
        return a;
    }

    /**
     * 11.6.15: self-play and cross-play.
     *
     * @param episodes training episodes per seed
     * @param seeds independent seeds
     * @return row to (self-play, cross-play)
     */
    public static Map<String, OtherPlay.Result> otherPlay(int episodes, int seeds) {
        Map<String, OtherPlay.Result> out = new LinkedHashMap<>();
        OtherPlay.Game[] games = {OtherPlay.levers(10, 0.9), OtherPlay.signalling(4, 0.2)};
        for (OtherPlay.Game g : games) {
            for (OtherPlay.Learner l : OtherPlay.Learner.values()) {
                for (OtherPlay.Training t : OtherPlay.Training.values()) {
                    out.put(g + " " + l + " " + t, OtherPlay.crossPlay(g, l, t, episodes, seeds));
                }
            }
        }
        float[][] payoff = new float[10][10];
        for (int i = 0; i < 10; i++) {
            payoff[i][i] = i == 9 ? 0.9f : 1f;
        }
        MultiAgentPpo[] ppo = new MultiAgentPpo[seeds];
        for (int s = 0; s < seeds; s++) {
            ppo[s] = MultiAgentPpo.ippo(() -> new CooperativeMatrixGame("levers", payoff), 16, s)
                    .fit(episodes);
        }
        float[][] obs = CooperativeMatrixGame.climbing().reset(new SplittableRandom(0));
        double self = 0;
        double cross = 0;
        int pairs = 0;
        for (int i = 0; i < seeds; i++) {
            int a0 = ppo[i].act(obs, false)[0];
            for (int j = 0; j < seeds; j++) {
                int a1 = ppo[j].act(obs, false)[1];
                double v = payoff[a0][a1];
                if (i == j) {
                    self += v;
                } else {
                    cross += v;
                    pairs++;
                }
            }
        }
        out.put("levers(10) IPPO SELF_PLAY", new OtherPlay.Result(self / seeds, cross / pairs));
        ReferentialGame.Result ref = ReferentialGame.run(1500, 1);
        out.put("ReferentialGame (test acc, chance)", new OtherPlay.Result(ref.testAccuracy(), ref.chance()));
        return out;
    }

    /** The heterogeneous-agent tasks. */
    public enum HeteroTask {
        /** {@link ScoutCarrier} with four sites, optimum 1 */
        SCOUT_CARRIER,
        /** the climbing game, optimum 11 */
        CLIMBING
    }

    /**
     * 11.6.16: sequential trust-region updates against simultaneous ones.
     *
     * @param t the task
     * @param steps joint steps per learner
     * @param seeds seeds
     * @return learner to score
     */
    public static Map<String, Score> heterogeneous(HeteroTask t, int steps, int seeds) {
        Supplier<MultiAgentEnvironment> env = t == HeteroTask.SCOUT_CARRIER ? ScoutCarrier::new
                : CooperativeMatrixGame::climbing;
        double threshold = t == HeteroTask.SCOUT_CARRIER ? 0.95 : 11;
        Map<String, Score> out = new LinkedHashMap<>();
        for (Happo.Mode m : Happo.Mode.values()) {
            double[] r = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                r[s] = Happo.of(m, env, 32, s).fit(steps).evaluate(8, s);
            }
            out.put(m.toString(), score(r, threshold));
        }
        double[] mappo = new double[seeds];
        double[] ippo = new double[seeds];
        double[] qplex = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            mappo[s] = MultiAgentPpo.mappo(env, 32, s).fit(steps).evaluate(8, s);
            ippo[s] = MultiAgentPpo.ippo(env, 32, s).fit(steps).evaluate(8, s);
            MultiAgentEnvironment e = env.get();
            CompleteFactorization q = CompleteFactorization.of(CompleteFactorization.Method.QPLEX, e,
                    RlNetworkFactory.mlp(Activation.RELU, 32), 61 + s).epsilon(Schedule.constant(1)).seed(s);
            q.fit(e, steps);
            double total = 0;
            for (int k = 0; k < 8; k++) {
                total += q.evaluate(e, s * 100L + k);
            }
            qplex[s] = total / 8;
        }
        out.put("MAPPO", score(mappo, threshold));
        out.put("IPPO", score(ippo, threshold));
        out.put("QPLEX", score(qplex, threshold));
        return out;
    }

    /**
     * @param map a result map
     * @return one line
     */
    public static String line(Map<String, ?> map) {
        StringBuilder sb = new StringBuilder();
        map.forEach((k, v) -> sb.append(k).append('=').append(v instanceof Double d
                ? String.format(Locale.ROOT, "%.2f", d) : v).append(' '));
        return sb.toString().trim();
    }
}
