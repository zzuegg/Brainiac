package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.AccessControlQueue;
import io.github.zzuegg.brainiac.experimental.deeprl.AverageReward;
import io.github.zzuegg.brainiac.experimental.explore.PointMaze;
import io.github.zzuegg.brainiac.experimental.explore.PreferenceRl;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Other ways to state what an agent should optimise (wave 11, 11.6.11-11.6.12).
 *
 * <ul>
 *   <li>{@link #preferences}: true return and success on a point-mass reach task for
 *       {@link PreferenceRl} trained from a scripted teacher's segment comparisons (Christiano
 *       et al., PEBBLE, SURF), with and without label noise, against the same TD3 on the true
 *       reward;</li>
 *   <li>{@link #averageReward}: the long-run reward rate on the access-control queue for
 *       differential Q-learning, R-learning and deep differential Q against the core discounted
 *       {@link Tabular} Q-learning at {@code gamma} 0.9 and 0.99, with the optimum from relative
 *       value iteration.</li>
 * </ul>
 */
public final class RewardFormulations {

    private RewardFormulations() {
    }

    private static PointMaze reach() {
        return new PointMaze(PointMaze.Layout.OPEN, 40, new float[] {-0.6f, -0.6f}, new float[] {0.5f, 0.5f})
                .dense(true);
    }

    private static double[] evaluate(Agent agent, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        int ok = 0;
        Environment env = reach();
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(r.split());
            agent.startEpisode();
            Environment.Step s;
            do {
                s = env.step(agent.act(obs, false));
                total += s.reward();
                obs = s.observation();
            } while (!s.done());
            if (s.terminated()) {
                ok++;
            }
        }
        return new double[] {total / episodes, ok / (double) episodes};
    }

    /**
     * 11.6.11: preference-based RL against learning from the true reward.
     *
     * @param steps environment steps
     * @param feedback teacher queries in total
     * @param seeds seeds
     * @param width hidden width of the TD3 networks
     * @return row to {mean true return, success rate, queries asked}
     */
    public static Map<String, double[]> preferences(long steps, int feedback, int seeds, int width) {
        Map<String, double[]> out = new LinkedHashMap<>();
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, width, width);
        Object[][] rows = {
            {"TRUE_REWARD", PreferenceRl.Method.TRUE_REWARD, 0.0},
            {"CHRISTIANO", PreferenceRl.Method.CHRISTIANO, 0.0},
            {"PEBBLE", PreferenceRl.Method.PEBBLE, 0.0},
            {"SURF", PreferenceRl.Method.SURF, 0.0},
            {"PEBBLE (10% label noise)", PreferenceRl.Method.PEBBLE, 0.1},
            {"SURF (10% label noise)", PreferenceRl.Method.SURF, 0.1},
        };
        int every = (int) Math.max(100, steps / 8);
        for (Object[] row : rows) {
            double ret = 0;
            double ok = 0;
            double asked = 0;
            for (int s = 0; s < seeds; s++) {
                PreferenceRl p = PreferenceRl.of((PreferenceRl.Method) row[1], 4, 2, nets, s)
                        .feedback(feedback, Math.max(1, feedback / 6), every).labelNoise((double) row[2])
                        .pretrainSteps((int) Math.min(300, steps / 6));
                p.fit(reach(), steps, null);
                double[] e = evaluate(p, 3, 50 + s);
                ret += e[0];
                ok += e[1];
                asked += p.queries();
            }
            out.put((String) row[0], new double[] {ret / seeds, ok / seeds, asked / seeds});
        }
        return out;
    }

    /**
     * 11.6.12: average-reward control on a continuing task.
     *
     * @param steps environment steps per learner
     * @param seeds seeds
     * @return row to the mean reward rate of the greedy policy (the optimum is the
     *     {@code "optimal (RVI)"} row)
     */
    public static Map<String, Double> averageReward(long steps, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        AccessControlQueue reference = new AccessControlQueue();
        out.put("optimal (RVI)", reference.optimalRewardRate());
        double dq = 0;
        double rl = 0;
        double deep = 0;
        double g90 = 0;
        double g99 = 0;
        for (int s = 0; s < seeds; s++) {
            AverageReward d = AverageReward.tabular(AverageReward.Method.DIFFERENTIAL_Q, 2).stepSizes(0.1, 0.1)
                    .seed(s);
            d.fit(new AccessControlQueue(), steps, null);
            dq += reference.rewardRate(d, 20_000, 7 + s);
            AverageReward r = AverageReward.tabular(AverageReward.Method.R_LEARNING, 2).stepSizes(0.1, 0.01).seed(s);
            r.fit(new AccessControlQueue(), steps, null);
            rl += reference.rewardRate(r, 20_000, 7 + s);
            AverageReward n = AverageReward.deep(2, 2, RlNetworkFactory.mlp(Activation.RELU, 32), s).seed(s);
            n.fit(new AccessControlQueue(), Math.min(steps, 4_000), null);
            deep += reference.rewardRate(n, 20_000, 7 + s);
            Tabular q90 = Tabular.qLearning(2).learningRate(0.1).discount(0.9).epsilon(Schedule.constant(0.1))
                    .seed(s);
            q90.fit(new AccessControlQueue(), steps, null);
            g90 += reference.rewardRate(q90, 20_000, 7 + s);
            Tabular q99 = Tabular.qLearning(2).learningRate(0.1).discount(0.99).epsilon(Schedule.constant(0.1))
                    .seed(s);
            q99.fit(new AccessControlQueue(), steps, null);
            g99 += reference.rewardRate(q99, 20_000, 7 + s);
        }
        out.put("differential Q", dq / seeds);
        out.put("R-learning", rl / seeds);
        out.put("deep differential Q (4k steps)", deep / seeds);
        out.put("Q-learning gamma 0.9", g90 / seeds);
        out.put("Q-learning gamma 0.99", g99 / seeds);
        out.put("always accept", reference.rewardRate((o, e) -> new float[] {1}, 20_000, 7));
        return out;
    }

    /**
     * @param table a result table
     * @return one line
     */
    public static String fmt(Map<String, double[]> table) {
        StringBuilder sb = new StringBuilder();
        table.forEach((k, v) -> {
            sb.append(k).append('=');
            for (int i = 0; i < v.length; i++) {
                sb.append(String.format(Locale.ROOT, i == 0 ? "%.2f" : "/%.2f", v[i]));
            }
            sb.append(' ');
        });
        return sb.toString().trim();
    }

    /**
     * @param table a result table
     * @return one line
     */
    public static String line(Map<String, Double> table) {
        StringBuilder sb = new StringBuilder();
        table.forEach((k, v) -> sb.append(k).append(String.format(Locale.ROOT, "=%.3f ", v)));
        return sb.toString().trim();
    }

    /**
     * Prints the tables at longer budgets.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println(fmt(preferences(6_000, 200, 3, 64)));
        System.out.println(line(averageReward(200_000, 3)));
    }
}
