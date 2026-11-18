package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 12 item 12.6.14: multi-objective RL on Deep Sea Treasure (a front with concave parts) and a
 * two-objective resource-gathering grid. Rows: linear scalarisation (tabular Q-learning per
 * preference weight, 11 weights), Pareto Q-learning, envelope Q-learning (one
 * preference-conditioned network), and scalarised core {@link Dqn} per weight; reported as the
 * hypervolume (reference {@code (0, -25)} on Deep Sea Treasure), the number of true Pareto points
 * found, and for envelope Q the mean relative utility loss on unseen preferences.
 */
public final class MultiObjectiveRl {

    private MultiObjectiveRl() {
    }

    private static io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.Envelope envelope(
            OperationsSims.VectorMdp mdp, double[] scale, int steps, long seed) {
        return io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.Envelope.of(mdp, scale, seed)
                .train(steps);
    }

    private static int found(List<double[]> points, double[][] front) {
        int k = 0;
        for (double[] f : front) {
            for (double[] p : points) {
                if (Math.abs(p[0] - f[0]) < 1e-6 && Math.abs(p[1] - f[1]) < 1e-6) {
                    k++;
                    break;
                }
            }
        }
        return k;
    }

    /**
     * 12.6.14 on Deep Sea Treasure.
     *
     * @param episodes tabular training episodes (per weight)
     * @param envelopeSteps environment steps of envelope Q-learning
     * @param dqnSteps environment steps of each scalarised DQN ({@code 0} to skip)
     * @param seed the seed
     * @return row to {@code {hypervolume, Pareto points found (of 10), utility loss on unseen weights}}
     */
    public static Map<String, double[]> deepSeaTreasure(int episodes, int envelopeSteps, int dqnSteps, long seed) {
        OperationsSims.VectorMdp dst = OperationsSims.deepSeaTreasure();
        double[] ref = {0, -25};
        double[][] front = OperationsSims.DEEP_SEA_TREASURE_FRONT;
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] trueHv = {io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(
                List.of(front), ref)};
        out.put("true front", new double[] {trueHv[0], front.length, 0});
        List<double[]> linear = new ArrayList<>();
        for (int k = 0; k <= 10; k++) {
            double a = k / 10.0;
            linear.add(io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.scalarizedQ(dst,
                    new double[] {a / 124, (1 - a) / 19}, episodes, seed + k));
        }
        out.put("linear scalarisation (11 weights)", new double[] {
            io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(linear, ref),
            found(linear, front), Double.NaN});
        List<double[]> pareto = io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.paretoQ(dst, episodes,
                seed);
        out.put("Pareto Q-learning", new double[] {
            io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(pareto, ref),
            found(pareto, front), Double.NaN});
        var env = envelope(dst, new double[] {124, 19}, envelopeSteps, seed);
        List<double[]> envPoints = new ArrayList<>();
        for (int k = 0; k <= 10; k++) {
            double a = k / 10.0;
            envPoints.add(env.returnFor(new double[] {a, 1 - a}));
        }
        double loss = 0;
        int unseen = 0;
        for (double a = 0.05; a < 1; a += 0.15) {
            double[] w = {a / 124, (1 - a) / 19};
            double best = Double.NEGATIVE_INFINITY;
            for (double[] f : front) {
                best = Math.max(best, w[0] * f[0] + w[1] * f[1]);
            }
            double[] got = env.returnFor(new double[] {a, 1 - a});
            double worst = w[0] * 0 + w[1] * -dst.maxSteps();
            loss += (best - (w[0] * got[0] + w[1] * got[1])) / (best - worst);
            unseen++;
        }
        out.put("envelope Q (one network)", new double[] {
            io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(envPoints, ref),
            found(envPoints, front), loss / unseen});
        if (dqnSteps > 0) {
            List<double[]> dqnPoints = new ArrayList<>();
            for (double a : new double[] {0, 0.5, 1}) {
                double[] w = {a / 124, (1 - a) / 19};
                Environment e = io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.scalarised(dst, w);
                Dqn dqn = Dqn.of(() -> Nets.mlp(dst.states(), 4, seed, 64), 4).optimizer(Optimizer.adam(1e-3f))
                        .learningStarts(200).batchSize(32).targetUpdateEvery(100).discount(1f)
                        .epsilon(Schedule.linear(1, 0.05, dqnSteps / 2)).seed(seed);
                dqn.fit(e, dqnSteps);
                dqnPoints.add(io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.rollout(dst,
                        s -> Math.round(greedy(dqn, dst.states(), s)[0])));
            }
            out.put("scalarised DQN (3 weights)", new double[] {
                io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(dqnPoints, ref),
                found(dqnPoints, front), Double.NaN});
        }
        double[] randomRet = io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.rollout(dst,
                new java.util.function.IntUnaryOperator() {
                    private final SplittableRandom r = new SplittableRandom(seed);

                    @Override
                    public int applyAsInt(int s) {
                        return r.nextInt(4);
                    }
                });
        out.put("random policy", new double[] {
            io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.hypervolume(List.of(randomRet), ref),
            found(List.<double[]>of(randomRet), front), Double.NaN});
        return out;
    }

    private static float[] greedy(Agent agent, int states, int s) {
        float[] o = new float[states];
        o[s] = 1;
        return agent.act(o, false);
    }

    /**
     * 12.6.14 on the resource-gathering grid: the coverage sets of linear scalarisation and Pareto
     * Q-learning.
     *
     * @param episodes training episodes
     * @param seed the seed
     * @return row to the non-dominated return vectors found, flattened
     */
    public static Map<String, double[]> resourceGathering(int episodes, long seed) {
        OperationsSims.VectorMdp rg = OperationsSims.resourceGathering();
        Map<String, double[]> out = new LinkedHashMap<>();
        List<double[]> linear = new ArrayList<>();
        for (int k = 0; k <= 10; k++) {
            double a = k / 10.0;
            linear.add(io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.scalarizedQ(rg,
                    new double[] {a, 1 - a}, episodes, seed + k));
        }
        out.put("linear scalarisation", flatten(
                io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.nondominated(linear)));
        out.put("Pareto Q-learning", flatten(
                io.github.zzuegg.brainiac.experimental.explore.MultiObjectiveRl.paretoQ(rg, episodes, seed)));
        return out;
    }

    private static double[] flatten(List<double[]> points) {
        double[] f = new double[2 * points.size()];
        for (int i = 0; i < points.size(); i++) {
            f[2 * i] = points.get(i)[0];
            f[2 * i + 1] = points.get(i)[1];
        }
        return f;
    }
}
