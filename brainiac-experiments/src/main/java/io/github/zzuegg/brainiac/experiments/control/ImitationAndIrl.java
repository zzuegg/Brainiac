package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.explore.InteractiveImitation;
import io.github.zzuegg.brainiac.experimental.explore.InverseRl;
import io.github.zzuegg.brainiac.experimental.rl.MaxEntIrl;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wave 12 items 12.6.4-12.6.5: interactive imitation and inverse RL.
 *
 * <ul>
 *   <li>{@link #imitation} (12.6.4): CartPole return against expert labels for behaviour cloning,
 *       DAgger, HG-DAgger (with expert interventions counted) and SafeDAgger-style gating, with a
 *       scripted linear expert and a softmax-regression learner;</li>
 *   <li>{@link #irl} (12.6.5): the expected value difference of apprenticeship learning, Bayesian IRL
 *       and deep MaxEnt IRL against the wave 6 linear {@code MaxEntIrl} and tabular behaviour cloning
 *       on a nonlinear object world, on the training layout and transferred to a new layout.</li>
 * </ul>
 *
 * <p>GAIL and AIRL (wave 6 and 7) are not re-run here: at these label budgets they need far more
 * environment interaction than the 10 s test budget allows.
 */
public final class ImitationAndIrl {

    private ImitationAndIrl() {
    }

    /** @return a scripted CartPole expert: push towards the side the pole is falling to */
    public static Agent cartPoleExpert() {
        return (o, explore) -> new float[] {o[2] + 0.5f * o[3] + 0.05f * o[0] + 0.2f * o[1] > 0 ? 1 : 0};
    }

    /**
     * 12.6.4 on CartPole.
     *
     * @param labelBudget expert labels per run
     * @param rounds data-collection rounds
     * @param seeds runs
     * @return scheme to {@code {mean final return, mean labels, mean interventions}}; the row
     *     {@code "expert"} holds the expert's return
     */
    public static Map<String, double[]> imitation(int labelBudget, int rounds, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        Agent expert = cartPoleExpert();
        out.put("expert", new double[] {Rl.evaluate(expert, new CartPole(), 5, 1), 0, 0});
        for (InteractiveImitation.Method m : InteractiveImitation.Method.values()) {
            double ret = 0;
            double labels = 0;
            double interventions = 0;
            for (int s = 0; s < seeds; s++) {
                InteractiveImitation.Result r = InteractiveImitation.of(m, 4, 2, s).rounds(rounds).evalEpisodes(5)
                        .run(CartPole::new, expert, o -> Math.abs(o[2]) > 0.06 || Math.abs(o[0]) > 1.2, labelBudget);
                ret += r.finalReturn();
                labels += r.labels()[r.labels().length - 1];
                interventions += r.interventions();
            }
            out.put(m.name(), new double[] {ret / seeds, labels / seeds, interventions / seeds});
        }
        return out;
    }

    /**
     * 12.6.5 on an object world.
     *
     * @param size the grid side
     * @param demonstrations expert trajectories
     * @param seed the layout seed (the transfer layout uses {@code seed + 1})
     * @return method to {@code {EVD on the training layout, EVD on the new layout (NaN if it cannot transfer)}}
     */
    public static Map<String, double[]> irl(int size, int demonstrations, long seed) {
        InverseRl.ObjectWorld train = InverseRl.objectWorld(size, size + 4, 0.3, 2 * size, seed);
        InverseRl.ObjectWorld test = InverseRl.objectWorld(size, size + 4, 0.3, 2 * size, seed + 1);
        OfflineDataset demos = InverseRl.demonstrations(train, demonstrations, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        MaxEntIrl linear = MaxEntIrl.of(train.mdp(), train.features()).learningRate(0.05);
        linear.fit(demos, 60);
        double[] w = linear.weights();
        out.put("MaxEntIrl (linear)", new double[] {InverseRl.expectedValueDifference(train, linear.reward()),
            InverseRl.expectedValueDifference(test, dot(test.features(), w))});
        for (InverseRl.Method m : InverseRl.Method.values()) {
            int iterations = switch (m) {
                case APPRENTICESHIP_PROJECTION -> 30;
                case BAYESIAN_IRL -> 600;
                case DEEP_MAXENT_IRL -> 120;
            };
            InverseRl.Fit fit = InverseRl.fit(m, train, demos, iterations, seed);
            double transfer = fit.transfer() == null ? Double.NaN
                    : InverseRl.expectedValueDifference(test, fit.transfer().apply(test.features()));
            out.put(m.name(), new double[] {InverseRl.expectedValueDifference(train, fit.reward()), transfer});
        }
        int n = train.mdp().states();
        int[][] votes = new int[n][train.mdp().actions()];
        for (OfflineDataset.Step s : demos.steps()) {
            int st = 0;
            for (int i = 1; i < n; i++) {
                if (s.observation()[i] > s.observation()[st]) {
                    st = i;
                }
            }
            votes[st][Math.round(s.action()[0])]++;
        }
        int[] bc = InverseRl.greedy(java.util.Arrays.stream(votes)
                .map(v -> java.util.Arrays.stream(v).asDoubleStream().toArray()).toArray(double[][]::new));
        int[] best = InverseRl.greedy(train.mdp().optimalQ(0.99, train.mdp().maxSteps()));
        int[] bestTest = InverseRl.greedy(test.mdp().optimalQ(0.99, test.mdp().maxSteps()));
        out.put("behaviour cloning", new double[] {InverseRl.policyReturn(train, best) - InverseRl.policyReturn(train, bc),
            InverseRl.policyReturn(test, bestTest) - InverseRl.policyReturn(test, bc)});
        return out;
    }

    private static double[] dot(double[][] phi, double[] w) {
        double[] r = new double[phi.length];
        for (int s = 0; s < phi.length; s++) {
            for (int j = 0; j < w.length; j++) {
                r[s] += w[j] * phi[s][j];
            }
        }
        return r;
    }
}
