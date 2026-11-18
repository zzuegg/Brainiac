package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.LatentPlanner;
import io.github.zzuegg.brainiac.experimental.rl.RecurrentStateSpaceModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * W6 3.7: PlaNet on the pendulum. A recurrent state-space model (with and without latent
 * overshooting) learns dynamics <em>and reward</em> from random episodes, and
 * {@link LatentPlanner} plans with CEM in its latent space. Compared with the
 * observation-space MLP dynamics model of {@link CemPlanner#learn} planned with CEM and the
 * true reward function, learned from the same number of random steps. Metrics: greedy
 * return and the open-loop observation error 10 steps ahead on held-out random episodes.
 *
 * <p>Paper: Hafner et al., <a href="https://arxiv.org/abs/1811.04551">arXiv:1811.04551</a>.
 */
public final class LatentPlanningComparison {

    private static final int CONTEXT = 5;

    private LatentPlanningComparison() {
    }

    /** A method's outcome. @param ret the greedy return @param error the k-step open-loop squared error */
    public record Outcome(double ret, double error) {
    }

    /**
     * @param episodes random training episodes (100 steps each)
     * @param modelSteps RSSM gradient steps
     * @param ahead the prediction distance
     * @param seed the seed
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int episodes, int modelSteps, int ahead, long seed) {
        Pendulum env = new Pendulum(100);
        List<RecurrentStateSpaceModel.Sequence> data = RecurrentStateSpaceModel.collectRandom(env, episodes, seed);
        List<RecurrentStateSpaceModel.Sequence> test = RecurrentStateSpaceModel.collectRandom(env, 10, seed + 50);
        Space.Box box = (Space.Box) env.actionSpace();
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (boolean over : new boolean[] {true, false}) {
            RecurrentStateSpaceModel m = RecurrentStateSpaceModel.of(3, 1, 32, 8, 64, seed)
                    .overshooting(over ? 3 : 1, 0.5f).train(data, modelSteps, 16, 12);
            LatentPlanner planner = LatentPlanner.of(m, box).horizon(10).search(80, 8, 3).seed(seed);
            out.put(over ? "PlaNet (RSSM + overshooting)" : "RSSM without overshooting",
                    new Outcome(Rl.evaluate(planner, env, 2, seed), rssmError(m, test, ahead)));
        }
        CemPlanner.Dynamics mlp = CemPlanner.learn(env, Nets.mlp(4, 3, Activation.RELU, seed, 64, 64),
                episodes * 100, 1500, seed);
        CemPlanner cem = CemPlanner.of(mlp, ContinuousControlComparison::pendulumReward, box).horizon(10)
                .search(80, 8, 3).seed(seed);
        out.put("MLP dynamics + CEM (true reward)", new Outcome(Rl.evaluate(cem, env, 2, seed),
                mlpError(mlp, test, ahead)));
        return out;
    }

    private static double rssmError(RecurrentStateSpaceModel m, List<RecurrentStateSpaceModel.Sequence> test,
            int ahead) {
        double err = 0;
        for (RecurrentStateSpaceModel.Sequence q : test) {
            RecurrentStateSpaceModel.Belief b = null;
            for (int t = 0; t < CONTEXT; t++) {
                b = m.observe(b, t == 0 ? null : q.actions()[t - 1], q.observations()[t]);
            }
            for (int k = 0; k < ahead; k++) {
                b = m.imagine(b, Tensor.of(Shape.of(1, 1), q.actions()[CONTEXT - 1 + k]));
            }
            err += sq(m.decode(b), q.observations()[CONTEXT - 1 + ahead]);
        }
        return err / test.size();
    }

    private static double mlpError(CemPlanner.Dynamics d, List<RecurrentStateSpaceModel.Sequence> test, int ahead) {
        double err = 0;
        for (RecurrentStateSpaceModel.Sequence q : test) {
            Tensor s = Tensor.of(Shape.of(1, 3), q.observations()[CONTEXT - 1]);
            for (int k = 0; k < ahead; k++) {
                s = d.next(s, Tensor.of(Shape.of(1, 1), q.actions()[CONTEXT - 1 + k]));
            }
            err += sq(s.toArray(), q.observations()[CONTEXT - 1 + ahead]);
        }
        return err / test.size();
    }

    private static double sq(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < b.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }
}
