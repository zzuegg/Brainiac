package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.DifferentiableSim;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Controller optimisation through a differentiable simulator (DiffTaichi, Hu et al.,
 * <a href="https://arxiv.org/abs/1910.00935">arXiv:1910.00935</a>) against two black-box
 * learners at the same number of simulated episodes: cross-entropy search over the
 * controller's weights (Rubinstein's CEM, as in Salimans et al.,
 * <a href="https://arxiv.org/abs/1703.03864">arXiv:1703.03864</a>, for evolution strategies)
 * and the core PPO (Schulman et al., <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a>).
 * All three train the same {@code 10-32-32-2} tanh MLP on {@link DifferentiableSim#massSpring}:
 * swing a ball on a spring to a random goal. The score is the mean final ball-to-goal
 * distance on 200 held-out goals.
 */
public final class DifferentiablePhysicsControl {
    private static final int BATCH = 10;
    private static final int CEM_POPULATION = 20;
    private static final int CEM_ELITES = 5;
    private static final int CEM_GOALS = 5;

    /**
     * The outcome: mean final distance to the goal (lower is better).
     *
     * @param untrained the initial controller
     * @param bptt gradient descent through the simulator
     * @param cem cross-entropy search over the weights
     * @param ppo the core PPO
     */
    public record Result(double untrained, double bptt, double cem, double ppo) {
    }

    private DifferentiablePhysicsControl() {
    }

    private static Model controller(long seed) {
        return Nets.mlp(10, 2, seed, 32, 32);
    }

    /**
     * Trains the three controllers on the same episode budget.
     *
     * @param episodes simulated episodes per method
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int episodes, long seed) {
        DifferentiableSim sim = DifferentiableSim.massSpring();
        Tensor test = sim.goals(200, new SplittableRandom(seed + 99));
        double untrained = sim.finalDistance(controller(seed), test);
        return new Result(untrained, sim.finalDistance(bptt(sim, episodes, seed), test),
                sim.finalDistance(cem(sim, episodes, seed), test), sim.finalDistance(ppo(sim, episodes, seed), test));
    }

    /**
     * Backpropagation through the simulator: Adam on the mean final cost of batches of 10 episodes.
     *
     * @param sim the simulator
     * @param episodes the episode budget
     * @param seed the seed
     * @return the trained controller
     */
    public static Model bptt(DifferentiableSim sim, int episodes, long seed) {
        Model c = controller(seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < episodes / BATCH; i++) {
            sim.loss(c, sim.goals(BATCH, r)).backward();
            opt.step(c.parameters());
        }
        return c;
    }

    /**
     * Cross-entropy search over the flattened weights: 20 candidates per generation, each scored
     * on the same 5 goals, the 5 best refit a diagonal Gaussian.
     *
     * @param sim the simulator
     * @param episodes the episode budget
     * @param seed the seed
     * @return the best mean found
     */
    public static Model cem(DifferentiableSim sim, int episodes, long seed) {
        Model c = controller(seed);
        List<Var> params = c.parameters();
        int n = 0;
        for (Var p : params) {
            n += p.value().toArray().length;
        }
        double[] mean = new double[n];
        int k = 0;
        for (Var p : params) {
            for (float v : p.value().toArray()) {
                mean[k++] = v;
            }
        }
        double[] std = new double[n];
        Arrays.fill(std, 0.3);
        SplittableRandom r = new SplittableRandom(seed);
        int generations = episodes / (CEM_POPULATION * CEM_GOALS);
        for (int gen = 0; gen < generations; gen++) {
            Tensor goals = sim.goals(CEM_GOALS, r);
            double[][] cand = new double[CEM_POPULATION][n];
            double[] score = new double[CEM_POPULATION];
            Integer[] order = new Integer[CEM_POPULATION];
            for (int i = 0; i < CEM_POPULATION; i++) {
                for (int j = 0; j < n; j++) {
                    cand[i][j] = mean[j] + std[j] * r.nextGaussian();
                }
                load(params, cand[i]);
                score[i] = sim.cost(sim.rollout(c, goals, false), Var.of(goals)).mean().value().item();
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(score[a], score[b]));
            for (int j = 0; j < n; j++) {
                double m = 0;
                for (int e = 0; e < CEM_ELITES; e++) {
                    m += cand[order[e]][j];
                }
                m /= CEM_ELITES;
                double v = 0;
                for (int e = 0; e < CEM_ELITES; e++) {
                    double d = cand[order[e]][j] - m;
                    v += d * d;
                }
                mean[j] = m;
                std[j] = Math.sqrt(v / CEM_ELITES) + 0.01;
            }
        }
        load(params, mean);
        return c;
    }

    private static void load(List<Var> params, double[] flat) {
        int k = 0;
        for (Var p : params) {
            float[] v = new float[p.value().toArray().length];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) flat[k++];
            }
            p.assign(Tensor.of(p.shape(), v));
        }
    }

    /**
     * The core PPO on the simulator as a plain environment, for {@code episodes * horizon} steps.
     *
     * @param sim the simulator
     * @param episodes the episode budget
     * @param seed the seed
     * @return the trained policy (its means are the actions)
     */
    public static Model ppo(DifferentiableSim sim, int episodes, long seed) {
        Ppo ppo = Ppo.of(controller(seed), Nets.mlp(10, 1, seed + 1, 32, 32), sim.asEnvironment().actionSpace())
                .rolloutSteps(10 * sim.horizon()).learningRate(1e-3f).seed(seed);
        ppo.fit(sim.asEnvironment(), (long) episodes * sim.horizon());
        return ppo.policy();
    }
}
