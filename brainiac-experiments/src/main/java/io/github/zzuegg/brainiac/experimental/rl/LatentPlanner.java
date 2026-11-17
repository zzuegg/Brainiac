package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * PlaNet's planner (Hafner et al., 2019, <a href="https://arxiv.org/abs/1811.04551">arXiv:1811.04551</a>):
 * the cross-entropy method over action sequences, evaluated entirely in the latent space of a
 * {@link RecurrentStateSpaceModel} with its learned reward, so no observation is ever
 * decoded while planning. The agent keeps a filtered belief: every real observation updates
 * it through the posterior. Only the first action of the plan is executed.
 */
public final class LatentPlanner implements Agent {

    private final RecurrentStateSpaceModel model;
    private final float[] low;
    private final float[] high;
    private int horizon = 12;
    private int population = 100;
    private int elites = 10;
    private int iterations = 3;
    private SplittableRandom random = new SplittableRandom(0);
    private RecurrentStateSpaceModel.Belief belief;
    private float[] lastAction;

    private LatentPlanner(RecurrentStateSpaceModel model, Space.Box actions) {
        this.model = model;
        this.low = actions.low();
        this.high = actions.high();
    }

    /**
     * @param model the trained latent model
     * @param actions the action box
     * @return the planner
     */
    public static LatentPlanner of(RecurrentStateSpaceModel model, Space.Box actions) {
        return new LatentPlanner(model, actions);
    }

    /** @param steps the planning horizon @return this planner */
    public LatentPlanner horizon(int steps) {
        this.horizon = steps;
        return this;
    }

    /**
     * @param samples candidate sequences per iteration
     * @param best elites refit per iteration
     * @param rounds iterations
     * @return this planner
     */
    public LatentPlanner search(int samples, int best, int rounds) {
        this.population = samples;
        this.elites = best;
        this.iterations = rounds;
        return this;
    }

    /** @param seed the seed @return this planner */
    public LatentPlanner seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    @Override
    public void startEpisode() {
        belief = null;
        lastAction = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        belief = model.observe(lastAction == null ? null : belief, lastAction, observation);
        int d = low.length;
        int len = horizon * d;
        float[] mean = new float[len];
        float[] std = new float[len];
        for (int i = 0; i < len; i++) {
            mean[i] = (low[i % d] + high[i % d]) / 2;
            std[i] = (high[i % d] - low[i % d]) / 2;
        }
        float[][] seqs = new float[population][len];
        double[] returns = new double[population];
        for (int it = 0; it < iterations; it++) {
            for (int p = 0; p < population; p++) {
                for (int i = 0; i < len; i++) {
                    int k = i % d;
                    double a = mean[i] + std[i] * random.nextGaussian();
                    seqs[p][i] = (float) Math.max(low[k], Math.min(high[k], a));
                }
            }
            Arrays.fill(returns, 0);
            RecurrentStateSpaceModel.Belief b = RecurrentStateSpaceModel.repeat(belief, population);
            for (int t = 0; t < horizon; t++) {
                float[] act = new float[population * d];
                for (int p = 0; p < population; p++) {
                    System.arraycopy(seqs[p], t * d, act, p * d, d);
                }
                b = model.imagine(b, Tensor.of(Shape.of(population, d), act));
                float[] r = model.reward(b).to(DType.FLOAT32).toArray();
                for (int p = 0; p < population; p++) {
                    returns[p] += r[p];
                }
            }
            Integer[] order = new Integer[population];
            for (int p = 0; p < population; p++) {
                order[p] = p;
            }
            Arrays.sort(order, (x, y) -> Double.compare(returns[y], returns[x]));
            for (int i = 0; i < len; i++) {
                double m = 0;
                for (int e = 0; e < elites; e++) {
                    m += seqs[order[e]][i];
                }
                m /= elites;
                double v = 0;
                for (int e = 0; e < elites; e++) {
                    v += (seqs[order[e]][i] - m) * (seqs[order[e]][i] - m);
                }
                mean[i] = (float) m;
                std[i] = (float) Math.sqrt(v / elites + 1e-6);
            }
        }
        float[] a = Arrays.copyOf(mean, d);
        if (explore) {
            for (int i = 0; i < d; i++) {
                a[i] = (float) Math.max(low[i], Math.min(high[i], a[i] + 0.3 * (high[i] - low[i]) / 2
                        * random.nextGaussian()));
            }
        }
        lastAction = a;
        return a.clone();
    }
}
