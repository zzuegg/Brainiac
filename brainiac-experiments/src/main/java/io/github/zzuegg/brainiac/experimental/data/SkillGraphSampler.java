package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.function.IntFunction;
import java.util.random.RandomGenerator;

/**
 * Skill-It (Chen et al., "Skill-it! A Data-Driven Skills Framework for Understanding and
 * Training Language Models", NeurIPS 2023, <a href="https://arxiv.org/abs/2307.14430">arXiv:2307.14430</a>).
 * Skills have prerequisites, and training on a prerequisite lowers the loss of later skills
 * faster than training on them directly.
 * <ul>
 *   <li>{@link #estimate}: the skill graph from short probe runs: from a common starting point,
 *       train on skill {@code i} alone and record each skill's loss drop; {@code A_ij} is the
 *       drop of skill {@code j} after training on {@code i}, clipped at zero and scaled so each
 *       column's diagonal is 1;</li>
 *   <li>{@link #estimateByAblation}: the same graph from leave-one-out runs, which avoids the
 *       interference of training on one skill alone: {@code A_ij} is how much higher skill
 *       {@code j}'s loss ends when skill {@code i} is left out of an otherwise uniform mixture;</li>
 *   <li>the online sampler: mirror descent with skill weights
 *       {@code p_i ∝ exp(η Σ_τ Σ_{j ∈ targets} A_ij L_j(τ))}, so a skill's weight rises with the
 *       remaining loss of the target skills it helps.</li>
 * </ul>
 */
public final class SkillGraphSampler {

    private final double[][] graph;
    private final boolean[] targets;
    private final double eta;
    private final double[] logits;
    private double[] weights;

    /**
     * Creates the online sampler.
     *
     * @param graph {@code A_ij}, the help of skill {@code i} to skill {@code j}
     * @param targets the skills whose loss is to be minimised
     * @param eta the mirror-descent step, for example 0.5
     */
    public SkillGraphSampler(double[][] graph, boolean[] targets, double eta) {
        this.graph = graph;
        this.targets = targets.clone();
        this.eta = eta;
        this.logits = new double[graph.length];
        this.weights = new double[graph.length];
        Arrays.fill(weights, 1.0 / graph.length);
    }

    /**
     * Estimates the skill graph from probe runs.
     *
     * @param skills the number of skills
     * @param initial every skill's loss at the common starting point
     * @param probe trains a copy of the starting model on skill {@code i} alone and returns every
     *     skill's loss afterwards
     * @return {@code A_ij}
     */
    public static double[][] estimate(int skills, double[] initial, IntFunction<double[]> probe) {
        double[][] a = new double[skills][skills];
        for (int i = 0; i < skills; i++) {
            double[] after = probe.apply(i);
            for (int j = 0; j < skills; j++) {
                a[i][j] = Math.max(0, initial[j] - after[j]);
            }
        }
        for (int j = 0; j < skills; j++) {
            double d = a[j][j];
            for (int i = 0; i < skills; i++) {
                a[i][j] = d > 0 ? a[i][j] / d : a[i][j];
            }
        }
        return a;
    }

    /**
     * Estimates the skill graph by leaving skills out of a uniform mixture.
     *
     * @param skills the number of skills
     * @param all every skill's loss after a probe run on all skills
     * @param without trains a probe on every skill but {@code i} and returns every skill's loss
     * @return {@code A_ij}, clipped at zero, each column scaled so its diagonal is 1
     */
    public static double[][] estimateByAblation(int skills, double[] all, IntFunction<double[]> without) {
        double[][] a = new double[skills][skills];
        for (int i = 0; i < skills; i++) {
            double[] w = without.apply(i);
            for (int j = 0; j < skills; j++) {
                a[i][j] = Math.max(0, w[j] - all[j]);
            }
        }
        for (int j = 0; j < skills; j++) {
            double d = a[j][j];
            for (int i = 0; i < skills; i++) {
                a[i][j] = d > 0 ? a[i][j] / d : a[i][j];
            }
        }
        return a;
    }

    /**
     * The edges of a graph above a threshold, without the diagonal.
     *
     * @param graph {@code A_ij}
     * @param threshold the edge threshold, for example 0.3
     * @return the adjacency matrix
     */
    public static boolean[][] edges(double[][] graph, double threshold) {
        boolean[][] e = new boolean[graph.length][graph.length];
        for (int i = 0; i < graph.length; i++) {
            for (int j = 0; j < graph.length; j++) {
                e[i][j] = i != j && graph[i][j] > threshold;
            }
        }
        return e;
    }

    /** @return the current skill weights */
    public double[] weights() {
        return weights.clone();
    }

    /** @return the graph */
    public double[][] graph() {
        return graph.clone();
    }

    /**
     * Updates the weights with the current per-skill (validation) losses.
     *
     * @param losses each skill's loss
     * @return the new weights
     */
    public double[] observe(double[] losses) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            for (int j = 0; j < losses.length; j++) {
                if (targets[j]) {
                    logits[i] += eta * graph[i][j] * losses[j];
                }
            }
            max = Math.max(max, logits[i]);
        }
        double sum = 0;
        double[] w = new double[logits.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = Math.exp(logits[i] - max);
            sum += w[i];
        }
        for (int i = 0; i < w.length; i++) {
            w[i] /= sum;
        }
        weights = w;
        return w.clone();
    }

    /**
     * Samples a skill from the current weights.
     *
     * @param random the random source
     * @return the skill
     */
    public int sample(RandomGenerator random) {
        double u = random.nextDouble();
        for (int i = 0; i < weights.length - 1; i++) {
            u -= weights[i];
            if (u < 0) {
                return i;
            }
        }
        return weights.length - 1;
    }

    @Override
    public String toString() {
        return "SkillGraphSampler" + Arrays.toString(weights);
    }
}
