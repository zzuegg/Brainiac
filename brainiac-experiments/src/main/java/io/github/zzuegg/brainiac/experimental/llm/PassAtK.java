package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The unbiased pass@k estimator (Chen et al., "Evaluating large language models trained on
 * code", <a href="https://arxiv.org/abs/2107.03374">2107.03374</a>): draw {@code n >= k}
 * samples per problem, count the {@code c} correct ones, and estimate the chance that at least
 * one of {@code k} samples is correct as {@code 1 - C(n - c, k) / C(n, k)}. Computing it from
 * {@code n} samples for every {@code k <= n} is far less noisy than drawing {@code k} samples
 * per {@code k}.
 *
 * <p>Pass@k curves up to large {@code k} are how Yue et al. ("Does reinforcement learning
 * really incentivize reasoning capacity beyond the base model?",
 * <a href="https://arxiv.org/abs/2504.13837">2504.13837</a>) tell <em>sharpening</em> (better
 * pass@1, same or lower pass@256) from widened <em>coverage</em>: {@link #curve} samples a
 * {@link WindowLm} and returns the mean curve over a problem set, and {@link #crossover}
 * finds where two curves cross.
 */
public final class PassAtK {

    /** Decides whether a completion solves a prompt. */
    @FunctionalInterface
    public interface Verifier {
        /**
         * @param prompt the prompt
         * @param completion a sampled completion
         * @return whether it is correct
         */
        boolean correct(int[] prompt, int[] completion);
    }

    private PassAtK() {
    }

    /**
     * The unbiased estimator, computed as the numerically stable product
     * {@code 1 - prod_{i = n - c + 1}^{n} (1 - k / i)}.
     *
     * @param n samples drawn
     * @param c correct samples among them
     * @param k the k of pass@k, at most {@code n}
     * @return the estimated pass@k
     */
    public static double estimate(int n, int c, int k) {
        if (k < 1 || k > n || c < 0 || c > n) {
            throw new IllegalArgumentException("need 1 <= k <= n and 0 <= c <= n, got n=" + n + " c=" + c + " k=" + k);
        }
        if (n - c < k) {
            return 1;
        }
        double fail = 1;
        for (int i = n - c + 1; i <= n; i++) {
            fail *= 1 - (double) k / i;
        }
        return 1 - fail;
    }

    /**
     * Counts correct samples per prompt.
     *
     * @param lm the model
     * @param prompts the problems
     * @param verifier judges completions
     * @param n samples per prompt
     * @param maxTokens the most tokens per completion
     * @param end the end token
     * @param temperature the sampling temperature
     * @param random the random source
     * @return the number of correct samples for each prompt
     */
    public static int[] correctCounts(WindowLm lm, List<int[]> prompts, Verifier verifier, int n, int maxTokens,
            int end, double temperature, SplittableRandom random) {
        int[] counts = new int[prompts.size()];
        for (int i = 0; i < prompts.size(); i++) {
            int[] p = prompts.get(i);
            int[][] group = new int[n][];
            Arrays.fill(group, p);
            for (int[] o : lm.sample(group, maxTokens, end, temperature, random)) {
                counts[i] += verifier.correct(p, o) ? 1 : 0;
            }
        }
        return counts;
    }

    /**
     * The mean pass@k over problems for each requested k, from per-problem counts.
     *
     * @param counts correct samples per problem
     * @param n samples per problem
     * @param ks the k values, each at most {@code n}
     * @return the mean pass@k for each k
     */
    public static double[] curve(int[] counts, int n, int... ks) {
        double[] out = new double[ks.length];
        for (int j = 0; j < ks.length; j++) {
            double s = 0;
            for (int c : counts) {
                s += estimate(n, c, ks[j]);
            }
            out[j] = s / counts.length;
        }
        return out;
    }

    /**
     * Samples a model and returns its mean pass@k curve.
     *
     * @param lm the model
     * @param prompts the problems
     * @param verifier judges completions
     * @param n samples per prompt, at least the largest k
     * @param maxTokens the most tokens per completion
     * @param end the end token
     * @param temperature the sampling temperature
     * @param random the random source
     * @param ks the k values
     * @return the mean pass@k for each k
     */
    public static double[] curve(WindowLm lm, List<int[]> prompts, Verifier verifier, int n, int maxTokens, int end,
            double temperature, SplittableRandom random, int... ks) {
        return curve(correctCounts(lm, prompts, verifier, n, maxTokens, end, temperature, random), n, ks);
    }

    /**
     * The first k at which curve {@code b} reaches curve {@code a} after starting below it,
     * the "crossover" of an RL model ({@code a}, better at pass@1) and its base ({@code b}).
     *
     * @param ks the k values of both curves
     * @param a the first curve
     * @param b the second curve
     * @return the first {@code ks[j]} with {@code b[j] >= a[j]} after {@code b[0] < a[0]}, or -1
     */
    public static int crossover(int[] ks, double[] a, double[] b) {
        if (b[0] >= a[0]) {
            return -1;
        }
        for (int j = 1; j < ks.length; j++) {
            if (b[j] >= a[j]) {
                return ks[j];
            }
        }
        return -1;
    }

    /**
     * Powers of two from 1 up to and including {@code max}.
     *
     * @param max the largest k (a power of two)
     * @return {@code 1, 2, 4, ..., max}
     */
    public static int[] powersOfTwo(int max) {
        int count = 0;
        for (int k = 1; k <= max; k *= 2) {
            count++;
        }
        int[] ks = new int[count];
        for (int i = 0, k = 1; i < count; i++, k *= 2) {
            ks[i] = k;
        }
        return ks;
    }
}
