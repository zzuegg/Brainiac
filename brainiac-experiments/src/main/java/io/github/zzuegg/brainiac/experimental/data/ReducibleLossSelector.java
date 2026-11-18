package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;

/**
 * RHO-LOSS (Mindermann et al., "Prioritized Training on Points that are Learnable, Worth
 * Learning, and Not Yet Learnt", ICML 2022, <a href="https://arxiv.org/abs/2206.07137">arXiv:2206.07137</a>).
 * From each large candidate batch, train on the points with the highest reducible holdout loss:
 * the current training loss minus the "irreducible" loss of a small model trained on a holdout
 * set. High training loss alone selects noisy and unlearnable points; subtracting the
 * irreducible loss filters them (their irreducible loss is high too), while points already
 * learnt (low training loss) are still skipped.
 *
 * <p>The irreducible losses are computed once, for every training point, with the holdout
 * model, and passed to the constructor.
 */
public final class ReducibleLossSelector {

    private final double[] irreducible;

    /**
     * Creates the selector.
     *
     * @param irreducible the holdout model's loss on every training point
     */
    public ReducibleLossSelector(double[] irreducible) {
        this.irreducible = irreducible.clone();
    }

    /**
     * Selects the top {@code k} candidates by reducible loss.
     *
     * @param candidates training-point indices
     * @param currentLoss the current model's loss on each candidate, aligned with {@code candidates}
     * @param k how many to keep
     * @return the selected training-point indices
     */
    public int[] select(int[] candidates, double[] currentLoss, int k) {
        double[] score = new double[candidates.length];
        for (int i = 0; i < score.length; i++) {
            score[i] = currentLoss[i] - irreducible[candidates[i]];
        }
        return top(candidates, score, k);
    }

    /**
     * The highest-loss baseline: the top {@code k} candidates by current loss.
     *
     * @param candidates training-point indices
     * @param currentLoss the current loss of each candidate
     * @param k how many to keep
     * @return the selected indices
     */
    public static int[] highestLoss(int[] candidates, double[] currentLoss, int k) {
        return top(candidates, currentLoss, k);
    }

    /**
     * The reducible loss of one training point.
     *
     * @param index the training point
     * @param currentLoss its current loss
     * @return {@code currentLoss - irreducible[index]}
     */
    public double reducible(int index, double currentLoss) {
        return currentLoss - irreducible[index];
    }

    private static int[] top(int[] candidates, double[] score, int k) {
        Integer[] order = new Integer[candidates.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        int[] out = new int[Math.min(k, candidates.length)];
        for (int i = 0; i < out.length; i++) {
            out[i] = candidates[order[i]];
        }
        return out;
    }
}
