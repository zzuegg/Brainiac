package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.SelectiveAttention;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Selective attention as a context pruner, on {@link MixerShootout#recentRecall recent
 * recall}: keys are re-bound several times and only the latest binding counts, so most of
 * the context becomes obsolete as the sequence goes on, the situation selective attention
 * was designed for. A model trained with {@link SelectiveAttention} drops every key whose
 * accumulated selection penalty has reached a threshold {@code tau} (its softmax weight is
 * already scaled down by at least {@code exp(-tau)}). The baseline is a plain softmax model
 * of the same size, evaluated with full attention and with H2O-style eviction at the same
 * per-query budget (half of the budget for the most recent keys, half for the "heavy
 * hitters" with the most attention accumulated by earlier queries).
 *
 * <p>H2O is evaluated as a per-query key mask computed from the model's own attention (an
 * idealized, non-greedy version of the cache policy).
 *
 * <p>Papers: Leviathan et al., "Selective Attention Improves Transformer",
 * <a href="https://arxiv.org/abs/2410.02703">arXiv:2410.02703</a>; Zhang et al., "H2O:
 * Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models",
 * <a href="https://arxiv.org/abs/2306.14048">arXiv:2306.14048</a>.
 */
public final class SelectiveContext {

    private SelectiveContext() {
    }

    /**
     * The comparison, as answer accuracies.
     *
     * @param full the softmax model with full attention
     * @param h2o the softmax model with H2O eviction at the kept share
     * @param selectiveFull the selective model without pruning
     * @param selectivePruned the selective model with pruning
     * @param keptShare the kept share of the causal context (both pruned rows)
     */
    public record Result(double full, double h2o, double selectiveFull, double selectivePruned, double keptShare) {
    }

    /**
     * Trains both models on recent recall (4 keys, 8 values, 8 bindings) and evaluates them.
     *
     * @param depth the number of blocks
     * @param steps training steps
     * @param tau the pruning threshold on the penalty, for example 2
     * @param seed the seed
     * @return the result
     */
    public static Result run(int depth, int steps, float tau, long seed) {
        MixerShootout.Task task = MixerShootout.recentRecall(4, 8, 8);
        RefinedAttention.Spec base = RefinedAttention.of(2).causal(true);
        ArchitectureBench.TinyModel full = ArchitectureBench.trainRetrieval(ArchitectureBench.stack(base, depth), 16,
                task, steps, 0.01f, seed, true);
        ArchitectureBench.TinyModel selective = ArchitectureBench.trainRetrieval(ArchitectureBench.stack(
                base.selective(true), depth), 16, task, steps, 0.01f, seed, true);
        int[][][] test = task.batch(new SplittableRandom(seed + 1000), 256);
        // selective: one pass for the penalties, then prune keys at or above the threshold
        List<RefinedAttention> sel = ArchitectureBench.attentions(selective.body());
        double selectiveFull = ArchitectureBench.accuracy(selective, test);
        Tensor[] penalties = new Tensor[sel.size()];
        double kept = 0;
        for (int l = 0; l < sel.size(); l++) {
            penalties[l] = sel.get(l).lastPenalty();
            kept += SelectiveAttention.keptContext(penalties[l], tau);
        }
        kept /= sel.size();
        for (int l = 0; l < sel.size(); l++) {
            sel.get(l).restrict(penalties[l].mul(-1f).greater(Tensor.full(Shape.of(), -tau)));
        }
        double pruned = ArchitectureBench.accuracy(selective, test);
        sel.forEach(a -> a.restrict(null));
        // H2O on the softmax model at the same kept share
        List<RefinedAttention> att = ArchitectureBench.attentions(full.body());
        double fullAcc = ArchitectureBench.accuracy(full, test);
        Tensor[] masks = new Tensor[att.size()];
        for (int l = 0; l < att.size(); l++) {
            masks[l] = heavyHitters(att.get(l).lastWeights(), kept);
        }
        for (int l = 0; l < att.size(); l++) {
            att.get(l).restrict(masks[l]);
        }
        double h2o = ArchitectureBench.accuracy(full, test);
        att.forEach(a -> a.restrict(null));
        return new Result(fullAcc, h2o, selectiveFull, pruned, kept);
    }

    /**
     * An H2O key mask: query {@code i} keeps {@code ceil(share (i + 1))} keys, half of them the
     * most recent (itself included) and the rest the keys with the most attention accumulated
     * by queries before {@code i} (summed over heads).
     *
     * @param weights attention weights {@code [batch, heads, t, t]}
     * @param share the kept share
     * @return a 0/1 mask {@code [batch, 1, t, t]}
     */
    static Tensor heavyHitters(Tensor weights, double share) {
        int b = weights.shape().dim(0);
        int h = weights.shape().dim(1);
        int t = weights.shape().dim(2);
        float[] w = weights.toArray();
        float[] mask = new float[b * t * t];
        for (int n = 0; n < b; n++) {
            double[] acc = new double[t];
            for (int i = 0; i < t; i++) {
                int budget = Math.max(1, (int) Math.ceil(share * (i + 1)));
                int recent = Math.max(1, budget / 2);
                for (int j = Math.max(0, i - recent + 1); j <= i; j++) {
                    mask[(n * t + i) * t + j] = 1;
                }
                Integer[] order = new Integer[Math.max(0, i - recent + 1)];
                for (int j = 0; j < order.length; j++) {
                    order[j] = j;
                }
                Arrays.sort(order, (x, y) -> Double.compare(acc[y], acc[x]));
                for (int k = 0; k < Math.min(budget - recent, order.length); k++) {
                    mask[(n * t + i) * t + order[k]] = 1;
                }
                for (int hh = 0; hh < h; hh++) {
                    for (int j = 0; j <= i; j++) {
                        acc[j] += w[((n * h + hh) * t + i) * t + j];
                    }
                }
            }
        }
        return Tensor.of(Shape.of(b, 1, t, t), mask);
    }

    /**
     * Runs the comparison with one block and threshold 2.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Result r = run(1, 400, 2f, 1);
        System.out.printf("kept %.2f of the context: selective %.3f (unpruned %.3f), H2O %.3f, full softmax %.3f%n",
                r.keptShare(), r.selectivePruned(), r.selectiveFull(), r.h2o(), r.full());
    }
}
