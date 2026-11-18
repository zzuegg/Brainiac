package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Movement pruning (Sanh, Wolf and Rush 2020, <a href="https://arxiv.org/abs/2005.07683">arXiv:2005.07683</a>):
 * during fine-tuning a weight's importance is how far it moves away from zero, not how large it
 * is. Magnitude pruning keeps weights that were large after pretraining even while fine-tuning is
 * shrinking them; movement keeps what the new task needs.
 *
 * <p>The layer uses {@code W * M} with a mask {@code M} derived from scores {@code S}; the mask
 * is passed straight through, so {@code dL/dS = dL/dW' * W} with {@code W' = W * M}:
 * <ul>
 * <li>{@linkplain #hard hard}: {@code S = -sum_t W * dL/dW'} accumulated over all steps, and
 *     {@code M = TopK(S)} per matrix at a sparsity that rises on {@link GradualPruningSchedule#cubic}'s
 *     cubic schedule; a pruned weight is frozen but can return when its score rises;</li>
 * <li>{@linkplain #soft soft}: learned scores with {@code M = 1[S > tau]}, trained by the same
 *     straight-through gradient (normalised by its root mean square per matrix, standing in
 *     for the paper's Adam on the scores) plus an L1 penalty {@code lambda * sum sigmoid(S)},
 *     so the sparsity is whatever the penalty buys.</li>
 * </ul>
 * Use {@link #masked} for every pruned weight in the forward pass and this object as the
 * optimizer; {@link #finish()} writes the masks into the weights at the end.
 */
public final class MovementPruning implements Optimizer {

    private final Optimizer inner;
    private final List<Var> weights;
    private final boolean soft;
    private final double target;
    private final int start;
    private final int end;
    private final int frequency;
    private final double threshold;
    private final double lambda;
    private final double scoreRate;
    private final Map<Var, double[]> scores = new IdentityHashMap<>();
    private final Map<Var, Tensor> masks = new IdentityHashMap<>();
    private int step;

    private MovementPruning(Optimizer inner, List<Var> weights, boolean soft, double target, int start, int end,
            int frequency, double threshold, double lambda, double scoreRate) {
        this.inner = inner;
        this.weights = List.copyOf(weights);
        this.soft = soft;
        this.target = target;
        this.start = start;
        this.end = end;
        this.frequency = Math.max(1, frequency);
        this.threshold = threshold;
        this.lambda = lambda;
        this.scoreRate = scoreRate;
        for (Var w : weights) {
            double[] s = new double[(int) w.shape().size()];
            if (soft) {
                Arrays.fill(s, threshold + 1);   // everything starts kept, one unit above the threshold
            }
            scores.put(w, s);
            masks.put(w, Tensor.ones(w.shape()));
        }
    }

    /**
     * Hard movement pruning to a target sparsity on a cubic schedule.
     *
     * @param inner the optimizer of the weights
     * @param weights the matrices to prune
     * @param finalSparsity the target sparsity per matrix
     * @param start the first pruning step
     * @param end the step reaching the target
     * @param frequency steps between mask updates
     * @return the pruner
     */
    public static MovementPruning hard(Optimizer inner, List<Var> weights, double finalSparsity, int start, int end,
            int frequency) {
        return new MovementPruning(inner, weights, false, finalSparsity, start, end, frequency, 0, 0, 0);
    }

    /**
     * Soft movement pruning with learned scores.
     *
     * @param inner the optimizer of the weights
     * @param weights the matrices to prune
     * @param threshold {@code tau}; a weight is kept while its score exceeds it
     * @param lambda the L1 penalty on {@code sigmoid(S)}; larger gives more sparsity
     * @param scoreRate the SGD step size of the scores
     * @return the pruner
     */
    public static MovementPruning soft(Optimizer inner, List<Var> weights, double threshold, double lambda,
            double scoreRate) {
        return new MovementPruning(inner, weights, true, 0, 0, 0, 1, threshold, lambda, scoreRate);
    }

    /**
     * The weight a layer should use: {@code W * M} forward, with the full gradient passed to
     * {@code W} (the straight-through mask) so the scores see it.
     *
     * @param w a pruned weight
     * @return the masked weight
     */
    public Var masked(Var w) {
        Tensor m = masks.get(w);
        return m == null ? w : Functional.straightThrough(w.mul(m), w);
    }

    /**
     * @param w a weight
     * @return its 0/1 mask
     */
    public Tensor mask(Var w) {
        return masks.get(w);
    }

    /** @return the share of masked weights over all pruned matrices */
    public double sparsity() {
        double zero = 0;
        double total = 0;
        for (Var w : weights) {
            total += w.shape().size();
            zero += w.shape().size() - masks.get(w).sum().item();
        }
        return zero / total;
    }

    /** Multiplies the weights by their masks, giving a plainly sparse network. */
    public void finish() {
        for (Var w : weights) {
            w.assign(w.value().mul(masks.get(w)));
        }
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public void step(List<Var> parameters) {
        step++;
        for (Var w : weights) {
            Tensor g = w.grad();
            if (g == null) {
                continue;
            }
            float[] gv = g.toArray();
            float[] wv = w.value().toArray();
            double[] s = scores.get(w);
            double rms = 0;
            if (soft) {
                for (int i = 0; i < s.length; i++) {
                    rms += gv[i] * (double) wv[i] * gv[i] * wv[i];
                }
                rms = Math.sqrt(rms / s.length) + 1e-30;
            }
            for (int i = 0; i < s.length; i++) {
                double move = gv[i] * (double) wv[i];   // dL/dS
                if (soft) {
                    double sig = 1 / (1 + Math.exp(-s[i]));
                    s[i] -= scoreRate * (move / rms + lambda * sig * (1 - sig));
                } else {
                    s[i] -= move;
                }
            }
            w.setGrad(g.mul(masks.get(w)));
        }
        inner.step(parameters);
        boolean update = soft || step >= start && step <= end && ((step - start) % frequency == 0 || step == end);
        if (update) {
            double sparsity = soft ? 0 : GradualPruningSchedule.cubic(0, target, start, end, step);
            for (Var w : weights) {
                masks.put(w, soft ? threshold(scores.get(w), w.shape()) : topK(scores.get(w), w.shape(), sparsity));
            }
        }
    }

    private Tensor threshold(double[] s, Shape shape) {
        float[] m = new float[s.length];
        for (int i = 0; i < s.length; i++) {
            m[i] = s[i] > threshold ? 1 : 0;
        }
        return Tensor.of(shape, m);
    }

    private static Tensor topK(double[] s, Shape shape, double sparsity) {
        int drop = (int) Math.round(sparsity * s.length);
        Integer[] idx = new Integer[s.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(s[a], s[b]));
        float[] m = new float[s.length];
        Arrays.fill(m, 1);
        for (int k = 0; k < drop; k++) {
            m[idx[k]] = 0;
        }
        return Tensor.of(shape, m);
    }
}
