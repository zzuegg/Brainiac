package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Pruning at initialisation: scoring weights before any training and pruning once, so the sparse
 * network is trained from the start (with {@link FixedSparsityOptimizer}). Scores are ranked
 * globally across all matrices:
 * <ul>
 * <li>{@link #snip}: SNIP (Lee et al. 2019, <a href="https://arxiv.org/abs/1810.02340">arXiv:1810.02340</a>),
 *     connection sensitivity {@code |w * dL/dw|} on one batch;</li>
 * <li>{@link #grasp}: GraSP (Wang et al. 2020, <a href="https://arxiv.org/abs/2002.07376">arXiv:2002.07376</a>),
 *     {@code S = -w * (H g)}; removing the weights with the largest {@code S} least reduces the
 *     gradient flow {@code g^T g};</li>
 * <li>{@link #synflow}: SynFlow (Tanaka et al. 2020, <a href="https://arxiv.org/abs/2006.05467">arXiv:2006.05467</a>),
 *     data-free: the synaptic-flow score {@code |w * dR/dw|} with {@code R = 1^T (prod |W_l|) 1},
 *     pruned iteratively on an exponential schedule, which avoids layer collapse at extreme
 *     sparsity;</li>
 * <li>{@link #random} (uniform per matrix) and {@link #randomLayerwise} (random within each
 *     matrix but with another method's per-matrix survival): the controls of Frankle et al. 2021
 *     (<a href="https://arxiv.org/abs/2009.08576">arXiv:2009.08576</a>), who found the methods
 *     barely beat random pruning with matched layer-wise ratios.</li>
 * </ul>
 * Every method returns 0/1 masks; {@link FixedSparsityOptimizer#apply} prunes the weights.
 * {@code ZeroCostProxies} computes SNIP and SynFlow scores to rank architectures; here they prune.
 */
public final class PruningAtInit {

    private PruningAtInit() {
    }

    /**
     * SNIP.
     *
     * @param weights the matrices
     * @param loss the loss of one batch at the current weights
     * @param sparsity the share to remove over all matrices
     * @return the masks
     */
    public static Map<Var, Tensor> snip(List<Var> weights, Supplier<Var> loss, double sparsity) {
        List<Var> g = Var.gradients(loss.get(), weights, false);
        List<float[]> scores = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            float[] w = weights.get(i).value().toArray();
            float[] gv = g.get(i).value().toArray();
            float[] s = new float[w.length];
            for (int j = 0; j < s.length; j++) {
                s[j] = Math.abs(w[j] * gv[j]);
            }
            scores.add(s);
        }
        return keepHighest(weights, scores, sparsity);
    }

    /**
     * GraSP.
     *
     * @param weights the matrices
     * @param loss the loss of one batch; called twice
     * @param sparsity the share to remove
     * @return the masks
     */
    public static Map<Var, Tensor> grasp(List<Var> weights, Supplier<Var> loss, double sparsity) {
        double[] g = CurvatureProducts.gradient(loss.get(), weights);
        double[] hg = CurvatureProducts.hessianVector(loss, weights, g);
        List<float[]> scores = new ArrayList<>();
        int o = 0;
        for (Var wv : weights) {
            float[] w = wv.value().toArray();
            float[] s = new float[w.length];
            for (int j = 0; j < s.length; j++) {
                s[j] = (float) (w[j] * hg[o + j]);   // = -S: keep the largest -S, remove the largest S
            }
            o += w.length;
            scores.add(s);
        }
        return keepHighest(weights, scores, sparsity);
    }

    /**
     * SynFlow for a chain of matrices {@code x W_0 W_1 ...} (an MLP; ReLU passes the positive
     * flow unchanged, biases are left out).
     *
     * @param weights the chain's matrices, input first
     * @param sparsity the share to remove
     * @param iterations pruning rounds of the exponential schedule, for example 100
     * @return the masks
     */
    public static Map<Var, Tensor> synflow(List<Var> weights, double sparsity, int iterations) {
        Map<Var, Tensor> masks = new IdentityHashMap<>();
        List<Var> abs = new ArrayList<>();
        for (Var w : weights) {
            masks.put(w, Tensor.ones(w.shape()));
            abs.add(Var.param(w.value().abs()));
        }
        for (int it = 1; it <= iterations; it++) {
            double keep = Math.pow(1 - sparsity, it / (double) iterations);
            for (int i = 0; i < weights.size(); i++) {
                abs.get(i).assign(weights.get(i).value().abs().mul(masks.get(weights.get(i))));
            }
            // scale-free flow: the score is invariant to rescaling, so normalise to avoid overflow
            Var h = Var.of(Tensor.ones(Shape.of(1, weights.getFirst().shape().dim(0))));
            for (Var a : abs) {
                h = h.matmul(a);
                h = h.div(Var.of(h.value().sum().add(1e-30f)));
            }
            List<Var> g = Var.gradients(h.sum(), abs, false);
            List<float[]> scores = new ArrayList<>();
            for (int i = 0; i < abs.size(); i++) {
                float[] w = abs.get(i).value().toArray();
                float[] gv = g.get(i).value().toArray();
                float[] s = new float[w.length];
                for (int j = 0; j < s.length; j++) {
                    s[j] = Math.abs(w[j] * gv[j]);
                }
                scores.add(s);
            }
            Map<Var, Tensor> next = keepHighest(weights, scores, 1 - keep);
            masks.clear();
            masks.putAll(next);
        }
        return masks;
    }

    /**
     * Uniform random pruning, the same share in every matrix.
     *
     * @param weights the matrices
     * @param sparsity the share to remove per matrix
     * @param seed the seed
     * @return the masks
     */
    public static Map<Var, Tensor> random(List<Var> weights, double sparsity, long seed) {
        Map<Var, Double> share = new LinkedHashMap<>();
        for (Var w : weights) {
            share.put(w, 1 - sparsity);
        }
        return randomWith(weights, share, seed);
    }

    /**
     * Random pruning with another method's per-matrix survival ("layer-wise ratios").
     *
     * @param weights the matrices
     * @param reference the masks whose per-matrix kept share to copy
     * @param seed the seed
     * @return the masks
     */
    public static Map<Var, Tensor> randomLayerwise(List<Var> weights, Map<Var, Tensor> reference, long seed) {
        Map<Var, Double> share = new LinkedHashMap<>();
        for (Var w : weights) {
            share.put(w, reference.get(w).sum().item() / (double) w.shape().size());
        }
        return randomWith(weights, share, seed);
    }

    /**
     * @param weights the matrices
     * @param masks their masks
     * @return the kept share of each matrix, in order
     */
    public static double[] survival(List<Var> weights, Map<Var, Tensor> masks) {
        double[] s = new double[weights.size()];
        for (int i = 0; i < s.length; i++) {
            s[i] = masks.get(weights.get(i)).sum().item() / (double) weights.get(i).shape().size();
        }
        return s;
    }

    private static Map<Var, Tensor> randomWith(List<Var> weights, Map<Var, Double> share, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<Var, Tensor> masks = new IdentityHashMap<>();
        for (Var w : weights) {
            int n = (int) w.shape().size();
            int keep = (int) Math.round(share.get(w) * n);
            int[] perm = new int[n];
            for (int i = 0; i < n; i++) {
                perm[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
            float[] m = new float[n];
            for (int i = 0; i < keep; i++) {
                m[perm[i]] = 1;
            }
            masks.put(w, Tensor.of(w.shape(), m));
        }
        return masks;
    }

    /** Keeps the highest scores globally. */
    static Map<Var, Tensor> keepHighest(List<Var> weights, List<float[]> scores, double sparsity) {
        int total = 0;
        for (float[] s : scores) {
            total += s.length;
        }
        float[] all = new float[total];
        int o = 0;
        for (float[] s : scores) {
            System.arraycopy(s, 0, all, o, s.length);
            o += s.length;
        }
        int keep = Math.max(1, (int) Math.round((1 - sparsity) * total));
        float[] sorted = all.clone();
        Arrays.sort(sorted);
        float cut = sorted[total - keep];
        int above = 0;
        for (float v : all) {
            above += v > cut ? 1 : 0;
        }
        int ties = keep - above;   // keep exactly `keep`: ties at the cut go in order
        Map<Var, Tensor> masks = new IdentityHashMap<>();
        for (int i = 0; i < weights.size(); i++) {
            float[] s = scores.get(i);
            float[] m = new float[s.length];
            for (int j = 0; j < s.length; j++) {
                if (s[j] > cut) {
                    m[j] = 1;
                } else if (s[j] == cut && ties > 0) {
                    m[j] = 1;
                    ties--;
                }
            }
            masks.put(weights.get(i), Tensor.of(weights.get(i).shape(), m));
        }
        return masks;
    }
}
