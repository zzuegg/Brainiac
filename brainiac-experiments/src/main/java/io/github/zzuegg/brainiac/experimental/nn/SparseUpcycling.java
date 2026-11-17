package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Sparse upcycling: starts a mixture of experts from a trained dense feed-forward network
 * instead of from scratch. Every expert is a copy of the dense FFN and the router starts
 * near zero, so the new MoE computes almost exactly what the dense model did and continued
 * training only has to learn to specialize, which is far cheaper than training the MoE from
 * scratch.
 *
 * <p><b>Drop-upcycling</b> fixes the weakness that identical experts specialize slowly: in
 * every expert a different random share {@code dropRatio} of the hidden units is
 * re-initialized (their input and output weights redrawn from a normal distribution with the
 * mean and standard deviation of the dense weights), which breaks the symmetry while
 * keeping most of the dense knowledge.
 *
 * <p>The MoE gates are the router's softmax probabilities without renormalization over the
 * chosen experts ({@link FineGrainedMoe}), so a near-uniform router would scale the output by
 * {@code topK / experts}; the copied output weights are scaled by {@code experts / topK} to
 * cancel that, and the upcycled MoE starts as a function-preserving copy of the dense FFN.
 *
 * <p>Papers: Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense
 * Checkpoints", <a href="https://arxiv.org/abs/2212.05055">arXiv:2212.05055</a>; Nakamura et
 * al., "Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization",
 * <a href="https://arxiv.org/abs/2502.19261">arXiv:2502.19261</a>.
 */
public final class SparseUpcycling {

    private SparseUpcycling() {
    }

    /**
     * Copies a dense FFN into experts, re-initializing a random share of each copy's hidden
     * units.
     *
     * @param dense the trained dense FFN
     * @param experts the number of copies
     * @param dropRatio the share of hidden units to re-initialize per expert, 0 for plain
     *     upcycling
     * @param outputScale a factor for the copied output weights
     * @param random the random source
     * @return the experts
     */
    public static List<SiluMlpExpert> copies(SiluMlpExpert dense, int experts, float dropRatio, float outputScale,
            RandomGenerator random) {
        Tensor w1 = dense.w1().value();
        Tensor w2 = dense.w2().value();
        int in = w1.shape().dim(0);
        int hidden = w1.shape().dim(1);
        int out = w2.shape().dim(1);
        float[] a = w1.toArray();
        float[] bias = dense.b1().value().toArray();
        float[] c = w2.toArray();
        double[] s1 = stats(a);
        double[] s2 = stats(c);
        int drop = Math.round(dropRatio * hidden);
        List<SiluMlpExpert> list = new ArrayList<>();
        for (int e = 0; e < experts; e++) {
            float[] na = a.clone();
            float[] nb = bias.clone();
            float[] nc = c.clone();
            int[] order = permutation(hidden, random);
            for (int k = 0; k < drop; k++) {
                int u = order[k];
                for (int i = 0; i < in; i++) {
                    na[i * hidden + u] = (float) (s1[0] + s1[1] * random.nextGaussian());
                }
                nb[u] = 0;
                for (int o = 0; o < out; o++) {
                    nc[u * out + o] = (float) (s2[0] + s2[1] * random.nextGaussian());
                }
            }
            for (int i = 0; i < nc.length; i++) {
                nc[i] *= outputScale;
            }
            list.add(new SiluMlpExpert(Tensor.of(Shape.of(in, hidden), na), Tensor.of(Shape.of(hidden), nb),
                    Tensor.of(Shape.of(hidden, out), nc)));
        }
        return list;
    }

    /**
     * Upcycles a dense FFN into a {@link FineGrainedMoe} with top-{@code k} routing and no
     * shared experts; the router starts at small random values.
     *
     * @param dense the trained dense FFN
     * @param experts the number of experts
     * @param topK experts per token
     * @param dropRatio the share of hidden units to re-initialize per expert (0 for plain
     *     upcycling, 0.5 in the drop-upcycling paper)
     * @param random the random source
     * @return the MoE, initially close to the dense FFN
     */
    public static FineGrainedMoe upcycle(SiluMlpExpert dense, int experts, int topK, float dropRatio,
            RandomGenerator random) {
        int features = dense.w1().shape().dim(0);
        List<Layer> list = new ArrayList<>(copies(dense, experts, dropRatio, experts / (float) topK, random));
        FineGrainedMoe moe = new FineGrainedMoe(features, dense.w2().shape().dim(1), topK, list, List.of(), random);
        moe.router().assign(Tensor.normal(moe.router().shape(), 0, 1e-3f, random));
        return moe;
    }

    private static double[] stats(float[] w) {
        double mean = 0;
        for (float v : w) {
            mean += v;
        }
        mean /= w.length;
        double var = 0;
        for (float v : w) {
            var += (v - mean) * (v - mean);
        }
        return new double[] {mean, Math.sqrt(var / w.length)};
    }

    private static int[] permutation(int n, RandomGenerator random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
