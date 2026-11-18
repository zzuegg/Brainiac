package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Weight sharing, the second stage of Deep Compression (Han et al. 2016,
 * <a href="https://arxiv.org/abs/1510.00149">arXiv:1510.00149</a>): the non-zero weights of a
 * layer are clustered by one-dimensional k-means ({@link KMeans}, from a linear initialisation
 * over the weight range, which the paper found best because it keeps the rare large weights) into
 * {@code 2^bits} shared values; each weight then stores only its cluster index. The centroids are
 * fine-tuned: {@link Shared#weight()} gathers them into the weight matrix, so the gradient of
 * every centroid is the sum of its weights' gradients. Pruned (zero) weights stay zero.
 *
 * <p>{@link Shared#storageBits} counts what the layer costs to store: the codebook, the indices,
 * and for a sparse layer the relative positions as fixed-width gaps (a filler entry is inserted
 * when a gap overflows), each stream optionally Huffman-coded ({@link HuffmanCoder}).
 */
public final class WeightSharingQuantizer {

    private WeightSharingQuantizer() {
    }

    /** A layer whose weights are indices into a trainable codebook. */
    public static final class Shared {
        private final Var centroids;
        private final Tensor indices;
        private final Tensor mask;
        private final Shape shape;
        private final int[] index;
        private final boolean[] nonZero;

        Shared(double[] centroids, int[] index, boolean[] nonZero, Shape shape) {
            this.centroids = Var.param(Tensor.of(Shape.of(centroids.length), WoodFisherPruning.toFloat(centroids)));
            float[] ids = new float[index.length];
            float[] m = new float[index.length];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = index[i];
                m[i] = nonZero[i] ? 1 : 0;
            }
            this.indices = Tensor.of(Shape.of(ids.length), ids);
            this.mask = Tensor.of(shape, m);
            this.shape = shape;
            this.index = index;
            this.nonZero = nonZero;
        }

        /** @return the trainable shared values {@code [2^bits]} */
        public Var centroids() {
            return centroids;
        }

        /** @return the weight matrix built from the codebook, differentiable in the centroids */
        public Var weight() {
            return centroids.gather(0, indices).reshape(shape.toArray()).mul(mask);
        }

        /** @return the current dequantised weight */
        public Tensor dequantized() {
            return weight().value();
        }

        /**
         * The storage cost.
         *
         * @param huffman whether the index and gap streams are Huffman-coded
         * @param gapBits the width of a relative position for sparse layers, for example 4;
         *     0 to store a dense layer (an index for every weight)
         * @return the bits, including the 32-bit codebook entries and any code tables
         */
        public long storageBits(boolean huffman, int gapBits) {
            int k = (int) centroids.shape().size();
            int bits = 32 - Integer.numberOfLeadingZeros(Math.max(1, k - 1));
            List<Integer> idx = new ArrayList<>();
            List<Integer> gaps = new ArrayList<>();
            if (gapBits == 0) {
                for (int i : index) {
                    idx.add(i);
                }
            } else {
                int maxGap = (1 << gapBits) - 1;
                int last = -1;
                for (int i = 0; i < index.length; i++) {
                    if (!nonZero[i]) {
                        continue;
                    }
                    int gap = i - last - 1;
                    while (gap > maxGap) {   // filler entry: a zero weight at the farthest gap
                        idx.add(k);
                        gaps.add(maxGap);
                        gap -= maxGap + 1;
                    }
                    idx.add(index[i]);
                    gaps.add(gap);
                    last = i;
                }
            }
            int alphabet = gapBits == 0 ? k : k + 1;
            int idxBits = 32 - Integer.numberOfLeadingZeros(Math.max(1, alphabet - 1));
            long total = 32L * k;
            int[] is = idx.stream().mapToInt(Integer::intValue).toArray();
            int[] gs = gaps.stream().mapToInt(Integer::intValue).toArray();
            if (huffman) {
                HuffmanCoder hi = HuffmanCoder.fit(is, alphabet);
                total += hi.encodedBits(is) + hi.tableBits();
                if (gs.length > 0) {
                    HuffmanCoder hg = HuffmanCoder.fit(gs, 1 << gapBits);
                    total += hg.encodedBits(gs) + hg.tableBits();
                }
            } else {
                total += (long) is.length * (gapBits == 0 ? bits : idxBits) + (long) gs.length * gapBits;
            }
            return total;
        }

        /** @return the cluster index of every weight (meaningless where the weight is zero) */
        public int[] indexArray() {
            return index.clone();
        }
    }

    /**
     * Clusters a layer's non-zero weights.
     *
     * @param weight the weight
     * @param bits {@code log2} of the number of shared values
     * @param iterations Lloyd iterations
     * @return the shared layer
     */
    public static Shared fit(Tensor weight, int bits, int iterations) {
        float[] w = weight.toArray();
        boolean[] nonZero = new boolean[w.length];
        List<double[]> pts = new ArrayList<>();
        double lo = Double.MAX_VALUE;
        double hi = -Double.MAX_VALUE;
        for (int i = 0; i < w.length; i++) {
            nonZero[i] = w[i] != 0;
            if (nonZero[i]) {
                pts.add(new double[] {w[i]});
                lo = Math.min(lo, w[i]);
                hi = Math.max(hi, w[i]);
            }
        }
        int k = 1 << bits;
        double[][] start = new double[k][1];
        for (int c = 0; c < k; c++) {
            start[c][0] = lo + (hi - lo) * (c + 0.5) / k;   // linear initialisation
        }
        KMeans km = KMeans.lloyd(pts.toArray(double[][]::new), start, iterations);
        double[] centroids = new double[k];
        for (int c = 0; c < k; c++) {
            centroids[c] = km.centres()[c][0];
        }
        int[] index = new int[w.length];
        int p = 0;
        for (int i = 0; i < w.length; i++) {
            if (nonZero[i]) {
                index[i] = km.labels()[p++];
            }
        }
        return new Shared(centroids, index, nonZero, weight.shape());
    }
}
