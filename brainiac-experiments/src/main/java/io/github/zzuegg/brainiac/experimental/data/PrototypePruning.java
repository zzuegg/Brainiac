package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.ml.KMeans;

import java.util.Arrays;

/**
 * Self-supervised prototype pruning (Sorscher et al., "Beyond neural scaling laws: beating
 * power law scaling via data pruning", NeurIPS 2022, <a href="https://arxiv.org/abs/2206.14486">arXiv:2206.14486</a>).
 * Embeddings from any (self-supervised, label-free) encoder are clustered by
 * {@link KMeans}; an example's difficulty is its distance to its cluster's prototype. Their
 * theory predicts that with little data one should keep the easy (prototypical) examples and
 * with much data the hard ones, so the best pruning direction flips with dataset size.
 */
public final class PrototypePruning {

    private final double[] distance;

    private PrototypePruning(double[] distance) {
        this.distance = distance;
    }

    /**
     * Scores examples by their distance to the nearest k-means prototype.
     *
     * @param embeddings the examples' embeddings
     * @param clusters the number of prototypes
     * @param seed the seed of k-means
     * @return the scores
     */
    public static PrototypePruning fit(double[][] embeddings, int clusters, long seed) {
        double[][] x = normalize(embeddings);
        KMeans km = KMeans.fit(x, clusters, KMeans.Init.PLUS_PLUS, 2, 50, seed);
        double[] d = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] c = km.centres()[km.labels()[i]];
            double s = 0;
            for (int j = 0; j < c.length; j++) {
                double e = x[i][j] - c[j];
                s += e * e;
            }
            d[i] = Math.sqrt(s);
        }
        return new PrototypePruning(d);
    }

    /** @return each example's distance to its prototype (its difficulty) */
    public double[] difficulty() {
        return distance.clone();
    }

    /**
     * The examples to keep.
     *
     * @param fraction the share kept
     * @param keepHard keep the farthest (hardest) examples if true, the closest otherwise
     * @return the kept indices
     */
    public int[] keep(double fraction, boolean keepHard) {
        Integer[] idx = new Integer[distance.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> keepHard ? Double.compare(distance[b], distance[a])
                : Double.compare(distance[a], distance[b]));
        int m = (int) Math.round(fraction * distance.length);
        int[] out = new int[m];
        for (int i = 0; i < m; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    /** Unit-normalises rows (cosine geometry, as for self-supervised embeddings). */
    private static double[][] normalize(double[][] e) {
        double[][] out = new double[e.length][];
        for (int i = 0; i < e.length; i++) {
            double s = 0;
            for (double v : e[i]) {
                s += v * v;
            }
            double inv = 1 / Math.max(Math.sqrt(s), 1e-12);
            out[i] = new double[e[i].length];
            for (int j = 0; j < e[i].length; j++) {
                out[i][j] = e[i][j] * inv;
            }
        }
        return out;
    }
}
