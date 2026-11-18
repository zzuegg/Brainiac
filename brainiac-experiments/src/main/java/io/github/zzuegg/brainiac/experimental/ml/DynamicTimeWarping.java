package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * Dynamic time warping: the cheapest monotone alignment of two feature sequences, by the
 * recursion {@code D(i, j) = d(a_i, b_j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1))} with a
 * Euclidean local cost, restricted to a Sakoe-Chiba band around the (length-normalised)
 * diagonal and normalised by {@code |a| + |b|}. {@link TemplateClassifier} is the classic
 * isolated-word recogniser built on it: store a few template utterances per word and
 * label a query by its nearest template. It needs no training, which makes it the floor
 * learned recognisers must beat when there are only a handful of examples.
 *
 * <p>Paper: Sakoe and Chiba, "Dynamic programming algorithm optimization for spoken word
 * recognition" (IEEE TASSP 1978), <a href="https://ieeexplore.ieee.org/document/1163055">doi</a>.
 */
public final class DynamicTimeWarping {

    private DynamicTimeWarping() {
    }

    /**
     * The normalised DTW distance.
     *
     * @param a a sequence {@code [frames][features]}
     * @param b another sequence with the same feature count
     * @param band the Sakoe-Chiba half-width in frames around the scaled diagonal; negative for none
     * @return the alignment cost divided by {@code |a| + |b|}
     */
    public static double distance(float[][] a, float[][] b, int band) {
        int n = a.length;
        int m = b.length;
        double[] prev = new double[m + 1];
        double[] cur = new double[m + 1];
        java.util.Arrays.fill(prev, Double.POSITIVE_INFINITY);
        prev[0] = 0;
        for (int i = 1; i <= n; i++) {
            java.util.Arrays.fill(cur, Double.POSITIVE_INFINITY);
            int centre = (int) Math.round((double) i * m / n);
            int lo = band < 0 ? 1 : Math.max(1, centre - band);
            int hi = band < 0 ? m : Math.min(m, centre + band);
            for (int j = lo; j <= hi; j++) {
                double d = 0;
                for (int k = 0; k < a[i - 1].length; k++) {
                    double diff = a[i - 1][k] - b[j - 1][k];
                    d += diff * diff;
                }
                double best = Math.min(prev[j], Math.min(cur[j - 1], prev[j - 1]));
                cur[j] = Math.sqrt(d) + best;
            }
            double[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[m] / (n + m);
    }

    /** A nearest-template classifier. */
    public static final class TemplateClassifier {

        private final List<float[][]> templates = new ArrayList<>();
        private final List<Integer> labels = new ArrayList<>();
        private final int band;

        /**
         * Creates an empty classifier.
         *
         * @param band the Sakoe-Chiba half-width in frames
         */
        public TemplateClassifier(int band) {
            this.band = band;
        }

        /**
         * Stores a template.
         *
         * @param label its class
         * @param features its feature sequence
         */
        public void add(int label, float[][] features) {
            templates.add(features);
            labels.add(label);
        }

        /**
         * Labels a query by its nearest template.
         *
         * @param features the query's feature sequence
         * @return the nearest template's class, or -1 if there are none
         */
        public int classify(float[][] features) {
            double best = Double.POSITIVE_INFINITY;
            int label = -1;
            for (int i = 0; i < templates.size(); i++) {
                double d = distance(features, templates.get(i), band);
                if (d < best) {
                    best = d;
                    label = labels.get(i);
                }
            }
            return label;
        }

        /** @return the number of stored templates */
        public int size() {
            return templates.size();
        }
    }
}
