package io.github.zzuegg.brainiac.experimental.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.nn.PermutationInvariantLoss;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Speaker diarisation, "who spoke when", in two designs.
 *
 * <p>The modular pipeline (Park et al., 2022 review): speech frames from a VAD are cut into
 * overlapping windows ({@link #windows}); each window is embedded by a speaker model; the
 * embeddings are clustered, by average-linkage agglomerative clustering on cosine similarity
 * ({@link #agglomerative}) or by spectral clustering (core-style {@link SpectralClustering}) with
 * the number of speakers from the largest eigengap of the normalised affinity
 * ({@link #eigengap}); window labels are spread back to frames by majority vote
 * ({@link #frameLabels}). It assigns one speaker per frame, so it cannot handle overlap.
 *
 * <p>End-to-end neural diarisation ({@link Eend}, Fujita et al., Interspeech 2019): a
 * transformer encoder over frame features outputs, per frame, an independent speech-activity
 * probability for each of K speaker slots, trained with a permutation-invariant binary
 * cross-entropy (the best of the K! slot orders, via
 * {@link PermutationInvariantLoss#permutations}); overlapping speech is just two active slots.
 *
 * <p>{@link #errorRate} scores both: the diarisation error rate
 * {@code (missed + false alarm + confusion) / reference speech}, counted per frame and speaker
 * under the best one-to-one speaker mapping ({@link Linalg#linearAssignment}).
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2101.09624">arXiv:2101.09624</a>,
 * <a href="https://arxiv.org/abs/1909.06247">arXiv:1909.06247</a>.
 */
public final class Diarization {

    private Diarization() {
    }

    /**
     * Windows over speech: a window starts every {@code hop} frames and is kept when at least
     * half of its frames are speech.
     *
     * @param speech per-frame speech decisions
     * @param window the window length in frames
     * @param hop the step
     * @return window start frames
     */
    public static List<Integer> windows(boolean[] speech, int window, int hop) {
        List<Integer> out = new ArrayList<>();
        for (int s = 0; s + window <= speech.length; s += hop) {
            int c = 0;
            for (int t = s; t < s + window; t++) {
                if (speech[t]) {
                    c++;
                }
            }
            if (2 * c >= window) {
                out.add(s);
            }
        }
        return out;
    }

    /**
     * Cosine similarities of embeddings, shifted to {@code [0, 1]}.
     *
     * @param embeddings the embeddings
     * @return the affinity matrix ({@code (1 + cos) / 2}, zero diagonal)
     */
    public static double[][] affinity(List<double[]> embeddings) {
        int n = embeddings.size();
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = i == j ? 0 : (1 + SpeechFeatures.cosine(embeddings.get(i), embeddings.get(j))) / 2;
            }
        }
        return a;
    }

    /**
     * Average-linkage agglomerative clustering on cosine similarity, merging until {@code k}
     * clusters remain, or (with {@code k <= 0}) while the best pair's mean similarity exceeds
     * {@code threshold}.
     *
     * @param embeddings the embeddings
     * @param k the number of clusters, or 0 to stop by threshold
     * @param threshold the similarity threshold
     * @return the cluster of each embedding, numbered from 0
     */
    public static int[] agglomerative(List<double[]> embeddings, int k, double threshold) {
        int n = embeddings.size();
        double[][] sim = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sim[i][j] = SpeechFeatures.cosine(embeddings.get(i), embeddings.get(j));
            }
        }
        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            clusters.add(new ArrayList<>(List.of(i)));
        }
        while (clusters.size() > Math.max(1, k)) {
            double best = -Double.MAX_VALUE;
            int ba = -1;
            int bb = -1;
            for (int a = 0; a < clusters.size(); a++) {
                for (int b = a + 1; b < clusters.size(); b++) {
                    double s = 0;
                    for (int i : clusters.get(a)) {
                        for (int j : clusters.get(b)) {
                            s += sim[i][j];
                        }
                    }
                    s /= clusters.get(a).size() * clusters.get(b).size();
                    if (s > best) {
                        best = s;
                        ba = a;
                        bb = b;
                    }
                }
            }
            if (k <= 0 && best < threshold) {
                break;
            }
            clusters.get(ba).addAll(clusters.remove(bb));
        }
        int[] labels = new int[n];
        for (int c = 0; c < clusters.size(); c++) {
            for (int i : clusters.get(c)) {
                labels[i] = c;
            }
        }
        return labels;
    }

    /**
     * The number of clusters from the largest gap between consecutive eigenvalues of the
     * normalised affinity {@code D^-1/2 A D^-1/2}, sorted in descending order.
     *
     * @param affinity a symmetric non-negative affinity
     * @param maxK the largest count considered
     * @return the estimated count, in {@code [1, maxK]}
     */
    public static int eigengap(double[][] affinity, int maxK) {
        int n = affinity.length;
        if (n < 3) {
            return 1;
        }
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            for (double v : affinity[i]) {
                d[i] += v;
            }
            d[i] = 1 / Math.sqrt(Math.max(d[i], 1e-12));
        }
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = (float) (d[i] * affinity[i][j] * d[j]);
            }
        }
        double[] ev = Linalg.eigh(Tensor.of(Shape.of(n, n), m))[0].toDoubleArray();
        int best = 1;
        double gap = -1;
        for (int k = 1; k <= Math.min(maxK, n - 1); k++) {
            double g = ev[n - k] - ev[n - k - 1];
            if (g > gap) {
                gap = g;
                best = k;
            }
        }
        return best;
    }

    /**
     * Spectral clustering of embeddings by their cosine affinity.
     *
     * @param embeddings the embeddings
     * @param k the number of clusters, or 0 to estimate it by {@link #eigengap} (at most 6)
     * @param seed seeds k-means
     * @return the cluster of each embedding
     */
    public static int[] spectral(List<double[]> embeddings, int k, long seed) {
        double[][] a = affinity(embeddings);
        // sharpen: keep the stronger similarities, as diarisation pipelines prune the affinity
        for (double[] row : a) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.pow(row[j], 4);
            }
        }
        int clusters = k > 0 ? k : eigengap(a, 6);
        if (clusters <= 1 || embeddings.size() <= clusters) {
            return new int[embeddings.size()];
        }
        return SpectralClustering.cluster(a, clusters, seed);
    }

    /**
     * Spreads window labels to frames: each speech frame takes the most frequent label of the
     * windows covering it; frames without a covering window take the label of the nearest
     * window start.
     *
     * @param speech per-frame speech decisions
     * @param starts window starts
     * @param window the window length
     * @param labels the label of each window
     * @param speakers the number of labels
     * @return {@code [speaker][frame]} activity
     */
    public static boolean[][] frameLabels(boolean[] speech, List<Integer> starts, int window, int[] labels,
            int speakers) {
        int n = speech.length;
        int[][] votes = new int[n][speakers];
        for (int w = 0; w < starts.size(); w++) {
            for (int t = starts.get(w); t < Math.min(n, starts.get(w) + window); t++) {
                votes[t][labels[w]]++;
            }
        }
        boolean[][] out = new boolean[speakers][n];
        for (int t = 0; t < n; t++) {
            if (!speech[t]) {
                continue;
            }
            int best = -1;
            int bv = 0;
            for (int s = 0; s < speakers; s++) {
                if (votes[t][s] > bv) {
                    bv = votes[t][s];
                    best = s;
                }
            }
            if (best < 0 && !starts.isEmpty()) {
                int nearest = 0;
                for (int w = 1; w < starts.size(); w++) {
                    if (Math.abs(starts.get(w) + window / 2 - t) < Math.abs(starts.get(nearest) + window / 2 - t)) {
                        nearest = w;
                    }
                }
                best = labels[nearest];
            }
            if (best >= 0) {
                out[best][t] = true;
            }
        }
        return out;
    }

    /**
     * The diarisation error rate and its parts.
     *
     * @param hypothesis {@code [speaker][frame]} predicted activity
     * @param reference {@code [speaker][frame]} true activity
     * @return {@code {DER, missed, false alarm, confusion}}, each a fraction of the reference
     *     speaker-frames
     */
    public static double[] errorRate(boolean[][] hypothesis, boolean[][] reference) {
        int frames = reference[0].length;
        int h = hypothesis.length;
        int r = reference.length;
        int size = Math.max(h, r);
        float[] cost = new float[size * size];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < r; j++) {
                int overlap = 0;
                for (int t = 0; t < Math.min(frames, hypothesis[i].length); t++) {
                    if (hypothesis[i][t] && reference[j][t]) {
                        overlap++;
                    }
                }
                cost[i * size + j] = -overlap;
            }
        }
        int[] map = Linalg.linearAssignment(Tensor.of(Shape.of(size, size), cost));
        double miss = 0;
        double fa = 0;
        double conf = 0;
        double total = 0;
        for (int t = 0; t < frames; t++) {
            int nr = 0;
            int nh = 0;
            int correct = 0;
            for (boolean[] ref : reference) {
                nr += ref[t] ? 1 : 0;
            }
            for (int i = 0; i < h; i++) {
                if (t < hypothesis[i].length && hypothesis[i][t]) {
                    nh++;
                    if (map[i] < r && reference[map[i]][t]) {
                        correct++;
                    }
                }
            }
            total += nr;
            miss += Math.max(0, nr - nh);
            fa += Math.max(0, nh - nr);
            conf += Math.min(nr, nh) - correct;
        }
        total = Math.max(1, total);
        return new double[] {(miss + fa + conf) / total, miss / total, fa / total, conf / total};
    }

    /**
     * End-to-end neural diarisation: {@code [batch, frames, bands]} to per-slot logits
     * {@code [batch, frames, slots]}.
     */
    public static final class Eend implements Layer {
        private final int slots;
        private final int width;
        private final Var inW;
        private final Var inB;
        private final List<Layer> blocks = new ArrayList<>();
        private final Var outW;
        private final Var outB;

        /**
         * Creates the network.
         *
         * @param bands input bands
         * @param width the model width
         * @param layers transformer blocks
         * @param slots speaker slots K
         * @param random the source of initial weights
         */
        public Eend(int bands, int width, int layers, int slots, SplittableRandom random) {
            this.slots = slots;
            this.width = width;
            inW = Var.param(Tensor.normal(Shape.of(bands, width), 0, (float) Math.sqrt(1.0 / bands), random));
            inB = Var.param(Tensor.zeros(Shape.of(width)));
            for (int l = 0; l < layers; l++) {
                blocks.add(TransformerBlock.of(4).ffnSize(2 * width).create(Shape.of(16, width), random));
            }
            outW = Var.param(Tensor.normal(Shape.of(width, slots), 0, (float) Math.sqrt(1.0 / width), random));
            outB = Var.param(Tensor.zeros(Shape.of(slots)));
        }

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public Var forward(Var input, boolean training) {
            int t = input.shape().dim(1);
            Var h = input.matmul(inW).add(inB);
            h = h.add(Functional.sinusoidal(t, width, 10000f, h.value().device()));
            for (Layer b : blocks) {
                h = b.forward(h, training);
            }
            return h.matmul(outW).add(outB);
        }

        /**
         * The permutation-invariant binary cross-entropy, per example the best slot order.
         *
         * @param features {@code [b, frames, bands]}
         * @param labels {@code [b, frames, slots]} in {0, 1}
         * @return the mean loss
         */
        public Var loss(Tensor features, Tensor labels) {
            Var z = forward(Var.of(features), true);
            int b = z.shape().dim(0);
            List<Var> perms = new ArrayList<>();
            for (int[] p : PermutationInvariantLoss.permutations(slots)) {
                Var[] cols = new Var[slots];
                for (int s = 0; s < slots; s++) {
                    cols[s] = z.slice(2, p[s], p[s] + 1);
                }
                Var zp = Var.concat(2, cols);
                Var bce = zp.softplus().sub(zp.mul(labels)).mean(2, false).mean(1, false); // [b]
                perms.add(bce.reshape(b, 1));
            }
            return Var.concat(1, perms.toArray(Var[]::new)).neg().max(1, false).neg().mean();
        }

        /**
         * Thresholded activities for one recording.
         *
         * @param features {@code [frames][bands]}
         * @param threshold the probability threshold (0.5)
         * @return {@code [slot][frame]}
         */
        public boolean[][] diarize(float[][] features, double threshold) {
            float[] p = Var.noGrad(() -> forward(Var.of(SpeechFeatures.batch(List.<float[][]>of(features)))).sigmoid()).value()
                    .toArray();
            int t = features.length;
            boolean[][] out = new boolean[slots][t];
            for (int f = 0; f < t; f++) {
                for (int s = 0; s < slots; s++) {
                    out[s][f] = p[f * slots + s] > threshold;
                }
            }
            return out;
        }

        @Override
        public Shape outputShape(Shape input) {
            return Shape.of(input.dim(0), slots);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(inW, inB));
            for (Layer b : blocks) {
                p.addAll(b.parameters());
            }
            p.add(outW);
            p.add(outB);
            return p;
        }

        @Override
        public String toString() {
            return "EEND(" + width + ", " + blocks.size() + " blocks, " + slots + " slots)";
        }
    }
}
