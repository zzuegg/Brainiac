package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Clustering by training a softmax "cluster head" end to end: invariant information
 * clustering (IIC; Ji, Henriques and Vedaldi, ICCV 2019) and SCAN (Van Gansbeke et al.,
 * ECCV 2020).
 *
 * <ul>
 *   <li><b>IIC</b> ({@link #iicLoss}, {@link #iic}) maximises the mutual information between
 *       the cluster assignments of two augmented views of the same image, computed from their
 *       joint distribution {@code P = mean_i p1_i p2_i^T} (symmetrised). High MI needs
 *       confident, balanced and augmentation-invariant clusters, so no degenerate solution
 *       exists. An auxiliary over-clustering head with more clusters than classes is trained
 *       alongside on the same trunk.</li>
 *   <li><b>SCAN</b> ({@link #scanLoss}, {@link #scan}) first mines each sample's {@code k}
 *       nearest neighbours in a self-supervised feature space ({@link #neighbours}), then
 *       trains a head so that a sample and its neighbours get the same cluster:
 *       {@code -mean log <p_i, p_j>} plus an entropy bonus on the mean assignment for balance.
 *       A self-labelling stage then fine-tunes on the confident predictions (max probability
 *       above a threshold) by cross-entropy.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1807.06653">arXiv:1807.06653</a>,
 * <a href="https://arxiv.org/abs/2005.12320">arXiv:2005.12320</a>.
 */
public final class NeighbourClustering {

    private NeighbourClustering() {
    }

    /**
     * IIC's loss: minus the mutual information of the two views' cluster variables.
     *
     * @param p1 cluster probabilities of view 1 {@code [n, k]}
     * @param p2 cluster probabilities of view 2 {@code [n, k]}
     * @return {@code -I(c1; c2)}
     */
    public static Var iicLoss(Var p1, Var p2) {
        int n = p1.shape().dim(0);
        Var joint = p1.transpose().matmul(p2).div(n);
        joint = joint.add(joint.transpose()).div(2f).maximum(1e-8f);
        joint = joint.div(joint.sum());
        Var pi = joint.sum(1, true);
        Var pj = joint.sum(0, true);
        return joint.mul(joint.log().sub(pi.log()).sub(pj.log())).sum().neg();
    }

    /**
     * SCAN's loss: neighbour consistency plus the entropy of the mean assignment.
     *
     * @param anchors cluster probabilities of the samples {@code [n, k]}
     * @param neighbours cluster probabilities of one mined neighbour each {@code [n, k]}
     * @param entropyWeight the entropy weight, 5 in the paper
     * @return {@code -mean log <p_a, p_n> - w H(mean p_a)}
     */
    public static Var scanLoss(Var anchors, Var neighbours, float entropyWeight) {
        Var consistency = anchors.mul(neighbours).sum(1, false).add(1e-8f).log().mean().neg();
        Var mean = anchors.mean(0, false).maximum(1e-8f);
        Var entropy = mean.mul(mean.log()).sum().neg();
        return consistency.sub(entropy.mul(entropyWeight));
    }

    /**
     * The {@code k} nearest neighbours of every row by cosine similarity (itself excluded).
     *
     * @param features {@code [n][d]}
     * @param k the neighbours per row
     * @return {@code [n][k]} indices, nearest first
     */
    public static int[][] neighbours(float[][] features, int k) {
        int n = features.length;
        double[][] u = new double[n][];
        for (int i = 0; i < n; i++) {
            double s = 1e-12;
            for (float v : features[i]) {
                s += (double) v * v;
            }
            s = Math.sqrt(s);
            u[i] = new double[features[i].length];
            for (int j = 0; j < u[i].length; j++) {
                u[i][j] = features[i][j] / s;
            }
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            double[] dist = new double[n];
            for (int j = 0; j < n; j++) {
                dist[j] = -MlMatrices.dot(u[i], u[j]);
            }
            out[i] = MlMetrics.nearest(dist, i, k);
        }
        return out;
    }

    /**
     * SCAN settings.
     *
     * @param neighbours mined neighbours per sample (20 in the paper; fewer for small data)
     * @param hidden the head's hidden width
     * @param steps clustering steps
     * @param selfLabelSteps self-labelling steps (0 to skip)
     * @param threshold self-labelling confidence, 0.99 in the paper
     * @param entropyWeight the entropy weight
     * @param batch rows per step
     */
    public record ScanSettings(int neighbours, int hidden, int steps, int selfLabelSteps, double threshold,
            float entropyWeight, int batch) {

        /**
         * Small defaults.
         *
         * @return 10 neighbours, 64 hidden, 300 + 100 steps of 128 rows, threshold 0.99, weight 5
         */
        public static ScanSettings small() {
            return new ScanSettings(10, 64, 300, 100, 0.99, 5, 128);
        }
    }

    /**
     * The result of SCAN.
     *
     * @param labels the clusters after self-labelling
     * @param scanLabels the clusters after the SCAN stage only
     * @param neighbourPurity the share of mined neighbours that share their anchor's label (if
     *     labels were given), else {@code NaN}
     */
    public record ScanResult(int[] labels, int[] scanLabels, double neighbourPurity) {
    }

    /**
     * SCAN on frozen features: mine neighbours, train an MLP cluster head with the SCAN loss,
     * then self-label.
     *
     * @param features the frozen self-supervised features {@code [n][d]}
     * @param clusters the number of clusters
     * @param s the settings
     * @param truth true labels for the neighbour-purity diagnostic, or {@code null}
     * @param seed the seed
     * @return the clusters
     */
    public static ScanResult scan(float[][] features, int clusters, ScanSettings s, int[] truth, long seed) {
        int n = features.length;
        int d = features[0].length;
        float[][] x = standardise(features);
        int[][] nb = neighbours(x, s.neighbours());
        double purity = Double.NaN;
        if (truth != null) {
            int same = 0;
            for (int i = 0; i < n; i++) {
                for (int j : nb[i]) {
                    same += truth[i] == truth[j] ? 1 : 0;
                }
            }
            purity = (double) same / (n * s.neighbours());
        }
        Model head = Model.sequential().input(d).dense(s.hidden(), Activation.RELU)
                .dense(clusters, Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int step = 0; step < s.steps(); step++) {
            int[] a = new int[s.batch()];
            int[] b = new int[s.batch()];
            for (int i = 0; i < s.batch(); i++) {
                a[i] = r.nextInt(n);
                b[i] = nb[a[i]][r.nextInt(s.neighbours())];
            }
            Var pa = Activation.SOFTMAX.forward(head.forward(Var.of(rows(x, a))));
            Var pb = Activation.SOFTMAX.forward(head.forward(Var.of(rows(x, b))));
            Var loss = scanLoss(pa, pb, s.entropyWeight());
            loss.backward();
            opt.step(head.parameters());
        }
        Tensor all = rows(x, null);
        int[] scanLabels = Var.noGrad(() -> head.forward(Var.of(all))).value().argmax(1).toIntArray();
        for (int step = 0; step < s.selfLabelSteps(); step++) {
            int[] a = new int[s.batch()];
            for (int i = 0; i < s.batch(); i++) {
                a[i] = r.nextInt(n);
            }
            Tensor batch = rows(x, a);
            Tensor probs = Var.noGrad(() -> Activation.SOFTMAX.forward(head.forward(Var.of(batch)))).value();
            float[] p = probs.toArray();
            float[] ids = new float[a.length];
            int confident = 0;
            for (int i = 0; i < a.length; i++) {
                int best = 0;
                for (int c = 1; c < clusters; c++) {
                    best = p[i * clusters + c] > p[i * clusters + best] ? c : best;
                }
                boolean keep = p[i * clusters + best] >= s.threshold();
                ids[i] = keep ? best : -1;
                confident += keep ? 1 : 0;
            }
            if (confident == 0) {
                continue;
            }
            // noisy copy as a weak stand-in for SCAN's strong augmentation
            Tensor noisy = batch.add(Tensor.normal(batch.shape(), 0, 0.1f, r));
            Var loss = Functional.crossEntropy(head.forward(Var.of(noisy)), Tensor.of(Shape.of(a.length), ids));
            loss.backward();
            opt.step(head.parameters());
        }
        int[] labels = Var.noGrad(() -> head.forward(Var.of(all))).value().argmax(1).toIntArray();
        return new ScanResult(labels, scanLabels, purity);
    }

    /**
     * IIC settings.
     *
     * @param hidden the trunk's hidden width
     * @param overClusters the over-clustering head's width (e.g. 5x the classes)
     * @param steps training steps
     * @param batch rows per step
     */
    public record IicSettings(int hidden, int overClusters, int steps, int batch) {
    }

    /**
     * IIC from raw inputs: an MLP trunk with a main head and an over-clustering head, both
     * trained to maximise the mutual information between two augmented views.
     *
     * @param x the data {@code [n][d]}
     * @param clusters the main head's clusters
     * @param augment given rows and a random source, an augmented copy
     * @param s the settings
     * @param seed the seed
     * @return the main head's clusters
     */
    public static int[] iic(float[][] x, int clusters, BiFunction<Tensor, SplittableRandom, Tensor> augment,
            IicSettings s, long seed) {
        int n = x.length;
        int d = x[0].length;
        Model trunk = Model.sequential().input(d).dense(s.hidden(), Activation.RELU)
                .dense(s.hidden(), Activation.RELU).seed(seed).build();
        Model main = Model.sequential().input(s.hidden()).dense(clusters, Activation.SOFTMAX).seed(seed + 1).build();
        Model over = Model.sequential().input(s.hidden()).dense(s.overClusters(), Activation.SOFTMAX)
                .seed(seed + 2).build();
        List<Var> params = new ArrayList<>(trunk.parameters());
        params.addAll(main.parameters());
        params.addAll(over.parameters());
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed + 3);
        for (int step = 0; step < s.steps(); step++) {
            int[] a = new int[s.batch()];
            for (int i = 0; i < a.length; i++) {
                a[i] = r.nextInt(n);
            }
            Tensor b = rows(x, a);
            Var h1 = trunk.forward(Var.of(augment.apply(b, r)));
            Var h2 = trunk.forward(Var.of(augment.apply(b, r)));
            Var loss = iicLoss(main.forward(h1), main.forward(h2)).add(iicLoss(over.forward(h1), over.forward(h2)));
            loss.backward();
            opt.step(params);
        }
        Tensor all = rows(x, null);
        return Var.noGrad(() -> main.forward(trunk.forward(Var.of(all)))).value().argmax(1).toIntArray();
    }

    private static float[][] standardise(float[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        double[] var = new double[d];
        for (float[] row : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / (double) n;
            }
        }
        for (float[] row : x) {
            for (int j = 0; j < d; j++) {
                var[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = (float) ((x[i][j] - mean[j]) / Math.sqrt(var[j] + 1e-8));
            }
        }
        return out;
    }

    private static Tensor rows(float[][] x, int[] idx) {
        int m = idx == null ? x.length : idx.length;
        int d = x[0].length;
        float[] out = new float[m * d];
        for (int i = 0; i < m; i++) {
            System.arraycopy(x[idx == null ? i : idx[i]], 0, out, i * d, d);
        }
        return Tensor.of(Shape.of(m, d), out);
    }

    /**
     * Copies an array of rows (for callers that hold features as a tensor).
     *
     * @param t {@code [n, d]}
     * @return {@code [n][d]}
     */
    public static float[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] v = t.toArray();
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = Arrays.copyOfRange(v, i * d, (i + 1) * d);
        }
        return out;
    }
}
