package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * PatchCore: a memory bank of locally aggregated patch features of normal images, reduced by
 * greedy k-center coreset selection (on a random projection, as in the paper), scored by the
 * distance of each test patch to its nearest bank entry ({@link KnnIndex}); the image score is
 * the maximum patch score, reweighted by the density around that nearest neighbour.
 *
 * <p>Features are given per image as {@code [positions][dims]} on a {@code grid x grid} map;
 * {@link #aggregate} averages each position's {@code 3 x 3} neighbourhood first.
 *
 * <p>Paper: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022,
 * <a href="https://arxiv.org/abs/2106.08265">arXiv:2106.08265</a>.
 */
public final class PatchCore {

    private final KnnIndex bank;
    private final int dims;

    private PatchCore(KnnIndex bank, int dims) {
        this.bank = bank;
        this.dims = dims;
    }

    /**
     * Averages every position's {@code 3 x 3} neighbourhood (local aggregation).
     *
     * @param features {@code [grid * grid][dims]}
     * @param grid the map side
     * @return the aggregated features
     */
    public static float[][] aggregate(float[][] features, int grid) {
        int d = features[0].length;
        float[][] out = new float[features.length][d];
        for (int y = 0; y < grid; y++) {
            for (int x = 0; x < grid; x++) {
                int n = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && xx >= 0 && yy < grid && xx < grid) {
                            float[] f = features[yy * grid + xx];
                            for (int k = 0; k < d; k++) {
                                out[y * grid + x][k] += f[k];
                            }
                            n++;
                        }
                    }
                }
                for (int k = 0; k < d; k++) {
                    out[y * grid + x][k] /= n;
                }
            }
        }
        return out;
    }

    /**
     * Greedy k-center coreset on a random projection.
     *
     * @param points the points
     * @param size how many to keep
     * @param projection the projected width (0 for none)
     * @param seed the seed
     * @return the indices kept
     */
    public static int[] coreset(List<float[]> points, int size, int projection, long seed) {
        int n = points.size();
        size = Math.min(size, n);
        int d = points.get(0).length;
        SplittableRandom r = new SplittableRandom(seed);
        float[][] p = new float[n][];
        if (projection > 0 && projection < d) {
            float[][] w = new float[d][projection];
            for (float[] row : w) {
                for (int k = 0; k < projection; k++) {
                    row[k] = (float) (r.nextGaussian() / Math.sqrt(projection));
                }
            }
            for (int i = 0; i < n; i++) {
                p[i] = new float[projection];
                float[] x = points.get(i);
                for (int j = 0; j < d; j++) {
                    for (int k = 0; k < projection; k++) {
                        p[i][k] += x[j] * w[j][k];
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                p[i] = points.get(i);
            }
        }
        double[] dist = new double[n];
        Arrays.fill(dist, Double.MAX_VALUE);
        int[] out = new int[size];
        int cur = r.nextInt(n);
        for (int s = 0; s < size; s++) {
            out[s] = cur;
            int next = 0;
            double far = -1;
            for (int i = 0; i < n; i++) {
                dist[i] = Math.min(dist[i], squared(p[i], p[cur]));
                if (dist[i] > far) {
                    far = dist[i];
                    next = i;
                }
            }
            cur = next;
        }
        return out;
    }

    private static double squared(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /**
     * Builds the memory bank.
     *
     * @param images per normal image its aggregated patch features {@code [positions][dims]}
     * @param ratio the share of patches kept in the coreset (1 keeps all)
     * @param seed the seed
     * @return the detector
     */
    public static PatchCore fit(List<float[][]> images, double ratio, long seed) {
        List<float[]> all = new ArrayList<>();
        for (float[][] im : images) {
            all.addAll(Arrays.asList(im));
        }
        int d = all.get(0).length;
        int[] keep = ratio >= 1 ? java.util.stream.IntStream.range(0, all.size()).toArray()
                : coreset(all, (int) Math.ceil(ratio * all.size()), Math.min(d, 32), seed);
        KnnIndex bank = KnnIndex.create(d, 0, KnnIndex.Metric.L2);
        for (int i : keep) {
            bank.add(all.get(i), new float[0]);
        }
        return new PatchCore(bank, d);
    }

    /** @return the bank size */
    public int size() {
        return bank.size();
    }

    /**
     * Patch scores: the Euclidean distance to the nearest bank entry.
     *
     * @param patches {@code [positions][dims]}
     * @return per position its score
     */
    public double[] map(float[][] patches) {
        double[] out = new double[patches.length];
        for (int i = 0; i < patches.length; i++) {
            out[i] = Math.sqrt(Math.max(0, -bank.search(patches[i], 1)[0].score()));
        }
        return out;
    }

    /**
     * The image score: the largest patch score, reweighted by
     * {@code 1 - softmax} of the distances to the nearest entry's own {@code b} neighbours.
     *
     * @param patches {@code [positions][dims]}
     * @param b neighbours for the reweighting (1 disables it)
     * @return the score
     */
    public double score(float[][] patches, int b) {
        double best = -1;
        int arg = 0;
        double[] m = map(patches);
        for (int i = 0; i < m.length; i++) {
            if (m[i] > best) {
                best = m[i];
                arg = i;
            }
        }
        if (b <= 1) {
            return best;
        }
        KnnIndex.Hit nn = bank.search(patches[arg], 1)[0];
        KnnIndex.Hit[] around = bank.search(bank.key(nn.row()), b);
        double z = 0;
        double first = 0;
        for (int k = 0; k < around.length; k++) {
            double dd = Math.sqrt(squared(patches[arg], bank.key(around[k].row())));
            double e = Math.exp(dd - best);
            z += e;
            if (k == 0) {
                first = e;
            }
        }
        return (1 - first / Math.max(z, 1e-12)) * best;
    }

    /** @return the feature width */
    public int dims() {
        return dims;
    }
}
