package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayDeque;
import java.util.Arrays;

/**
 * Reads instances out of per-pixel embeddings or label maps, for instance segmentation:
 *
 * <ul>
 *   <li>{@link #seeds}: the readout of De Brabandere et al. (2017), greedy seed thresholding.
 *       Take an unassigned foreground pixel, move its centre by a few mean-shift steps to the
 *       mean of the unassigned pixels within the bandwidth, and claim every unassigned pixel
 *       within the threshold of that centre as one instance; repeat until none is left;</li>
 *   <li>{@link #meanShift}: plain flat-kernel mean shift of every pixel, then grouping of the
 *       converged points;</li>
 *   <li>{@link #connectedComponents}: 4-connected components of a semantic label map, the
 *       classic baseline, which merges touching objects of one class.</li>
 * </ul>
 *
 * Embeddings are laid out channel-major, {@code [dims][pixels]}, as a network's output plane
 * for one image.
 */
public final class EmbeddingClustering {

    private EmbeddingClustering() {
    }

    private static double dist2(float[] emb, int dims, int pixels, int p, double[] c) {
        double s = 0;
        for (int d = 0; d < dims; d++) {
            double v = emb[d * pixels + p] - c[d];
            s += v * v;
        }
        return s;
    }

    /**
     * Greedy seed clustering.
     *
     * @param embeddings {@code dims * pixels} values, channel-major
     * @param dims the embedding size
     * @param foreground which pixels to cluster
     * @param bandwidth the mean-shift radius around a seed, such as {@code deltaVar}
     * @param threshold the radius claimed as one instance, such as {@code deltaDist}
     * @param minPixels clusters smaller than this are dropped to background
     * @return per pixel 0 for background or the instance id, from 1
     */
    public static int[] seeds(float[] embeddings, int dims, boolean[] foreground, double bandwidth, double threshold,
            int minPixels) {
        int pixels = foreground.length;
        int[] label = new int[pixels];
        boolean[] free = foreground.clone();
        int next = 1;
        for (int seed = 0; seed < pixels; seed++) {
            if (!free[seed]) {
                continue;
            }
            double[] c = new double[dims];
            for (int d = 0; d < dims; d++) {
                c[d] = embeddings[d * pixels + seed];
            }
            for (int it = 0; it < 5; it++) {
                double[] sum = new double[dims];
                int n = 0;
                for (int p = 0; p < pixels; p++) {
                    if (free[p] && dist2(embeddings, dims, pixels, p, c) <= bandwidth * bandwidth) {
                        for (int d = 0; d < dims; d++) {
                            sum[d] += embeddings[d * pixels + p];
                        }
                        n++;
                    }
                }
                if (n == 0) {
                    break;
                }
                for (int d = 0; d < dims; d++) {
                    c[d] = sum[d] / n;
                }
            }
            int size = 0;
            for (int p = 0; p < pixels; p++) {
                if (free[p] && dist2(embeddings, dims, pixels, p, c) <= threshold * threshold) {
                    free[p] = false;
                    label[p] = next;
                    size++;
                }
            }
            if (free[seed]) {
                free[seed] = false;
                label[seed] = next;
                size++;
            }
            if (size < minPixels) {
                for (int p = 0; p < pixels; p++) {
                    if (label[p] == next) {
                        label[p] = 0;
                    }
                }
            } else {
                next++;
            }
        }
        return label;
    }

    /**
     * Flat-kernel mean shift: every foreground pixel's point moves to the mean of the
     * original points within the bandwidth for a few iterations; converged points within
     * half the bandwidth of each other form one instance.
     *
     * @param embeddings {@code dims * pixels} values, channel-major
     * @param dims the embedding size
     * @param foreground which pixels to cluster
     * @param bandwidth the kernel radius
     * @param iterations the shift steps
     * @return per pixel 0 for background or the instance id, from 1
     */
    public static int[] meanShift(float[] embeddings, int dims, boolean[] foreground, double bandwidth,
            int iterations) {
        int pixels = foreground.length;
        int[] fg = java.util.stream.IntStream.range(0, pixels).filter(p -> foreground[p]).toArray();
        double[][] pts = new double[fg.length][dims];
        for (int i = 0; i < fg.length; i++) {
            for (int d = 0; d < dims; d++) {
                pts[i][d] = embeddings[d * pixels + fg[i]];
            }
        }
        for (int it = 0; it < iterations; it++) {
            double[][] next = new double[fg.length][dims];
            for (int i = 0; i < fg.length; i++) {
                int n = 0;
                for (int j = 0; j < fg.length; j++) {
                    if (dist2(embeddings, dims, pixels, fg[j], pts[i]) <= bandwidth * bandwidth) {
                        for (int d = 0; d < dims; d++) {
                            next[i][d] += embeddings[d * pixels + fg[j]];
                        }
                        n++;
                    }
                }
                for (int d = 0; d < dims; d++) {
                    next[i][d] = n == 0 ? pts[i][d] : next[i][d] / n;
                }
            }
            pts = next;
        }
        int[] label = new int[pixels];
        int[] group = new int[fg.length];
        Arrays.fill(group, -1);
        int next = 0;
        for (int i = 0; i < fg.length; i++) {
            if (group[i] >= 0) {
                continue;
            }
            group[i] = next;
            for (int j = i + 1; j < fg.length; j++) {
                double s = 0;
                for (int d = 0; d < dims; d++) {
                    double v = pts[i][d] - pts[j][d];
                    s += v * v;
                }
                if (group[j] < 0 && s <= bandwidth * bandwidth / 4) {
                    group[j] = next;
                }
            }
            next++;
        }
        for (int i = 0; i < fg.length; i++) {
            label[fg[i]] = group[i] + 1;
        }
        return label;
    }

    /**
     * Connected components of each non-zero class of a label map.
     *
     * @param labels per pixel a class, 0 for background
     * @param width the map width
     * @param eight whether diagonal neighbours connect (8-neighbourhood) or not (4)
     * @param minPixels components smaller than this are dropped to background
     * @return per pixel 0 for background or the component id, from 1
     */
    public static int[] connectedComponents(int[] labels, int width, boolean eight, int minPixels) {
        int n = labels.length;
        int height = n / width;
        int[] out = new int[n];
        int next = 1;
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        java.util.List<Integer> members = new java.util.ArrayList<>();
        for (int s = 0; s < n; s++) {
            if (labels[s] == 0 || out[s] != 0) {
                continue;
            }
            members.clear();
            out[s] = next;
            queue.add(s);
            while (!queue.isEmpty()) {
                int p = queue.poll();
                members.add(p);
                int x = p % width;
                int y = p / width;
                int[][] nb = eight
                        ? new int[][] {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}, {x - 1, y - 1}, {x + 1, y - 1},
                            {x - 1, y + 1}, {x + 1, y + 1}}
                        : new int[][] {{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}};
                for (int[] q : nb) {
                    if (q[0] < 0 || q[1] < 0 || q[0] >= width || q[1] >= height) {
                        continue;
                    }
                    int i = q[1] * width + q[0];
                    if (out[i] == 0 && labels[i] == labels[s]) {
                        out[i] = next;
                        queue.add(i);
                    }
                }
            }
            if (members.size() < minPixels) {
                for (int p : members) {
                    out[p] = -1;
                }
            } else {
                next++;
            }
        }
        for (int i = 0; i < n; i++) {
            out[i] = Math.max(0, out[i]);
        }
        return out;
    }
}
