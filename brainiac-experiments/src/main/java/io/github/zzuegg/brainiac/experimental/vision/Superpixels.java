package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.PriorityQueue;

/**
 * Superpixel over-segmentation of a grey image, row-major {@code h x w}:
 *
 * <ul>
 *   <li>{@link Method#SLIC}: localised k-means in (intensity, x, y) from a grid of seeds
 *       moved to the lowest gradient in their 3 x 3 neighbourhood, each centre searching a
 *       {@code 2S x 2S} window, then connectivity enforcement (Achanta et al., TPAMI 2012);</li>
 *   <li>{@link Method#FELZENSZWALB}: graph-based merging along sorted 8-neighbour edges while
 *       the edge weight is below each component's internal difference plus {@code k / |C|},
 *       then merging components below a minimum size (Felzenszwalb &amp; Huttenlocher, IJCV
 *       2004);</li>
 *   <li>{@link Method#WATERSHED}: priority flooding (Meyer) of the gradient magnitude from one
 *       marker per grid cell at the cell's gradient minimum.</li>
 * </ul>
 *
 * Also the standard quality measures (boundary recall, under-segmentation error, achievable
 * segmentation accuracy) and per-superpixel mean pooling.
 */
public final class Superpixels {

    /** The algorithm. */
    public enum Method {
        /** Simple linear iterative clustering. */
        SLIC,
        /** Felzenszwalb-Huttenlocher graph segmentation. */
        FELZENSZWALB,
        /** Marker-based watershed. */
        WATERSHED
    }

    private Superpixels() {
    }

    /**
     * Segments an image.
     *
     * @param method the algorithm
     * @param image the image
     * @param h the height
     * @param w the width
     * @param segments the target number of segments (SLIC, watershed)
     * @param parameter SLIC's intensity scale {@code m} (smaller is more compact), or
     *     Felzenszwalb's {@code k}; unused by the watershed
     * @return per pixel the segment label, {@code 0..n-1}
     */
    public static int[] segment(Method method, float[] image, int h, int w, int segments, double parameter) {
        return switch (method) {
            case SLIC -> slic(image, h, w, segments, parameter, 10);
            case FELZENSZWALB -> felzenszwalb(image, h, w, parameter, Math.max(4, h * w / (4 * segments)));
            case WATERSHED -> watershed(image, h, w, segments);
        };
    }

    private static double[] gradient(float[] image, int h, int w) {
        double[] g = new double[h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double gx = image[y * w + Math.min(w - 1, x + 1)] - image[y * w + Math.max(0, x - 1)];
                double gy = image[Math.min(h - 1, y + 1) * w + x] - image[Math.max(0, y - 1) * w + x];
                g[y * w + x] = Math.hypot(gx, gy);
            }
        }
        return g;
    }

    /**
     * SLIC.
     *
     * @param image the image
     * @param h the height
     * @param w the width
     * @param segments the target number of segments
     * @param m the intensity scale
     * @param iterations the k-means iterations
     * @return the labels
     */
    public static int[] slic(float[] image, int h, int w, int segments, double m, int iterations) {
        double s = Math.sqrt(h * w / (double) segments);
        int step = Math.max(1, (int) Math.round(s));
        double[] grad = gradient(image, h, w);
        int count = 0;
        double[][] centres = new double[(h / step + 1) * (w / step + 1)][];
        for (int y = step / 2; y < h; y += step) {
            for (int x = step / 2; x < w; x += step) {
                int bx = x;
                int by = y;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && ny >= 0 && nx < w && ny < h && grad[ny * w + nx] < grad[by * w + bx]) {
                            bx = nx;
                            by = ny;
                        }
                    }
                }
                centres[count++] = new double[] {image[by * w + bx], bx, by};
            }
        }
        centres = Arrays.copyOf(centres, count);
        int[] label = new int[h * w];
        double[] dist = new double[h * w];
        for (int it = 0; it < iterations; it++) {
            Arrays.fill(dist, Double.POSITIVE_INFINITY);
            Arrays.fill(label, -1);
            for (int k = 0; k < count; k++) {
                double[] c = centres[k];
                int x0 = Math.max(0, (int) (c[1] - step));
                int x1 = Math.min(w - 1, (int) (c[1] + step));
                int y0 = Math.max(0, (int) (c[2] - step));
                int y1 = Math.min(h - 1, (int) (c[2] + step));
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        double di = (image[y * w + x] - c[0]) / m;
                        double dx = (x - c[1]) / s;
                        double dy = (y - c[2]) / s;
                        double d = di * di + dx * dx + dy * dy;
                        if (d < dist[y * w + x]) {
                            dist[y * w + x] = d;
                            label[y * w + x] = k;
                        }
                    }
                }
            }
            double[][] sums = new double[count][4];
            for (int p = 0; p < h * w; p++) {
                if (label[p] >= 0) {
                    double[] t = sums[label[p]];
                    t[0] += image[p];
                    t[1] += p % w;
                    t[2] += p / w;
                    t[3]++;
                }
            }
            for (int k = 0; k < count; k++) {
                if (sums[k][3] > 0) {
                    centres[k] = new double[] {sums[k][0] / sums[k][3], sums[k][1] / sums[k][3],
                        sums[k][2] / sums[k][3]};
                }
            }
        }
        for (int p = 0; p < h * w; p++) {
            if (label[p] < 0) {
                label[p] = 0;
            }
        }
        return enforceConnectivity(label, h, w, Math.max(1, step * step / 4));
    }

    /**
     * Relabels connected components and merges those below a size into a neighbour.
     *
     * @param label the labels
     * @param h the height
     * @param w the width
     * @param minSize the smallest kept component
     * @return the labels, {@code 0..n-1}
     */
    public static int[] enforceConnectivity(int[] label, int h, int w, int minSize) {
        int[] out = new int[h * w];
        Arrays.fill(out, -1);
        int next = 0;
        int[] queue = new int[h * w];
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        for (int start = 0; start < h * w; start++) {
            if (out[start] >= 0) {
                continue;
            }
            int adjacent = -1;
            for (int d = 0; d < 4; d++) {
                int nx = start % w + dx[d];
                int ny = start / w + dy[d];
                if (nx >= 0 && ny >= 0 && nx < w && ny < h && out[ny * w + nx] >= 0) {
                    adjacent = out[ny * w + nx];
                }
            }
            int head = 0;
            int tail = 0;
            queue[tail++] = start;
            out[start] = next;
            while (head < tail) {
                int p = queue[head++];
                for (int d = 0; d < 4; d++) {
                    int nx = p % w + dx[d];
                    int ny = p / w + dy[d];
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
                        int q = ny * w + nx;
                        if (out[q] < 0 && label[q] == label[start]) {
                            out[q] = next;
                            queue[tail++] = q;
                        }
                    }
                }
            }
            if (tail < minSize && adjacent >= 0) {
                for (int i = 0; i < tail; i++) {
                    out[queue[i]] = adjacent;
                }
            } else {
                next++;
            }
        }
        return out;
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /**
     * Felzenszwalb-Huttenlocher segmentation on 8-neighbour intensity differences.
     *
     * @param image the image
     * @param h the height
     * @param w the width
     * @param k the scale parameter
     * @param minSize the smallest component after merging
     * @return the labels
     */
    public static int[] felzenszwalb(float[] image, int h, int w, double k, int minSize) {
        float[] sm = blur(image, h, w);
        int n = h * w;
        int[][] offsets = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        int m = 0;
        double[] weight = new double[n * 4];
        int[] ea = new int[n * 4];
        int[] eb = new int[n * 4];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int[] o : offsets) {
                    int nx = x + o[0];
                    int ny = y + o[1];
                    if (nx < w && ny >= 0 && ny < h) {
                        ea[m] = y * w + x;
                        eb[m] = ny * w + nx;
                        weight[m] = Math.abs(sm[ea[m]] - sm[eb[m]]);
                        m++;
                    }
                }
            }
        }
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(weight[a], weight[b]));
        int[] parent = new int[n];
        int[] size = new int[n];
        double[] internal = new double[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
        for (int e : order) {
            int a = find(parent, ea[e]);
            int b = find(parent, eb[e]);
            if (a != b && weight[e] <= Math.min(internal[a] + k / size[a], internal[b] + k / size[b])) {
                parent[b] = a;
                size[a] += size[b];
                internal[a] = weight[e];
            }
        }
        for (int e : order) {
            int a = find(parent, ea[e]);
            int b = find(parent, eb[e]);
            if (a != b && (size[a] < minSize || size[b] < minSize)) {
                parent[b] = a;
                size[a] += size[b];
            }
        }
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            label[i] = find(parent, i);
        }
        return compact(label);
    }

    private static float[] blur(float[] image, int h, int w) {
        float[] out = new float[h * w];
        double[] k = {0.25, 0.5, 0.25};
        float[] tmp = new float[h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v = 0;
                for (int d = -1; d <= 1; d++) {
                    v += k[d + 1] * image[y * w + Math.max(0, Math.min(w - 1, x + d))];
                }
                tmp[y * w + x] = (float) v;
            }
        }
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v = 0;
                for (int d = -1; d <= 1; d++) {
                    v += k[d + 1] * tmp[Math.max(0, Math.min(h - 1, y + d)) * w + x];
                }
                out[y * w + x] = (float) v;
            }
        }
        return out;
    }

    /**
     * Marker-based watershed of the gradient magnitude.
     *
     * @param image the image
     * @param h the height
     * @param w the width
     * @param segments the target number of markers
     * @return the labels
     */
    public static int[] watershed(float[] image, int h, int w, int segments) {
        double[] grad = gradient(blur(image, h, w), h, w);
        int step = Math.max(1, (int) Math.round(Math.sqrt(h * w / (double) segments)));
        int[] label = new int[h * w];
        Arrays.fill(label, -1);
        PriorityQueue<double[]> queue = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        int next = 0;
        long order = 0;
        for (int y0 = 0; y0 < h; y0 += step) {
            for (int x0 = 0; x0 < w; x0 += step) {
                int best = -1;
                for (int y = y0; y < Math.min(h, y0 + step); y++) {
                    for (int x = x0; x < Math.min(w, x0 + step); x++) {
                        if (best < 0 || grad[y * w + x] < grad[best]) {
                            best = y * w + x;
                        }
                    }
                }
                label[best] = next++;
                queue.add(new double[] {grad[best], order++, best});
            }
        }
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        while (!queue.isEmpty()) {
            double[] e = queue.poll();
            int p = (int) e[2];
            for (int d = 0; d < 4; d++) {
                int nx = p % w + dx[d];
                int ny = p / w + dy[d];
                if (nx >= 0 && ny >= 0 && nx < w && ny < h && label[ny * w + nx] < 0) {
                    int q = ny * w + nx;
                    label[q] = label[p];
                    queue.add(new double[] {Math.max(grad[q], e[0]), order++, q});
                }
            }
        }
        return compact(label);
    }

    private static int[] compact(int[] label) {
        int max = 0;
        for (int l : label) {
            max = Math.max(max, l);
        }
        int[] map = new int[max + 1];
        Arrays.fill(map, -1);
        int next = 0;
        int[] out = new int[label.length];
        for (int i = 0; i < label.length; i++) {
            if (map[label[i]] < 0) {
                map[label[i]] = next++;
            }
            out[i] = map[label[i]];
        }
        return out;
    }

    /**
     * @param label the labels
     * @return the number of segments
     */
    public static int count(int[] label) {
        int max = -1;
        for (int l : label) {
            max = Math.max(max, l);
        }
        return max + 1;
    }

    private static boolean[] boundaries(int[] label, int h, int w) {
        boolean[] b = new boolean[h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = y * w + x;
                if ((x + 1 < w && label[p + 1] != label[p]) || (y + 1 < h && label[p + w] != label[p])) {
                    b[p] = true;
                }
            }
        }
        return b;
    }

    /**
     * The share of ground-truth boundary pixels within {@code tolerance} (Chebyshev) of a
     * segment boundary.
     *
     * @param label the segments
     * @param truth the ground-truth regions
     * @param h the height
     * @param w the width
     * @param tolerance the distance tolerance in pixels
     * @return the boundary recall
     */
    public static double boundaryRecall(int[] label, int[] truth, int h, int w, int tolerance) {
        boolean[] seg = boundaries(label, h, w);
        boolean[] gt = boundaries(truth, h, w);
        int hit = 0;
        int total = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!gt[y * w + x]) {
                    continue;
                }
                total++;
                search:
                for (int dy = -tolerance; dy <= tolerance; dy++) {
                    for (int dx = -tolerance; dx <= tolerance; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && ny >= 0 && nx < w && ny < h && seg[ny * w + nx]) {
                            hit++;
                            break search;
                        }
                    }
                }
            }
        }
        return total == 0 ? 1 : hit / (double) total;
    }

    private static int[][] overlap(int[] label, int[] truth) {
        int[][] o = new int[count(label)][count(truth)];
        for (int i = 0; i < label.length; i++) {
            o[label[i]][truth[i]]++;
        }
        return o;
    }

    /**
     * The under-segmentation error of Neubert &amp; Protzel: for every ground-truth region and
     * every segment overlapping it, the smaller of the inside and outside parts, summed and
     * divided by the pixel count.
     *
     * @param label the segments
     * @param truth the ground-truth regions, {@code 0..r-1}
     * @return the error
     */
    public static double undersegmentationError(int[] label, int[] truth) {
        int[][] o = overlap(label, truth);
        double sum = 0;
        for (int[] row : o) {
            int size = 0;
            for (int v : row) {
                size += v;
            }
            for (int v : row) {
                if (v > 0) {
                    sum += Math.min(v, size - v);
                }
            }
        }
        return sum / label.length;
    }

    /**
     * The achievable segmentation accuracy: the pixel accuracy when every segment takes its
     * majority ground-truth label.
     *
     * @param label the segments
     * @param truth the ground-truth regions
     * @return the accuracy
     */
    public static double achievableAccuracy(int[] label, int[] truth) {
        int[][] o = overlap(label, truth);
        double sum = 0;
        for (int[] row : o) {
            int best = 0;
            for (int v : row) {
                best = Math.max(best, v);
            }
            sum += best;
        }
        return sum / label.length;
    }

    /**
     * Mean-pools a per-pixel signal over the segments.
     *
     * @param label the segments
     * @param values the per-pixel values
     * @return per segment the mean
     */
    public static double[] pool(int[] label, float[] values) {
        int n = count(label);
        double[] sum = new double[n];
        int[] size = new int[n];
        for (int i = 0; i < label.length; i++) {
            sum[label[i]] += values[i];
            size[label[i]]++;
        }
        for (int k = 0; k < n; k++) {
            sum[k] /= Math.max(1, size[k]);
        }
        return sum;
    }

    /**
     * Labels the 4-connected components of a per-pixel class map (to turn a pixel clustering
     * into segments).
     *
     * @param classes the per-pixel classes
     * @param h the height
     * @param w the width
     * @return the labels
     */
    public static int[] components(int[] classes, int h, int w) {
        int[] out = new int[h * w];
        Arrays.fill(out, -1);
        int next = 0;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        for (int s = 0; s < h * w; s++) {
            if (out[s] >= 0) {
                continue;
            }
            out[s] = next;
            q.add(s);
            while (!q.isEmpty()) {
                int p = q.poll();
                for (int d = 0; d < 4; d++) {
                    int nx = p % w + dx[d];
                    int ny = p / w + dy[d];
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
                        int r = ny * w + nx;
                        if (out[r] < 0 && classes[r] == classes[s]) {
                            out[r] = next;
                            q.add(r);
                        }
                    }
                }
            }
            next++;
        }
        return out;
    }
}
