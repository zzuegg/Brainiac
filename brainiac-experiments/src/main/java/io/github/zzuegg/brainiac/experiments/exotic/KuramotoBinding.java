package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KuramotoLayer;
import io.github.zzuegg.brainiac.experiments.reasoning.LatinSquarePuzzles;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two uses of artificial Kuramoto oscillatory neurons ({@link KuramotoLayer}, Miyato et al.,
 * <a href="https://arxiv.org/abs/2410.13821">arXiv:2410.13821</a>):
 *
 * <ul>
 *   <li><b>Binding by synchrony</b> ({@link #discover}): 12x12 scenes of a square frame, a small
 *       block inside it and a bar elsewhere (objects whose positions overlap, so clustering by
 *       position cannot separate them); every pixel holds four 4-d oscillators, coupled to its lit
 *       four-neighbours when it is lit itself (an input-gated local "convolutional" coupling, set
 *       by hand rather than learned). Oscillators of one object synchronize, separate objects keep independent random
 *       phases, so clustering the final oscillator states segments the objects. Scored by the
 *       foreground adjusted Rand index (FG-ARI) against k-means on pixel positions and against an
 *       untrained slot-attention-style soft clustering of the pixel positions (three slots,
 *       softmax over slots, weighted-mean updates).</li>
 *   <li><b>Constraint reasoning</b> ({@link #latinSquares}): the {@link LatinSquarePuzzles}
 *       4x4 Latin-square completion, each cell a token with eight 4-d oscillators, the given
 *       digits as stimulus, a learned 16x16 token coupling (the row/column constraints) and a
 *       linear read-out per cell, trained with deep supervision on the last steps; compared with
 *       the Tiny Recursive Model by the caller.</li>
 * </ul>
 */
public final class KuramotoBinding {
    /** Image side. */
    public static final int SIDE = 12;

    /**
     * Segmentation scores.
     *
     * @param kuramoto FG-ARI of clustering the synchronized oscillators
     * @param kmeans FG-ARI of k-means on pixel positions
     * @param slots FG-ARI of the slot-attention-style soft clustering
     * @param energyDrop the mean relative fall of the Kuramoto energy during the dynamics
     */
    public record Discovery(double kuramoto, double kmeans, double slots, double energyDrop) {
    }

    private KuramotoBinding() {
    }

    /**
     * A scene of three non-touching objects that position alone does not separate: a square
     * frame, a small block inside the frame, and a bar elsewhere.
     *
     * @param random the random source
     * @return per pixel the object id (1 frame, 2 block, 3 bar) or 0 for background
     */
    public static int[] scene(SplittableRandom random) {
        while (true) {
            int[] img = new int[SIDE * SIDE];
            int side = 6 + random.nextInt(3);
            int fy = random.nextInt(SIDE - side + 1);
            int fx = random.nextInt(SIDE - side + 1);
            for (int k = 0; k < side; k++) {
                img[fy * SIDE + fx + k] = 1;
                img[(fy + side - 1) * SIDE + fx + k] = 1;
                img[(fy + k) * SIDE + fx] = 1;
                img[(fy + k) * SIDE + fx + side - 1] = 1;
            }
            int by = fy + 2 + random.nextInt(side - 5);
            int bx = fx + 2 + random.nextInt(side - 5);
            for (int y = by; y < by + 2; y++) {
                for (int x = bx; x < bx + 2; x++) {
                    img[y * SIDE + x] = 2;
                }
            }
            for (int attempt = 0; attempt < 50; attempt++) {
                int len = 4 + random.nextInt(5);
                boolean vertical = random.nextBoolean();
                int h = vertical ? len : 1;
                int w = vertical ? 1 : len;
                int y0 = random.nextInt(SIDE - h + 1);
                int x0 = random.nextInt(SIDE - w + 1);
                boolean free = true;
                for (int y = y0 - 1; y <= y0 + h && free; y++) {
                    for (int x = x0 - 1; x <= x0 + w && free; x++) {
                        if (y >= 0 && y < SIDE && x >= 0 && x < SIDE && img[y * SIDE + x] != 0) {
                            free = false;
                        }
                    }
                }
                if (free) {
                    for (int y = y0; y < y0 + h; y++) {
                        for (int x = x0; x < x0 + w; x++) {
                            img[y * SIDE + x] = 3;
                        }
                    }
                    return img;
                }
            }
        }
    }

    /**
     * The input-gated local coupling: 1 between lit four-neighbours.
     *
     * @param scenes the images
     * @return {@code [batch, pixels, pixels]}
     */
    public static Tensor coupling(List<int[]> scenes) {
        int p = SIDE * SIDE;
        float[] j = new float[scenes.size() * p * p];
        for (int b = 0; b < scenes.size(); b++) {
            int[] img = scenes.get(b);
            for (int y = 0; y < SIDE; y++) {
                for (int x = 0; x < SIDE; x++) {
                    int i = y * SIDE + x;
                    if (img[i] == 0) {
                        continue;
                    }
                    int[][] nb = {{y - 1, x}, {y + 1, x}, {y, x - 1}, {y, x + 1}};
                    for (int[] q : nb) {
                        if (q[0] >= 0 && q[0] < SIDE && q[1] >= 0 && q[1] < SIDE && img[q[0] * SIDE + q[1]] != 0) {
                            j[(b * p + i) * p + q[0] * SIDE + q[1]] = 1;
                        }
                    }
                }
            }
        }
        return Tensor.of(Shape.of(scenes.size(), p, p), j);
    }

    /**
     * Segments random scenes.
     *
     * @param scenes how many images
     * @param steps Kuramoto steps
     * @param seed the seed
     * @return the scores
     */
    public static Discovery discover(int scenes, int steps, long seed) {
        int objects = 3;
        int oscillators = 4;
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> imgs = new ArrayList<>();
        for (int i = 0; i < scenes; i++) {
            imgs.add(scene(r));
        }
        int n = 4;
        Tensor j = coupling(imgs);
        Tensor x0 = KuramotoLayer.randomStates(scenes, SIDE * SIDE, oscillators, n, r);
        Tensor x = KuramotoLayer.synchronize(Var.of(x0), j, null, steps, 0.5f, oscillators, n).value();
        float[] e0 = KuramotoLayer.energy(x0, j).toArray();
        float[] e1 = KuramotoLayer.energy(x, j).toArray();
        float[] xs = x.toArray();
        double ak = 0;
        double am = 0;
        double as = 0;
        double drop = 0;
        for (int b = 0; b < scenes; b++) {
            int[] img = imgs.get(b);
            List<Integer> fg = new ArrayList<>();
            for (int i = 0; i < img.length; i++) {
                if (img[i] != 0) {
                    fg.add(i);
                }
            }
            int[] truth = fg.stream().mapToInt(i -> img[i]).toArray();
            int dim = oscillators * n;
            double[][] osc = new double[fg.size()][dim];
            double[][] pos = new double[fg.size()][2];
            for (int k = 0; k < fg.size(); k++) {
                int i = fg.get(k);
                for (int d = 0; d < dim; d++) {
                    osc[k][d] = xs[(b * SIDE * SIDE + i) * dim + d];
                }
                pos[k][0] = (i / SIDE) / (double) SIDE;
                pos[k][1] = (i % SIDE) / (double) SIDE;
            }
            ak += ari(truth, kmeans(osc, objects, r));
            am += ari(truth, kmeans(pos, objects, r));
            as += ari(truth, softSlots(pos, objects, 3, r));
            drop += (e0[b] - e1[b]) / Math.max(1e-9, Math.abs(e1[b]));
        }
        return new Discovery(ak / scenes, am / scenes, as / scenes, drop / scenes);
    }

    /** k-means with ten random restarts, the lowest inertia kept. */
    static int[] kmeans(double[][] x, int k, SplittableRandom r) {
        int[] best = null;
        double bestInertia = Double.MAX_VALUE;
        for (int restart = 0; restart < 10; restart++) {
            double[][] c = new double[k][];
            for (int i = 0; i < k; i++) {
                c[i] = x[r.nextInt(x.length)].clone();
            }
            int[] assign = new int[x.length];
            for (int it = 0; it < 30; it++) {
                for (int i = 0; i < x.length; i++) {
                    assign[i] = nearest(x[i], c);
                }
                double[][] sum = new double[k][x[0].length];
                int[] count = new int[k];
                for (int i = 0; i < x.length; i++) {
                    count[assign[i]]++;
                    for (int d = 0; d < x[0].length; d++) {
                        sum[assign[i]][d] += x[i][d];
                    }
                }
                for (int i = 0; i < k; i++) {
                    if (count[i] > 0) {
                        for (int d = 0; d < x[0].length; d++) {
                            c[i][d] = sum[i][d] / count[i];
                        }
                    }
                }
            }
            double inertia = 0;
            for (int i = 0; i < x.length; i++) {
                inertia += dist2(x[i], c[assign[i]]);
            }
            if (inertia < bestInertia) {
                bestInertia = inertia;
                best = assign.clone();
            }
        }
        return best;
    }

    private static int nearest(double[] p, double[][] c) {
        int best = 0;
        for (int i = 1; i < c.length; i++) {
            if (dist2(p, c[i]) < dist2(p, c[best])) {
                best = i;
            }
        }
        return best;
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /** Slot-attention-style soft clustering: softmax over slots of -distance / tau, mean updates. */
    static int[] softSlots(double[][] x, int k, int iterations, SplittableRandom r) {
        double[][] slots = new double[k][x[0].length];
        for (double[] s : slots) {
            for (int d = 0; d < s.length; d++) {
                s[d] = 0.5 + 0.1 * r.nextGaussian();
            }
        }
        double tau = 0.01;
        double[][] attn = new double[x.length][k];
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < x.length; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int s = 0; s < k; s++) {
                    attn[i][s] = -dist2(x[i], slots[s]) / tau;
                    max = Math.max(max, attn[i][s]);
                }
                double z = 0;
                for (int s = 0; s < k; s++) {
                    attn[i][s] = Math.exp(attn[i][s] - max);
                    z += attn[i][s];
                }
                for (int s = 0; s < k; s++) {
                    attn[i][s] /= z;
                }
            }
            for (int s = 0; s < k; s++) {
                double w = 1e-9;
                double[] m = new double[x[0].length];
                for (int i = 0; i < x.length; i++) {
                    w += attn[i][s];
                    for (int d = 0; d < m.length; d++) {
                        m[d] += attn[i][s] * x[i][d];
                    }
                }
                for (int d = 0; d < m.length; d++) {
                    slots[s][d] = m[d] / w;
                }
            }
        }
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int s = 1; s < k; s++) {
                if (attn[i][s] > attn[i][out[i]]) {
                    out[i] = s;
                }
            }
        }
        return out;
    }

    /**
     * The adjusted Rand index of two labelings.
     *
     * @param a labels
     * @param b labels
     * @return ARI, 1 for identical partitions, about 0 for chance
     */
    public static double ari(int[] a, int[] b) {
        java.util.Map<Long, Integer> table = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> ra = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> rb = new java.util.HashMap<>();
        for (int i = 0; i < a.length; i++) {
            table.merge(((long) a[i] << 32) | (b[i] & 0xffffffffL), 1, Integer::sum);
            ra.merge(a[i], 1, Integer::sum);
            rb.merge(b[i], 1, Integer::sum);
        }
        double index = 0;
        for (int v : table.values()) {
            index += v * (v - 1) / 2.0;
        }
        double sa = 0;
        for (int v : ra.values()) {
            sa += v * (v - 1) / 2.0;
        }
        double sb = 0;
        for (int v : rb.values()) {
            sb += v * (v - 1) / 2.0;
        }
        double total = a.length * (a.length - 1) / 2.0;
        double expected = sa * sb / total;
        double max = (sa + sb) / 2;
        return max == expected ? 1 : (index - expected) / (max - expected);
    }

    /**
     * An AKOrN model of the Latin squares: tokens are the 16 cells, the read-out maps each cell's
     * oscillators to four digit logits after each of the last two steps.
     *
     * @param oscillators oscillators per cell
     * @param steps Kuramoto steps
     * @param seed the seed
     * @return the model, {@code [batch, 80] -> per-step [batch, 64]}, and its parameters
     */
    public static LatinModel latinSquares(int oscillators, int steps, long seed) {
        return new LatinModel(oscillators, steps, seed);
    }

    /** The AKOrN Latin-square solver. */
    public static final class LatinModel {
        private final KuramotoLayer layer;
        private final Var readout;
        private final Var bias;

        private LatinModel(int oscillators, int steps, long seed) {
            int n = 4;
            this.layer = KuramotoLayer.of(5, LatinSquarePuzzles.CELLS, oscillators, n, steps, 0.5f, seed);
            SplittableRandom r = new SplittableRandom(seed + 1);
            float[] w = new float[oscillators * n * 4];
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) (r.nextGaussian() / Math.sqrt(oscillators * n));
            }
            this.readout = Var.param(Tensor.of(Shape.of(oscillators * n, 4), w));
            this.bias = Var.param(Tensor.zeros(Shape.of(4)));
        }

        /**
         * @param input {@code [batch, 80]} one-hot cells
         * @return the digit logits {@code [batch, 64]} after each of the last two steps
         */
        public List<Var> supervised(Var input) {
            int b = input.shape().dim(0);
            List<Var> trace = layer.trajectory(input.reshape(b, LatinSquarePuzzles.CELLS, 5));
            List<Var> out = new ArrayList<>();
            for (int k = Math.max(0, trace.size() - 2); k < trace.size(); k++) {
                out.add(trace.get(k).matmul(readout).add(bias).reshape(b, LatinSquarePuzzles.OUTPUTS));
            }
            return out;
        }

        /** @return every parameter */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(layer.parameters());
            p.add(readout);
            p.add(bias);
            return p;
        }
    }
}
