package io.github.zzuegg.brainiac.experimental.geometry;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Occupancy grid mapping from range scans at known poses (Elfes, Computer 1989; Moravec 1988):
 * each cell holds the log-odds of being occupied, and every beam adds an inverse-sensor-model
 * increment to the cells it traverses (Bresenham ray casting): {@link Model#CONE} adds
 * {@code lFree} along the ray and {@code lOcc} at the hit; {@link Model#LEARNED} adds the
 * logit of a small network over per-cell beam features (distance to the hit, the range, the
 * neighbouring beams' ranges), fitted to the true map on training scans (Thrun, Autonomous
 * Robots 2003); {@link Model#HIT_COUNT} is the naive baseline that marks every cell ever hit as
 * occupied. Clamping the log-odds to {@code [lMin, lMax]} keeps the map responsive to change.
 */
public final class OccupancyGrid {

    /** The inverse sensor model. */
    public enum Model {
        /** Free along the ray, occupied at the hit. */
        CONE,
        /** A learned per-cell logit. */
        LEARNED,
        /** Occupied if ever hit. */
        HIT_COUNT
    }

    /** Number of per-cell beam features used by the learned model. */
    public static final int FEATURES = 5;

    private final int width;
    private final int height;
    private final Model model;
    private final double lOcc;
    private final double lFree;
    private final double lMin;
    private final double lMax;
    private final double[] logOdds;
    private final int[] hits;
    private final boolean[] seen;
    private SimpleMlp learned;

    /**
     * Creates an empty map.
     *
     * @param width the width in cells
     * @param height the height in cells
     * @param model the inverse sensor model
     * @param lOcc the occupied increment (cone model)
     * @param lFree the free increment, negative (cone model)
     * @param lMin the lower clamp (use a large negative number for no clamping)
     * @param lMax the upper clamp
     */
    public OccupancyGrid(int width, int height, Model model, double lOcc, double lFree, double lMin, double lMax) {
        this.width = width;
        this.height = height;
        this.model = model;
        this.lOcc = lOcc;
        this.lFree = lFree;
        this.lMin = lMin;
        this.lMax = lMax;
        this.logOdds = new double[width * height];
        this.hits = new int[width * height];
        this.seen = new boolean[width * height];
    }

    /**
     * Sets the learned inverse model.
     *
     * @param net an MLP with {@link #FEATURES} inputs and one logit output
     * @return this map
     */
    public OccupancyGrid learned(SimpleMlp net) {
        this.learned = net;
        return this;
    }

    /**
     * The cells along a ray from {@code (x0, y0)} to {@code (x1, y1)} by Bresenham's line
     * algorithm, inclusive.
     *
     * @param x0 start column
     * @param y0 start row
     * @param x1 end column
     * @param y1 end row
     * @return the cells as {@code {x, y}}
     */
    public static List<int[]> bresenham(int x0, int y0, int x1, int y1) {
        List<int[]> out = new ArrayList<>();
        int dx = Math.abs(x1 - x0);
        int dy = -Math.abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        int x = x0;
        int y = y0;
        while (true) {
            out.add(new int[] {x, y});
            if (x == x1 && y == y1) {
                break;
            }
            int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y += sy;
            }
        }
        return out;
    }

    /**
     * Per-cell features of one beam.
     *
     * @param along the cell's distance along the beam (cells)
     * @param range the measured range
     * @param previous the previous beam's range
     * @param next the next beam's range
     * @param maxRange the sensor range
     * @return the features
     */
    public static float[] features(double along, double range, double previous, double next, double maxRange) {
        return new float[] {(float) ((along - range) / 2), (float) (range / maxRange),
            (float) ((range - previous) / maxRange), (float) ((range - next) / maxRange),
            range >= maxRange - 1e-9 ? 1f : 0f};
    }

    /**
     * The cells a beam updates, with their features: the ray up to {@code range + 1} cells
     * (clipped to the sensor range).
     *
     * @param x the sensor column (cell units, continuous)
     * @param y the sensor row
     * @param angle the beam angle
     * @param range the measured range in cells
     * @param previous the previous beam's range
     * @param next the next beam's range
     * @param maxRange the sensor range
     * @param width the map width
     * @param height the map height
     * @return per cell {@code {index, along}} and the features in parallel lists
     */
    public static List<Object[]> beamCells(double x, double y, double angle, double range, double previous,
            double next, double maxRange, int width, int height) {
        double reach = Math.min(range + 1, maxRange);
        int x1 = (int) Math.floor(x + reach * Math.cos(angle));
        int y1 = (int) Math.floor(y + reach * Math.sin(angle));
        List<Object[]> out = new ArrayList<>();
        for (int[] c : bresenham((int) Math.floor(x), (int) Math.floor(y), x1, y1)) {
            if (c[0] < 0 || c[1] < 0 || c[0] >= width || c[1] >= height) {
                break;
            }
            double along = Math.hypot(c[0] + 0.5 - x, c[1] + 0.5 - y);
            out.add(new Object[] {c[1] * width + c[0], along, features(along, range, previous, next, maxRange)});
        }
        return out;
    }

    /**
     * Integrates one scan.
     *
     * @param x the sensor column
     * @param y the sensor row
     * @param angles the beam angles
     * @param ranges the measured ranges (cells; {@code maxRange} means no return)
     * @param maxRange the sensor range
     */
    public void update(double x, double y, double[] angles, double[] ranges, double maxRange) {
        int b = angles.length;
        for (int i = 0; i < b; i++) {
            double prev = ranges[(i + b - 1) % b];
            double next = ranges[(i + 1) % b];
            List<Object[]> cells = beamCells(x, y, angles[i], ranges[i], prev, next, maxRange, width, height);
            boolean noReturn = ranges[i] >= maxRange - 1e-9;
            float[][] batch = null;
            float[] logits = null;
            if (model == Model.LEARNED) {
                batch = new float[cells.size()][];
                for (int k = 0; k < cells.size(); k++) {
                    batch[k] = (float[]) cells.get(k)[2];
                }
                logits = predict(batch);
            }
            for (int k = 0; k < cells.size(); k++) {
                int c = (Integer) cells.get(k)[0];
                double along = (Double) cells.get(k)[1];
                seen[c] = true;
                double inc;
                if (model == Model.LEARNED) {
                    inc = logits[k];
                } else {
                    boolean hit = !noReturn && Math.abs(along - ranges[i]) < 0.75;
                    if (hit) {
                        hits[c]++;
                    }
                    if (along > ranges[i] + 0.75) {
                        continue;
                    }
                    inc = hit ? lOcc : lFree;
                }
                logOdds[c] = Math.max(lMin, Math.min(lMax, logOdds[c] + inc));
            }
        }
    }

    private float[] predict(float[][] batch) {
        if (batch.length == 0) {
            return new float[0];
        }
        float[] flat = new float[batch.length * FEATURES];
        for (int i = 0; i < batch.length; i++) {
            System.arraycopy(batch[i], 0, flat, i * FEATURES, FEATURES);
        }
        return learned.predict(Tensor.of(Shape.of(batch.length, FEATURES), flat)).toArray();
    }

    /**
     * @param cell a cell
     * @return the occupancy probability (hit-count model: 1 if ever hit, 0 if seen, 0.5 if not)
     */
    public double probability(int cell) {
        if (model == Model.HIT_COUNT) {
            return hits[cell] > 0 ? 1 : seen[cell] ? 0 : 0.5;
        }
        return 1 / (1 + Math.exp(-logOdds[cell]));
    }

    /**
     * @param cell a cell
     * @return whether any beam reached it
     */
    public boolean seen(int cell) {
        return seen[cell];
    }

    /**
     * Fits a learned inverse sensor model by logistic regression of the true occupancy on
     * per-cell beam features (scaled down so that one beam's evidence is a log-odds increment).
     *
     * @param features per sample the features
     * @param occupied per sample the true occupancy
     * @param steps gradient steps
     * @param prior the share of occupied samples' log-odds subtracted (the map prior)
     * @param seed the seed
     * @return the network
     */
    public static SimpleMlp fitInverseModel(List<float[]> features, List<Boolean> occupied, int steps, double prior,
            long seed) {
        SimpleMlp net = SimpleMlp.tanh(seed, FEATURES, 16, 16, 1);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = 256;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[b * FEATURES];
            float[] y = new float[b];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(features.size());
                System.arraycopy(features.get(k), 0, x, i * FEATURES, FEATURES);
                y[i] = occupied.get(k) ? 1 : 0;
            }
            Var z = net.forward(Var.of(Tensor.of(Shape.of(b, FEATURES), x)));
            Var yt = Var.of(Tensor.of(Shape.of(b, 1), y));
            Var loss = z.softplus().sub(z.mul(yt)).mean();
            loss.backward();
            opt.step(net.parameters());
        }
        float shift = (float) Math.log(prior / (1 - prior));
        net.bias(net.depth() - 1).assign(net.bias(net.depth() - 1).value().sub(shift));
        return net;
    }

    /**
     * Map quality on seen cells.
     *
     * @param truth per cell whether it is occupied
     * @param cells the cells to score, or {@code null} for every seen cell
     * @return {@code {accuracy, AUROC}}
     */
    public double[] score(boolean[] truth, int[] cells) {
        List<double[]> pts = new ArrayList<>();
        int correct = 0;
        int total = 0;
        int[] use = cells;
        if (use == null) {
            List<Integer> s = new ArrayList<>();
            for (int i = 0; i < truth.length; i++) {
                if (seen[i]) {
                    s.add(i);
                }
            }
            use = s.stream().mapToInt(Integer::intValue).toArray();
        }
        for (int c : use) {
            double p = probability(c);
            pts.add(new double[] {p, truth[c] ? 1 : 0});
            correct += (p > 0.5) == truth[c] ? 1 : 0;
            total++;
        }
        return new double[] {correct / (double) Math.max(1, total), auroc(pts)};
    }

    private static double auroc(List<double[]> pts) {
        double[][] a = pts.toArray(new double[0][]);
        Arrays.sort(a, (p, q) -> Double.compare(p[0], q[0]));
        double pos = 0;
        double neg = 0;
        for (double[] p : a) {
            if (p[1] > 0) {
                pos++;
            } else {
                neg++;
            }
        }
        if (pos == 0 || neg == 0) {
            return Double.NaN;
        }
        double rankSum = 0;
        int i = 0;
        while (i < a.length) {
            int j = i;
            while (j < a.length && a[j][0] == a[i][0]) {
                j++;
            }
            double avg = (i + j + 1) / 2.0;
            for (int k = i; k < j; k++) {
                if (a[k][1] > 0) {
                    rankSum += avg;
                }
            }
            i = j;
        }
        return (rankSum - pos * (pos + 1) / 2) / (pos * neg);
    }
}
