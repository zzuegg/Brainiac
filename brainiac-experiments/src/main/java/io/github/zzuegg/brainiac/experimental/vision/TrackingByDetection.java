package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-object tracking by detection, frame by frame:
 *
 * <ul>
 *   <li>{@link Mode#SORT} (Bewley et al., 2016): each track carries a constant-velocity
 *       Kalman filter over its box centre and size; tracks are predicted into the new frame
 *       and assigned to the confident detections by the Hungarian algorithm
 *       ({@link MinibatchOptimalTransport#assignCost}) on {@code 1 - IoU}, matches below an
 *       IoU of 0.3 rejected; unmatched detections start tracks and unmatched tracks age out;</li>
 *   <li>{@link Mode#BYTE_TRACK} (Zhang et al., 2022): as SORT, then a second association of
 *       the low-confidence detections with the tracks left over, which keeps occluded objects
 *       (whose detections are weak) on their tracks instead of dropping them;</li>
 *   <li>{@link Mode#SORT_NO_KALMAN}: SORT matching against each track's last box, without
 *       motion prediction;</li>
 *   <li>{@link Mode#GREEDY_CENTROID}: nearest-centroid greedy association, the simplest
 *       baseline.</li>
 * </ul>
 *
 * The Kalman update reuses {@link StateSpaceFilter#kalman} one observation at a time (a
 * one-step filter from the predicted state); the prediction is done here.
 *
 * <p>Papers: Bewley et al., "Simple Online and Realtime Tracking" (2016),
 * <a href="https://arxiv.org/abs/1602.00763">arXiv:1602.00763</a>; Zhang et al., "ByteTrack:
 * Multi-Object Tracking by Associating Every Detection Box" (2022),
 * <a href="https://arxiv.org/abs/2110.06864">arXiv:2110.06864</a>.
 */
public final class TrackingByDetection {

    /** The association scheme. */
    public enum Mode {
        /** Kalman prediction and Hungarian IoU matching of confident detections. */
        SORT,
        /** SORT plus a second round with the low-confidence detections. */
        BYTE_TRACK,
        /** SORT matching against the last box, no motion model. */
        SORT_NO_KALMAN,
        /** Greedy nearest-centroid matching. */
        GREEDY_CENTROID
    }

    /**
     * A detection.
     *
     * @param box {@code [x1, y1, x2, y2]}
     * @param score its confidence
     */
    public record Detection(float[] box, float score) {
    }

    /**
     * A track's box in the current frame.
     *
     * @param id the track id
     * @param box {@code [x1, y1, x2, y2]}
     */
    public record Output(int id, float[] box) {
    }

    private static final double[][] F = {
        {1, 0, 0, 0, 1, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}};
    private static final double[][] H = {{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0}};

    private final class Track {
        final int id;
        double[] mean;
        double[][] cov;
        float[] last;
        int misses;

        Track(int id, float[] box) {
            this.id = id;
            this.mean = new double[] {(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1],
                0, 0};
            this.cov = diag(1, 1, 1, 1, 10, 10);
            this.last = box.clone();
        }

        float[] predicted() {
            if (mode == Mode.SORT_NO_KALMAN || mode == Mode.GREEDY_CENTROID) {
                return last;
            }
            return box(mean);
        }

        void predict() {
            mean = mul(F, mean);
            cov = add(mul(mul(F, cov), transpose(F)), diag(0.05, 0.05, 0.01, 0.01, 0.05, 0.05));
        }

        void update(float[] box) {
            double[] y = {(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]};
            StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(new StateSpaceFilter.LinearModel(F,
                    diag(0.05, 0.05, 0.01, 0.01, 0.05, 0.05), H, diag(0.25, 0.25, 0.25, 0.25), mean, cov),
                    new double[][] {y}, false);
            mean = e.means()[0];
            cov = e.covariances()[0];
            last = box.clone();
            misses = 0;
        }
    }

    private final Mode mode;
    private final float highThreshold;
    private final float lowThreshold;
    private final double iouThreshold;
    private final int maxMisses;
    private final List<Track> tracks = new ArrayList<>();
    private int nextId;

    /**
     * Creates a tracker.
     *
     * @param mode the association scheme
     * @param highThreshold detections at or above this score are confident
     * @param lowThreshold ByteTrack's second round uses detections from here to the high
     *     threshold
     * @param iouThreshold the least IoU a match needs (the centroid baseline uses
     *     {@code 4 / iouThreshold} pixels as its distance gate)
     * @param maxMisses frames a track survives unmatched
     */
    public TrackingByDetection(Mode mode, float highThreshold, float lowThreshold, double iouThreshold,
            int maxMisses) {
        this.mode = mode;
        this.highThreshold = highThreshold;
        this.lowThreshold = lowThreshold;
        this.iouThreshold = iouThreshold;
        this.maxMisses = maxMisses;
    }

    /**
     * Processes one frame.
     *
     * @param detections the frame's detections
     * @return the boxes of the tracks matched in this frame
     */
    public List<Output> update(List<Detection> detections) {
        if (mode == Mode.SORT || mode == Mode.BYTE_TRACK) {
            for (Track t : tracks) {
                t.predict();
            }
        }
        List<Detection> high = new ArrayList<>();
        List<Detection> low = new ArrayList<>();
        for (Detection d : detections) {
            if (d.score() >= highThreshold) {
                high.add(d);
            } else if (d.score() >= lowThreshold) {
                low.add(d);
            }
        }
        List<Output> out = new ArrayList<>();
        List<Track> pool = new ArrayList<>(tracks);
        boolean[] usedHigh = associate(pool, high, out, iouThreshold);
        if (mode == Mode.BYTE_TRACK) {
            associate(pool, low, out, 0.5);
        }
        for (Track t : pool) {
            t.misses++;
        }
        tracks.removeIf(t -> t.misses > maxMisses);
        for (int i = 0; i < high.size(); i++) {
            if (!usedHigh[i]) {
                Track t = new Track(nextId++, high.get(i).box());
                tracks.add(t);
                out.add(new Output(t.id, high.get(i).box()));
            }
        }
        return out;
    }

    /** Matches pool tracks to detections, updating and removing matched tracks from the pool. */
    private boolean[] associate(List<Track> pool, List<Detection> dets, List<Output> out, double threshold) {
        boolean[] used = new boolean[dets.size()];
        if (pool.isEmpty() || dets.isEmpty()) {
            return used;
        }
        int[] match = new int[pool.size()];
        java.util.Arrays.fill(match, -1);
        if (mode == Mode.GREEDY_CENTROID) {
            double gate = 4 / threshold;
            List<double[]> pairs = new ArrayList<>();
            for (int i = 0; i < pool.size(); i++) {
                for (int j = 0; j < dets.size(); j++) {
                    pairs.add(new double[] {centreDistance(pool.get(i).predicted(), dets.get(j).box()), i, j});
                }
            }
            pairs.sort((a, b) -> Double.compare(a[0], b[0]));
            boolean[] trackUsed = new boolean[pool.size()];
            for (double[] p : pairs) {
                if (p[0] > gate) {
                    break;
                }
                if (!trackUsed[(int) p[1]] && !used[(int) p[2]]) {
                    trackUsed[(int) p[1]] = true;
                    used[(int) p[2]] = true;
                    match[(int) p[1]] = (int) p[2];
                }
            }
        } else {
            int n = Math.max(pool.size(), dets.size());
            double[][] cost = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    cost[i][j] = i < pool.size() && j < dets.size()
                            ? 1 - DetectionMetrics.iou(pool.get(i).predicted(), dets.get(j).box()) : 1;
                }
            }
            int[] assign = MinibatchOptimalTransport.assignCost(cost);
            for (int i = 0; i < pool.size(); i++) {
                int j = assign[i];
                if (j < dets.size() && 1 - cost[i][j] >= threshold) {
                    match[i] = j;
                    used[j] = true;
                }
            }
        }
        List<Track> matched = new ArrayList<>();
        for (int i = 0; i < pool.size(); i++) {
            if (match[i] >= 0) {
                Track t = pool.get(i);
                t.update(dets.get(match[i]).box());
                out.add(new Output(t.id, dets.get(match[i]).box()));
                matched.add(t);
            }
        }
        pool.removeAll(matched);
        return used;
    }

    private static double centreDistance(float[] a, float[] b) {
        return Math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2);
    }

    private static float[] box(double[] s) {
        return new float[] {(float) (s[0] - s[2] / 2), (float) (s[1] - s[3] / 2), (float) (s[0] + s[2] / 2),
            (float) (s[1] + s[3] / 2)};
    }

    private static double[][] diag(double... v) {
        double[][] m = new double[v.length][v.length];
        for (int i = 0; i < v.length; i++) {
            m[i][i] = v[i];
        }
        return m;
    }

    private static double[] mul(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += a[i][j] * x[j];
            }
        }
        return y;
    }

    private static double[][] mul(double[][] a, double[][] b) {
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < b.length; k++) {
                for (int j = 0; j < b[0].length; j++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    private static double[][] add(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    private static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    /** @return the association scheme */
    public Mode mode() {
        return mode;
    }

    @Override
    public String toString() {
        return "TrackingByDetection(" + mode + ")";
    }
}
