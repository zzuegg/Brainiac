package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Hough transforms on edge points {@code (x, y)}:
 *
 * <ul>
 *   <li>{@link #lines}: every point votes for all {@code (rho, theta)} with
 *       {@code rho = x cos theta + y sin theta} (Duda &amp; Hart, CACM 1972); peaks are taken
 *       greedily with non-maximum suppression in a window of the accumulator;</li>
 *   <li>{@link #probabilisticLines}: the same accumulator fed by a random subset of the
 *       points (Kiryati, Eldar &amp; Bruckstein, Pattern Recognition 1991);</li>
 *   <li>{@link #circles}: every point votes for the centres at each candidate radius, in a
 *       3-D accumulator {@code (x, y, r)}; with gradient directions only the two centres
 *       along the gradient get votes (Ballard, Pattern Recognition 1981); peaks are taken on
 *       the 3 x 3 box sum of each radius slice, which absorbs centre jitter.</li>
 * </ul>
 *
 * Lines are returned as {@code (rho, theta, votes)}, circles as {@code (cx, cy, r, votes)}.
 */
public final class HoughTransform {

    private HoughTransform() {
    }

    /**
     * Detects lines.
     *
     * @param points the edge points {@code [n][2]}
     * @param width the image width
     * @param height the image height
     * @param thetaBins the angle bins over {@code [0, pi)}
     * @param minVotes the smallest accepted peak
     * @param maxLines the most lines returned
     * @param suppression the half-width of the suppression window in bins
     * @return the lines
     */
    public static List<double[]> lines(double[][] points, int width, int height, int thetaBins, int minVotes,
            int maxLines, int suppression) {
        double diag = Math.hypot(width, height);
        int rhoBins = (int) Math.ceil(2 * diag) + 1;
        int[][] acc = new int[thetaBins][rhoBins];
        double[] cos = new double[thetaBins];
        double[] sin = new double[thetaBins];
        for (int t = 0; t < thetaBins; t++) {
            double th = Math.PI * t / thetaBins;
            cos[t] = Math.cos(th);
            sin[t] = Math.sin(th);
        }
        for (double[] p : points) {
            for (int t = 0; t < thetaBins; t++) {
                int r = (int) Math.round(p[0] * cos[t] + p[1] * sin[t] + diag);
                acc[t][r]++;
            }
        }
        List<double[]> out = new ArrayList<>();
        for (int[] peak : peaks2d(acc, minVotes, maxLines, suppression, true)) {
            out.add(new double[] {peak[1] - diag, Math.PI * peak[0] / thetaBins, peak[2]});
        }
        return out;
    }

    /**
     * Detects lines from a random share of the points, with the vote threshold scaled.
     *
     * @param points the edge points
     * @param width the width
     * @param height the height
     * @param thetaBins the angle bins
     * @param minVotes the vote threshold for all points
     * @param maxLines the most lines
     * @param suppression the suppression half-width
     * @param share the share of points voting
     * @param random the random source
     * @return the lines
     */
    public static List<double[]> probabilisticLines(double[][] points, int width, int height, int thetaBins,
            int minVotes, int maxLines, int suppression, double share, SplittableRandom random) {
        List<double[]> sub = new ArrayList<>();
        for (double[] p : points) {
            if (random.nextDouble() < share) {
                sub.add(p);
            }
        }
        return lines(sub.toArray(new double[0][]), width, height, thetaBins,
            Math.max(2, (int) Math.round(minVotes * share)), maxLines, suppression);
    }

    private static List<int[]> peaks2d(int[][] acc, int minVotes, int max, int suppression, boolean wrapRows) {
        int rows = acc.length;
        int cols = acc[0].length;
        boolean[][] dead = new boolean[rows][cols];
        List<int[]> out = new ArrayList<>();
        while (out.size() < max) {
            int bi = -1;
            int bj = -1;
            int best = minVotes - 1;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (!dead[i][j] && acc[i][j] > best) {
                        best = acc[i][j];
                        bi = i;
                        bj = j;
                    }
                }
            }
            if (bi < 0) {
                break;
            }
            out.add(new int[] {bi, bj, best});
            for (int di = -suppression; di <= suppression; di++) {
                int i = bi + di;
                int j0 = bj;
                if (wrapRows && (i < 0 || i >= rows)) {
                    // theta wraps to theta - pi with rho negated
                    i = Math.floorMod(i, rows);
                    j0 = cols - 1 - bj;
                }
                if (i < 0 || i >= rows) {
                    continue;
                }
                for (int dj = -suppression; dj <= suppression; dj++) {
                    int j = j0 + dj;
                    if (j >= 0 && j < cols) {
                        dead[i][j] = true;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Detects circles.
     *
     * @param points the edge points {@code [n][2]}
     * @param gradients per point the gradient direction in radians, or {@code null} to vote
     *     around the full circle
     * @param width the width
     * @param height the height
     * @param minRadius the smallest radius
     * @param maxRadius the largest radius
     * @param minVotesPerUnitRadius the threshold as votes per pixel of radius (a full circle
     *     of radius {@code r} has about {@code 2 pi r} points)
     * @param maxCircles the most circles
     * @param suppression the suppression half-width in pixels
     * @return the circles
     */
    public static List<double[]> circles(double[][] points, double[] gradients, int width, int height,
            int minRadius, int maxRadius, double minVotesPerUnitRadius, int maxCircles, int suppression) {
        int radii = maxRadius - minRadius + 1;
        int[][][] acc = new int[radii][height][width];
        for (int i = 0; i < points.length; i++) {
            double[] p = points[i];
            for (int ri = 0; ri < radii; ri++) {
                int r = minRadius + ri;
                if (gradients != null) {
                    for (int sign = -1; sign <= 1; sign += 2) {
                        int cx = (int) Math.round(p[0] + sign * r * Math.cos(gradients[i]));
                        int cy = (int) Math.round(p[1] + sign * r * Math.sin(gradients[i]));
                        if (cx >= 0 && cy >= 0 && cx < width && cy < height) {
                            acc[ri][cy][cx]++;
                        }
                    }
                } else {
                    int steps = (int) Math.ceil(2 * Math.PI * r);
                    int lastX = Integer.MIN_VALUE;
                    int lastY = Integer.MIN_VALUE;
                    for (int s = 0; s < steps; s++) {
                        double a = 2 * Math.PI * s / steps;
                        int cx = (int) Math.round(p[0] + r * Math.cos(a));
                        int cy = (int) Math.round(p[1] + r * Math.sin(a));
                        if ((cx != lastX || cy != lastY) && cx >= 0 && cy >= 0 && cx < width && cy < height) {
                            acc[ri][cy][cx]++;
                        }
                        lastX = cx;
                        lastY = cy;
                    }
                }
            }
        }
        int[][][] smooth = new int[radii][height][width];
        for (int ri = 0; ri < radii; ri++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int v = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int yy = y + dy;
                            int xx = x + dx;
                            if (yy >= 0 && xx >= 0 && yy < height && xx < width) {
                                v += acc[ri][yy][xx];
                            }
                        }
                    }
                    smooth[ri][y][x] = v;
                }
            }
        }
        acc = smooth;
        boolean[][][] dead = new boolean[radii][height][width];
        List<double[]> out = new ArrayList<>();
        while (out.size() < maxCircles) {
            double best = 0;
            int br = -1;
            int bx = -1;
            int by = -1;
            for (int ri = 0; ri < radii; ri++) {
                double need = minVotesPerUnitRadius * (minRadius + ri);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double score = acc[ri][y][x] / (double) (minRadius + ri);
                        if (!dead[ri][y][x] && acc[ri][y][x] >= need && score > best) {
                            best = score;
                            br = ri;
                            bx = x;
                            by = y;
                        }
                    }
                }
            }
            if (br < 0) {
                break;
            }
            out.add(new double[] {bx, by, minRadius + br, acc[br][by][bx]});
            for (int ri = Math.max(0, br - suppression); ri <= Math.min(radii - 1, br + suppression); ri++) {
                for (int y = Math.max(0, by - suppression); y <= Math.min(height - 1, by + suppression); y++) {
                    for (int x = Math.max(0, bx - suppression); x <= Math.min(width - 1, bx + suppression); x++) {
                        dead[ri][y][x] = true;
                    }
                }
            }
        }
        return out;
    }
}
