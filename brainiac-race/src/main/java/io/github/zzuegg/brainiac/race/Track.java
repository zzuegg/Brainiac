package io.github.zzuegg.brainiac.race;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A closed race track: a smooth centre line through control points, a fixed width, and
 * the walls on both sides. Positions along the track are measured as arc length from the
 * start line.
 *
 * <p>The built-in tracks ({@link #kidney()}, {@link #oval()}, {@link #serpent()}) fit a
 * 1000 by 750 world.
 */
public final class Track {

    /** Width of the world the built-in tracks are laid out in. */
    public static final double WORLD_WIDTH = 1000;
    /** Height of the world the built-in tracks are laid out in. */
    public static final double WORLD_HEIGHT = 750;

    private final String name;
    private final int n;
    private final double halfWidth;
    private final double[] cx;
    private final double[] cy;
    private final double[] tx;
    private final double[] ty;
    private final double[] lx;
    private final double[] ly;
    private final double[] rx;
    private final double[] ry;
    private final double[] curvature;
    private final double length;
    private final double segment;

    private Track(String name, double[][] controls, double halfWidth, int samples) {
        if (controls.length < 4) {
            throw new IllegalArgumentException("a track needs at least 4 control points");
        }
        if (!(halfWidth > 0) || samples < 16) {
            throw new IllegalArgumentException("halfWidth must be positive and samples at least 16");
        }
        this.name = name;
        this.n = samples;
        this.halfWidth = halfWidth;
        // a dense Catmull-Rom spline through the control points, resampled at equal arc length
        int m = controls.length;
        int per = 40;
        double[][] dense = new double[m * per][];
        for (int i = 0; i < m; i++) {
            for (int s = 0; s < per; s++) {
                dense[i * per + s] = catmull(controls[(i - 1 + m) % m], controls[i], controls[(i + 1) % m],
                        controls[(i + 2) % m], s / (double) per);
            }
        }
        double[] cum = new double[dense.length + 1];
        for (int i = 1; i <= dense.length; i++) {
            double[] a = dense[i - 1];
            double[] b = dense[i % dense.length];
            cum[i] = cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]);
        }
        length = cum[dense.length];
        segment = length / n;
        cx = new double[n];
        cy = new double[n];
        int j = 0;
        for (int i = 0; i < n; i++) {
            double d = i * segment;
            while (cum[j + 1] < d) {
                j++;
            }
            double t = (d - cum[j]) / (cum[j + 1] - cum[j]);
            double[] a = dense[j];
            double[] b = dense[(j + 1) % dense.length];
            cx[i] = a[0] + (b[0] - a[0]) * t;
            cy[i] = a[1] + (b[1] - a[1]) * t;
        }
        tx = new double[n];
        ty = new double[n];
        lx = new double[n];
        ly = new double[n];
        rx = new double[n];
        ry = new double[n];
        for (int i = 0; i < n; i++) {
            int a = (i - 1 + n) % n;
            int b = (i + 1) % n;
            double dx = cx[b] - cx[a];
            double dy = cy[b] - cy[a];
            double l = Math.hypot(dx, dy);
            tx[i] = dx / l;
            ty[i] = dy / l;
            // left is the side to the left of the driving direction, in a y-down world
            lx[i] = cx[i] + ty[i] * halfWidth;
            ly[i] = cy[i] - tx[i] * halfWidth;
            rx[i] = cx[i] - ty[i] * halfWidth;
            ry[i] = cy[i] + tx[i] * halfWidth;
        }
        curvature = new double[n];
        for (int i = 0; i < n; i++) {
            int a = (i - 1 + n) % n;
            int b = (i + 1) % n;
            double turn = wrap(Math.atan2(ty[b], tx[b]) - Math.atan2(ty[a], tx[a]));
            curvature[i] = turn / (2 * segment);
        }
    }

    /**
     * Builds a track through control points given as fractions of the world size.
     *
     * @param name the track's name
     * @param points control points, each {@code {x, y}} in 0..1 of the world width and height
     * @param halfWidth half the road width, in world units
     * @return the track, sampled at 360 points
     */
    public static Track of(String name, double[][] points, double halfWidth) {
        double[][] p = new double[points.length][];
        for (int i = 0; i < points.length; i++) {
            p[i] = new double[] {points[i][0] * WORLD_WIDTH, points[i][1] * WORLD_HEIGHT};
        }
        return new Track(name, p, halfWidth, 360);
    }

    /** @return a kidney-shaped track with a hairpin, 58 units half width */
    public static Track kidney() {
        return kidney(58);
    }

    /**
     * @param halfWidth half the road width
     * @return a kidney-shaped track with a hairpin
     */
    public static Track kidney(double halfWidth) {
        return of("kidney", new double[][] {{.12, .5}, {.17, .2}, {.4, .12}, {.57, .33}, {.77, .13}, {.9, .36},
                {.87, .75}, {.64, .88}, {.48, .66}, {.3, .87}, {.12, .8}}, halfWidth);
    }

    /** @return a slightly wobbly oval, 58 units half width */
    public static Track oval() {
        return oval(58);
    }

    /**
     * @param halfWidth half the road width
     * @return a slightly wobbly oval
     */
    public static Track oval(double halfWidth) {
        double[][] p = new double[14][];
        for (int i = 0; i < p.length; i++) {
            double t = i / (double) p.length * Math.PI * 2;
            double w = 1 + 0.05 * Math.sin(t * 3);
            p[i] = new double[] {.5 + .38 * Math.cos(t) * w, .5 + .36 * Math.sin(t) * w};
        }
        return of("oval", p, halfWidth);
    }

    /** @return a long winding track with S-bends, 58 units half width */
    public static Track serpent() {
        return serpent(58);
    }

    /**
     * @param halfWidth half the road width
     * @return a long winding track with S-bends
     */
    public static Track serpent(double halfWidth) {
        return of("serpent", new double[][] {{.07, .16}, {.5, .1}, {.9, .14}, {.93, .32}, {.72, .41}, {.37, .39},
                {.27, .5}, {.38, .61}, {.78, .59}, {.93, .72}, {.83, .89}, {.45, .9}, {.13, .86}, {.06, .52}},
                halfWidth);
    }

    /**
     * @param name {@code kidney}, {@code oval} or {@code serpent}
     * @param halfWidth half the road width
     * @return the built-in track of that name
     */
    public static Track named(String name, double halfWidth) {
        return switch (name.toLowerCase(Locale.ROOT)) {
            case "kidney" -> kidney(halfWidth);
            case "oval" -> oval(halfWidth);
            case "serpent" -> serpent(halfWidth);
            default -> throw new IllegalArgumentException("unknown track " + name + "; try " + names());
        };
    }

    /** @return the names of the built-in tracks */
    public static List<String> names() {
        return List.of("kidney", "oval", "serpent");
    }

    /** @return the track's name */
    public String name() {
        return name;
    }

    /** @return the length of one lap along the centre line */
    public double length() {
        return length;
    }

    /** @return half the road width */
    public double halfWidth() {
        return halfWidth;
    }

    /** @return the number of centre line samples */
    public int samples() {
        return n;
    }

    /** @return the arc length between two centre line samples */
    public double segment() {
        return segment;
    }

    /**
     * @param i a sample index (wrapped around the lap)
     * @return the centre line point {@code {x, y}}
     */
    public double[] center(int i) {
        i = Math.floorMod(i, n);
        return new double[] {cx[i], cy[i]};
    }

    /**
     * @param i a sample index (wrapped around the lap)
     * @return the unit driving direction {@code {x, y}} there
     */
    public double[] tangent(int i) {
        i = Math.floorMod(i, n);
        return new double[] {tx[i], ty[i]};
    }

    /**
     * @param i a sample index (wrapped around the lap)
     * @return the track's heading there, in radians
     */
    public double heading(int i) {
        i = Math.floorMod(i, n);
        return Math.atan2(ty[i], tx[i]);
    }

    /**
     * @param i a sample index (wrapped around the lap)
     * @return the signed curvature there (1 / radius; positive turns towards +y)
     */
    public double curvature(int i) {
        return curvature[Math.floorMod(i, n)];
    }

    /**
     * @param i a sample index
     * @return the left wall point there
     */
    public double[] left(int i) {
        i = Math.floorMod(i, n);
        return new double[] {lx[i], ly[i]};
    }

    /**
     * @param i a sample index
     * @return the right wall point there
     */
    public double[] right(int i) {
        i = Math.floorMod(i, n);
        return new double[] {rx[i], ry[i]};
    }

    /**
     * Where a point is relative to the track, searching near a known sample.
     *
     * @param x the point
     * @param y the point
     * @param near a sample index the point is known to be close to
     * @param back how many samples behind {@code near} to search
     * @param ahead how many samples ahead of {@code near} to search
     * @return the projection onto the centre line
     */
    Projection project(double x, double y, int near, int back, int ahead) {
        double best = Double.POSITIVE_INFINITY;
        int bi = near;
        double bt = 0;
        double bLat = 0;
        for (int k = -back; k <= ahead; k++) {
            int i = Math.floorMod(near + k, n);
            int j = (i + 1) % n;
            double ex = cx[j] - cx[i];
            double ey = cy[j] - cy[i];
            double t = ((x - cx[i]) * ex + (y - cy[i]) * ey) / (ex * ex + ey * ey);
            t = Math.max(0, Math.min(1, t));
            double px = cx[i] + ex * t;
            double py = cy[i] + ey * t;
            double d = Math.hypot(x - px, y - py);
            if (d < best) {
                best = d;
                bi = i;
                bt = t;
                // positive lateral offset is to the right of the driving direction (y-down world)
                double len = Math.sqrt(ex * ex + ey * ey);
                bLat = ((x - px) * -ey + (y - py) * ex) / len;
            }
        }
        return new Projection(bi, (bi + bt) * segment, bLat, best);
    }

    /** Where a point is along and across the track. */
    record Projection(int index, double along, double lateral, double distance) {
    }

    /**
     * Casts a ray against the walls near a sample.
     *
     * @return the distance to the first wall hit, or {@code max}
     */
    double raycast(double x, double y, double angle, double max, int near) {
        double dx = Math.cos(angle);
        double dy = Math.sin(angle);
        double best = max;
        int reach = (int) Math.ceil(max / segment) + 8;
        for (int w = 0; w < 2; w++) {
            double[] wx = w == 0 ? lx : rx;
            double[] wy = w == 0 ? ly : ry;
            for (int k = -reach; k <= reach; k++) {
                int i = Math.floorMod(near + k, n);
                int j = (i + 1) % n;
                double ex = wx[j] - wx[i];
                double ey = wy[j] - wy[i];
                double den = dx * ey - dy * ex;
                if (Math.abs(den) < 1e-12) {
                    continue;
                }
                double ax = wx[i] - x;
                double ay = wy[i] - y;
                double t = (ax * ey - ay * ex) / den;
                double u = (ax * dy - ay * dx) / den;
                if (t > 0 && t < best && u >= 0 && u <= 1) {
                    best = t;
                }
            }
        }
        return best;
    }

    /** @return the left wall as a closed polyline of {@code {x, y}} points */
    public List<double[]> leftWall() {
        List<double[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(new double[] {lx[i], ly[i]});
        }
        return out;
    }

    /** @return the right wall as a closed polyline of {@code {x, y}} points */
    public List<double[]> rightWall() {
        List<double[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(new double[] {rx[i], ry[i]});
        }
        return out;
    }

    static double wrap(double a) {
        a = Math.IEEEremainder(a, 2 * Math.PI);
        return a;
    }

    private static double[] catmull(double[] p0, double[] p1, double[] p2, double[] p3, double t) {
        double t2 = t * t;
        double t3 = t2 * t;
        double[] out = new double[2];
        for (int k = 0; k < 2; k++) {
            out[k] = 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3);
        }
        return out;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s (%.0f long, %.0f wide)", name, length, 2 * halfWidth);
    }
}
