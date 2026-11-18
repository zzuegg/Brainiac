package io.github.zzuegg.brainiac.experimental.geometry;

import io.github.zzuegg.brainiac.experimental.data.GeometryScenes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Simultaneous localisation and mapping for a planar robot with odometry and range-bearing
 * landmark measurements ({@link GeometryScenes.RobotWorld}):
 *
 * <ul>
 *   <li>{@link #ekfSlam}: one joint Gaussian over the pose and every landmark (Smith, Self &amp;
 *       Cheeseman 1990), quadratic in the map size per update, with known data association or
 *       nearest-neighbour association by a Mahalanobis gate;</li>
 *   <li>{@link #fastSlam2}: a particle filter over paths with one 2x2 EKF per landmark per
 *       particle (Montemerlo et al., AAAI 2002); version 2.0 (IJCAI 2003) samples each pose from
 *       a proposal that already includes the step's measurements and weighs by the
 *       measurement likelihood under the motion and landmark uncertainty;</li>
 *   <li>{@link #poseGraph}: graph-based SLAM over keyframes (Lu &amp; Milios 1997; Grisetti et
 *       al. 2010): odometry and loop-closure constraints as SE(2) residuals, solved by
 *       Gauss-Newton on the information matrix with the first pose fixed; landmarks are then
 *       placed from the optimised keyframes;</li>
 *   <li>baselines: {@link #deadReckoning}, EKF localisation with the true map
 *       ({@link #ekfLocalisation}, the oracle) and a particle-filter localiser with the true map
 *       ({@link #particleLocalisation}).</li>
 * </ul>
 */
public final class Slam {

    private Slam() {
    }

    /**
     * An estimate.
     *
     * @param poses the estimated pose per step (or per keyframe, see {@code steps})
     * @param steps the time step of each estimated pose
     * @param landmarks the estimated landmarks per true landmark id ({@code null} if never seen)
     * @param millisPerStep mean wall time per time step
     */
    public record Estimate(double[][] poses, int[] steps, double[][] landmarks, double millisPerStep) {

        /**
         * The absolute trajectory error (RMSE of positions; the first pose is known, so no
         * alignment).
         *
         * @param world the world
         * @return the position RMSE
         */
        public double trajectoryError(GeometryScenes.RobotWorld world) {
            double s = 0;
            for (int i = 0; i < poses.length; i++) {
                double[] t = world.poses()[steps[i]];
                s += sq(poses[i][0] - t[0]) + sq(poses[i][1] - t[1]);
            }
            return Math.sqrt(s / poses.length);
        }

        /**
         * The landmark RMSE over landmarks the method placed.
         *
         * @param world the world
         * @return the RMSE, NaN if no map
         */
        public double landmarkError(GeometryScenes.RobotWorld world) {
            if (landmarks == null) {
                return Double.NaN;
            }
            double s = 0;
            int c = 0;
            for (int i = 0; i < landmarks.length; i++) {
                if (landmarks[i] != null) {
                    s += sq(landmarks[i][0] - world.landmarks()[i][0]) + sq(landmarks[i][1] - world.landmarks()[i][1]);
                    c++;
                }
            }
            return c == 0 ? Double.NaN : Math.sqrt(s / c);
        }
    }

    private static double sq(double x) {
        return x * x;
    }

    private static int[] allSteps(int n) {
        int[] s = new int[n];
        for (int i = 0; i < n; i++) {
            s[i] = i;
        }
        return s;
    }

    /**
     * Integrates odometry alone.
     *
     * @param world the world
     * @return the estimate (no map)
     */
    public static Estimate deadReckoning(GeometryScenes.RobotWorld world) {
        int n = world.steps();
        double[][] p = new double[n][];
        p[0] = world.poses()[0].clone();
        for (int k = 1; k < n; k++) {
            p[k] = GeometryScenes.compose(p[k - 1], world.odometry()[k - 1]);
        }
        return new Estimate(p, allSteps(n), null, 0);
    }

    // ---- shared EKF pieces ---------------------------------------------------------

    private static double[][] motionJacobian(double[] pose, double[] u) {
        double c = Math.cos(pose[2]);
        double s = Math.sin(pose[2]);
        return new double[][] {{1, 0, -s * u[0] - c * u[1]}, {0, 1, c * u[0] - s * u[1]}, {0, 0, 1}};
    }

    private static double[][] noiseJacobian(double[] pose) {
        double c = Math.cos(pose[2]);
        double s = Math.sin(pose[2]);
        return new double[][] {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
    }

    private static double[][] motionNoise(GeometryScenes.RobotWorld w, double[] pose) {
        double[][] g = noiseJacobian(pose);
        double[] q = w.odometryNoise();
        double[][] d = {{q[0] * q[0], 0, 0}, {0, q[1] * q[1], 0}, {0, 0, q[2] * q[2]}};
        return GeometryLinalg.multiply(GeometryLinalg.multiply(g, d), GeometryLinalg.transpose(g));
    }

    /** Predicted measurement and Jacobians: {@code {z, Hpose (2x3), Hlandmark (2x2)}}. */
    private static double[][][] observe(double[] pose, double[] l) {
        double dx = l[0] - pose[0];
        double dy = l[1] - pose[1];
        double q = Math.max(dx * dx + dy * dy, 1e-12);
        double r = Math.sqrt(q);
        double[][] z = {{r, GeometryLinalg.wrap(Math.atan2(dy, dx) - pose[2])}};
        double[][] hp = {{-dx / r, -dy / r, 0}, {dy / q, -dx / q, -1}};
        double[][] hl = {{dx / r, dy / r}, {-dy / q, dx / q}};
        return new double[][][] {z, hp, hl};
    }

    private static double[] innovation(double[] z, double[] zhat) {
        return new double[] {z[0] - zhat[0], GeometryLinalg.wrap(z[1] - zhat[1])};
    }

    private static double[] place(double[] pose, double range, double bearing) {
        return new double[] {pose[0] + range * Math.cos(pose[2] + bearing), pose[1] + range * Math.sin(pose[2] + bearing)};
    }

    /**
     * EKF-SLAM.
     *
     * @param world the world
     * @param knownAssociation use the true landmark ids; otherwise associate by nearest
     *     Mahalanobis distance with a 99% gate and start a new landmark when nothing matches
     * @return the estimate
     */
    public static Estimate ekfSlam(GeometryScenes.RobotWorld world, boolean knownAssociation) {
        int k = world.landmarks().length;
        int cap = knownAssociation ? k : 3 * k;
        int dim = 3 + 2 * cap;
        double[] x = new double[dim];
        double[][] p = new double[dim][dim];
        System.arraycopy(world.poses()[0], 0, x, 0, 3);
        int[] slot = new int[k];
        Arrays.fill(slot, -1);
        int[] truthOfSlot = new int[cap];
        Arrays.fill(truthOfSlot, -1);
        int used = 0;
        double rr = sq(world.rangeNoise());
        double rb = sq(world.bearingNoise());
        int n = world.steps();
        double[][] poses = new double[n][];
        poses[0] = world.poses()[0].clone();
        long t0 = System.nanoTime();
        for (int step = 1; step < n; step++) {
            double[] u = world.odometry()[step - 1];
            double[] pose = {x[0], x[1], x[2]};
            double[][] f = motionJacobian(pose, u);
            double[][] qn = motionNoise(world, pose);
            double[] np = GeometryScenes.compose(pose, u);
            System.arraycopy(np, 0, x, 0, 3);
            int live = 3 + 2 * used;
            double[][] fpp = new double[3][3];
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    for (int c = 0; c < 3; c++) {
                        fpp[a][b] += f[a][c] * p[c][b];
                    }
                }
            }
            double[][] fppf = new double[3][3];
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    for (int c = 0; c < 3; c++) {
                        fppf[a][b] += fpp[a][c] * f[b][c];
                    }
                    fppf[a][b] += qn[a][b];
                }
            }
            for (int j = 3; j < live; j++) {
                double[] col = new double[3];
                for (int a = 0; a < 3; a++) {
                    for (int c = 0; c < 3; c++) {
                        col[a] += f[a][c] * p[c][j];
                    }
                }
                for (int a = 0; a < 3; a++) {
                    p[a][j] = col[a];
                    p[j][a] = col[a];
                }
            }
            for (int a = 0; a < 3; a++) {
                System.arraycopy(fppf[a], 0, p[a], 0, 3);
            }
            for (double[] z : world.measurements().get(step)) {
                int id = (int) z[0];
                double[] meas = {z[1], z[2]};
                int s;
                if (knownAssociation) {
                    s = slot[id];
                } else {
                    s = associate(x, p, used, meas, rr, rb);
                }
                if (s < 0) {
                    if (used >= cap) {
                        continue;
                    }
                    s = used++;
                    if (knownAssociation) {
                        slot[id] = s;
                    }
                    truthOfSlot[s] = id;
                    initialise(x, p, s, meas, rr, rb, 3 + 2 * used);
                    continue;
                }
                if (!knownAssociation && truthOfSlot[s] < 0) {
                    truthOfSlot[s] = id;
                }
                update(x, p, s, meas, rr, rb, 3 + 2 * used);
            }
            poses[step] = new double[] {x[0], x[1], x[2]};
        }
        double ms = (System.nanoTime() - t0) / 1e6 / (n - 1);
        double[][] lm = new double[k][];
        for (int s = 0; s < used; s++) {
            int id = truthOfSlot[s];
            if (id >= 0 && lm[id] == null) {
                lm[id] = new double[] {x[3 + 2 * s], x[4 + 2 * s]};
            }
        }
        return new Estimate(poses, allSteps(n), lm, ms);
    }

    private static int associate(double[] x, double[][] p, int used, double[] z, double rr, double rb) {
        int best = -1;
        double bd = 9.21;
        double[] pose = {x[0], x[1], x[2]};
        for (int s = 0; s < used; s++) {
            double[][][] o = observe(pose, new double[] {x[3 + 2 * s], x[4 + 2 * s]});
            double[][] sm = innovationCovariance(p, s, o[1], o[2], rr, rb);
            double[] v = innovation(z, o[0][0]);
            double[][] si = GeometryLinalg.inverse(sm);
            if (si == null) {
                continue;
            }
            double d = v[0] * (si[0][0] * v[0] + si[0][1] * v[1]) + v[1] * (si[1][0] * v[0] + si[1][1] * v[1]);
            if (d < bd) {
                bd = d;
                best = s;
            }
        }
        return best;
    }

    /** The 2 x live measurement Jacobian row pair as a sparse pair (pose and landmark parts). */
    private static double[][] innovationCovariance(double[][] p, int s, double[][] hp, double[][] hl, double rr,
            double rb) {
        int li = 3 + 2 * s;
        int[] idx = {0, 1, 2, li, li + 1};
        double[][] h = new double[2][5];
        for (int a = 0; a < 2; a++) {
            System.arraycopy(hp[a], 0, h[a], 0, 3);
            h[a][3] = hl[a][0];
            h[a][4] = hl[a][1];
        }
        double[][] sm = new double[2][2];
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                double v = 0;
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        v += h[a][i] * p[idx[i]][idx[j]] * h[b][j];
                    }
                }
                sm[a][b] = v;
            }
        }
        sm[0][0] += rr;
        sm[1][1] += rb;
        return sm;
    }

    private static void initialise(double[] x, double[][] p, int s, double[] z, double rr, double rb, int live) {
        double[] pose = {x[0], x[1], x[2]};
        double[] l = place(pose, z[0], z[1]);
        int li = 3 + 2 * s;
        x[li] = l[0];
        x[li + 1] = l[1];
        double a = pose[2] + z[1];
        double[][] gp = {{1, 0, -z[0] * Math.sin(a)}, {0, 1, z[0] * Math.cos(a)}};
        double[][] gz = {{Math.cos(a), -z[0] * Math.sin(a)}, {Math.sin(a), z[0] * Math.cos(a)}};
        for (int j = 0; j < live; j++) {
            if (j == li || j == li + 1) {
                continue;
            }
            for (int r = 0; r < 2; r++) {
                double v = 0;
                for (int c = 0; c < 3; c++) {
                    v += gp[r][c] * p[c][j];
                }
                p[li + r][j] = v;
                p[j][li + r] = v;
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                double v = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        v += gp[r][i] * p[i][j] * gp[c][j];
                    }
                }
                v += gz[r][0] * rr * gz[c][0] + gz[r][1] * rb * gz[c][1];
                p[li + r][li + c] = v;
            }
        }
    }

    private static void update(double[] x, double[][] p, int s, double[] z, double rr, double rb, int live) {
        double[] pose = {x[0], x[1], x[2]};
        int li = 3 + 2 * s;
        double[][][] o = observe(pose, new double[] {x[li], x[li + 1]});
        double[] v = innovation(z, o[0][0]);
        int[] idx = {0, 1, 2, li, li + 1};
        double[][] h = new double[2][5];
        for (int a = 0; a < 2; a++) {
            System.arraycopy(o[1][a], 0, h[a], 0, 3);
            h[a][3] = o[2][a][0];
            h[a][4] = o[2][a][1];
        }
        double[][] pht = new double[live][2];
        for (int i = 0; i < live; i++) {
            for (int a = 0; a < 2; a++) {
                double sum = 0;
                for (int j = 0; j < 5; j++) {
                    sum += p[i][idx[j]] * h[a][j];
                }
                pht[i][a] = sum;
            }
        }
        double[][] sm = new double[2][2];
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                double sum = 0;
                for (int j = 0; j < 5; j++) {
                    sum += h[a][j] * pht[idx[j]][b];
                }
                sm[a][b] = sum;
            }
        }
        sm[0][0] += rr;
        sm[1][1] += rb;
        double[][] si = GeometryLinalg.inverse(sm);
        if (si == null) {
            return;
        }
        double[][] gain = new double[live][2];
        for (int i = 0; i < live; i++) {
            for (int a = 0; a < 2; a++) {
                gain[i][a] = pht[i][0] * si[0][a] + pht[i][1] * si[1][a];
            }
        }
        for (int i = 0; i < live; i++) {
            x[i] += gain[i][0] * v[0] + gain[i][1] * v[1];
        }
        x[2] = GeometryLinalg.wrap(x[2]);
        for (int i = 0; i < live; i++) {
            for (int j = 0; j < live; j++) {
                p[i][j] -= gain[i][0] * pht[j][0] + gain[i][1] * pht[j][1];
            }
        }
        for (int i = 0; i < live; i++) {
            for (int j = i + 1; j < live; j++) {
                double m = 0.5 * (p[i][j] + p[j][i]);
                p[i][j] = m;
                p[j][i] = m;
            }
        }
    }

    /**
     * EKF localisation with the true landmark map (the oracle).
     *
     * @param world the world
     * @return the estimate
     */
    public static Estimate ekfLocalisation(GeometryScenes.RobotWorld world) {
        int n = world.steps();
        double[] x = world.poses()[0].clone();
        double[][] p = new double[3][3];
        double rr = sq(world.rangeNoise());
        double rb = sq(world.bearingNoise());
        double[][] poses = new double[n][];
        poses[0] = x.clone();
        long t0 = System.nanoTime();
        for (int step = 1; step < n; step++) {
            double[] u = world.odometry()[step - 1];
            double[][] f = motionJacobian(x, u);
            double[][] qn = motionNoise(world, x);
            x = GeometryScenes.compose(x, u);
            p = GeometryLinalg.multiply(GeometryLinalg.multiply(f, p), GeometryLinalg.transpose(f));
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    p[a][b] += qn[a][b];
                }
            }
            for (double[] z : world.measurements().get(step)) {
                double[][][] o = observe(x, world.landmarks()[(int) z[0]]);
                double[] v = innovation(new double[] {z[1], z[2]}, o[0][0]);
                double[][] h = o[1];
                double[][] pht = GeometryLinalg.multiply(p, GeometryLinalg.transpose(h));
                double[][] sm = GeometryLinalg.multiply(h, pht);
                sm[0][0] += rr;
                sm[1][1] += rb;
                double[][] gain = GeometryLinalg.multiply(pht, GeometryLinalg.inverse(sm));
                for (int a = 0; a < 3; a++) {
                    x[a] += gain[a][0] * v[0] + gain[a][1] * v[1];
                }
                x[2] = GeometryLinalg.wrap(x[2]);
                double[][] kh = GeometryLinalg.multiply(gain, h);
                double[][] ikh = GeometryLinalg.identity(3);
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        ikh[a][b] -= kh[a][b];
                    }
                }
                p = GeometryLinalg.multiply(ikh, p);
            }
            poses[step] = x.clone();
        }
        return new Estimate(poses, allSteps(n), null, (System.nanoTime() - t0) / 1e6 / (n - 1));
    }

    /**
     * Monte Carlo localisation with the true map.
     *
     * @param world the world
     * @param particles the number of particles
     * @param seed the seed
     * @return the estimate
     */
    public static Estimate particleLocalisation(GeometryScenes.RobotWorld world, int particles, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = world.steps();
        double[][] ps = new double[particles][];
        for (int i = 0; i < particles; i++) {
            ps[i] = world.poses()[0].clone();
        }
        double[] w = new double[particles];
        Arrays.fill(w, 1.0 / particles);
        double[][] poses = new double[n][];
        poses[0] = world.poses()[0].clone();
        double[] q = world.odometryNoise();
        long t0 = System.nanoTime();
        for (int step = 1; step < n; step++) {
            double[] u = world.odometry()[step - 1];
            double[] logw = new double[particles];
            for (int i = 0; i < particles; i++) {
                double[] noisy = {u[0] + q[0] * r.nextGaussian(), u[1] + q[1] * r.nextGaussian(),
                    u[2] + q[2] * r.nextGaussian()};
                ps[i] = GeometryScenes.compose(ps[i], noisy);
                double lw = Math.log(Math.max(w[i], 1e-300));
                for (double[] z : world.measurements().get(step)) {
                    double[] zh = observe(ps[i], world.landmarks()[(int) z[0]])[0][0];
                    double[] v = innovation(new double[] {z[1], z[2]}, zh);
                    lw -= 0.5 * (v[0] * v[0] / sq(world.rangeNoise()) + v[1] * v[1] / sq(world.bearingNoise()));
                }
                logw[i] = lw;
            }
            w = normalise(logw);
            poses[step] = mean(ps, w);
            if (effective(w) < particles / 2.0) {
                int[] pick = resample(w, r);
                double[][] np = new double[particles][];
                for (int i = 0; i < particles; i++) {
                    np[i] = ps[pick[i]].clone();
                }
                ps = np;
                Arrays.fill(w, 1.0 / particles);
            }
        }
        return new Estimate(poses, allSteps(n), null, (System.nanoTime() - t0) / 1e6 / (n - 1));
    }

    private static double[] normalise(double[] logw) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logw) {
            m = Math.max(m, v);
        }
        double s = 0;
        double[] w = new double[logw.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = Math.exp(logw[i] - m);
            s += w[i];
        }
        for (int i = 0; i < w.length; i++) {
            w[i] /= s;
        }
        return w;
    }

    private static double effective(double[] w) {
        double s = 0;
        for (double v : w) {
            s += v * v;
        }
        return 1 / s;
    }

    private static int[] resample(double[] w, SplittableRandom r) {
        int n = w.length;
        int[] out = new int[n];
        double step = 1.0 / n;
        double u = r.nextDouble() * step;
        double c = w[0];
        int i = 0;
        for (int m = 0; m < n; m++) {
            double target = u + m * step;
            while (target > c && i < n - 1) {
                i++;
                c += w[i];
            }
            out[m] = i;
        }
        return out;
    }

    private static double[] mean(double[][] ps, double[] w) {
        double x = 0;
        double y = 0;
        double c = 0;
        double s = 0;
        for (int i = 0; i < ps.length; i++) {
            x += w[i] * ps[i][0];
            y += w[i] * ps[i][1];
            c += w[i] * Math.cos(ps[i][2]);
            s += w[i] * Math.sin(ps[i][2]);
        }
        return new double[] {x, y, Math.atan2(s, c)};
    }

    // ---- FastSLAM 2.0 --------------------------------------------------------------

    /** One particle: a pose and a map of landmark EKFs. */
    private static final class Particle {
        double[] pose;
        double[][] mean;
        double[][][] cov;

        Particle(double[] pose, int landmarks) {
            this.pose = pose.clone();
            this.mean = new double[landmarks][];
            this.cov = new double[landmarks][][];
        }

        Particle copy() {
            Particle p = new Particle(pose, mean.length);
            for (int i = 0; i < mean.length; i++) {
                if (mean[i] != null) {
                    p.mean[i] = mean[i].clone();
                    p.cov[i] = new double[][] {cov[i][0].clone(), cov[i][1].clone()};
                }
            }
            return p;
        }
    }

    /**
     * FastSLAM 2.0 with known data association.
     *
     * @param world the world
     * @param particles the number of particles
     * @param seed the seed
     * @return the estimate (the map from the most likely particle)
     */
    public static Estimate fastSlam2(GeometryScenes.RobotWorld world, int particles, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int k = world.landmarks().length;
        int n = world.steps();
        Particle[] ps = new Particle[particles];
        for (int i = 0; i < particles; i++) {
            ps[i] = new Particle(world.poses()[0], k);
        }
        double[] w = new double[particles];
        Arrays.fill(w, 1.0 / particles);
        double[][] rm = {{sq(world.rangeNoise()), 0}, {0, sq(world.bearingNoise())}};
        double[][] poses = new double[n][];
        poses[0] = world.poses()[0].clone();
        long t0 = System.nanoTime();
        for (int step = 1; step < n; step++) {
            double[] u = world.odometry()[step - 1];
            List<double[]> zs = world.measurements().get(step);
            double[] logw = new double[particles];
            for (int i = 0; i < particles; i++) {
                Particle pt = ps[i];
                double[] mu = GeometryScenes.compose(pt.pose, u);
                double[][] motion = motionNoise(world, pt.pose);
                double[][] sigma = motion;
                for (double[] z : zs) {
                    int id = (int) z[0];
                    if (pt.mean[id] == null) {
                        continue;
                    }
                    double[][][] o = observe(mu, pt.mean[id]);
                    double[][] hl = o[2];
                    double[][] q = add(GeometryLinalg.multiply(GeometryLinalg.multiply(hl, pt.cov[id]),
                        GeometryLinalg.transpose(hl)), rm);
                    double[][] qi = GeometryLinalg.inverse(q);
                    double[][] hp = o[1];
                    double[][] si = GeometryLinalg.inverse(sigma);
                    if (qi == null || si == null) {
                        continue;
                    }
                    double[][] info = add(GeometryLinalg.multiply(GeometryLinalg.multiply(GeometryLinalg.transpose(hp),
                        qi), hp), si);
                    double[][] ns = GeometryLinalg.inverse(info);
                    if (ns == null) {
                        continue;
                    }
                    double[] v = innovation(new double[] {z[1], z[2]}, o[0][0]);
                    double[] dm = GeometryLinalg.multiply(GeometryLinalg.multiply(GeometryLinalg.multiply(ns,
                        GeometryLinalg.transpose(hp)), qi), v);
                    mu = new double[] {mu[0] + dm[0], mu[1] + dm[1], GeometryLinalg.wrap(mu[2] + dm[2])};
                    sigma = ns;
                }
                double[] sample = gaussian(mu, sigma, r);
                double lw = Math.log(Math.max(w[i], 1e-300));
                for (double[] z : zs) {
                    int id = (int) z[0];
                    if (pt.mean[id] == null) {
                        pt.mean[id] = place(sample, z[1], z[2]);
                        double a = sample[2] + z[2];
                        double[][] gz = {{Math.cos(a), -z[1] * Math.sin(a)}, {Math.sin(a), z[1] * Math.cos(a)}};
                        pt.cov[id] = GeometryLinalg.multiply(GeometryLinalg.multiply(gz, rm), GeometryLinalg.transpose(gz));
                        continue;
                    }
                    double[][][] o = observe(sample, pt.mean[id]);
                    double[] v = innovation(new double[] {z[1], z[2]}, o[0][0]);
                    double[][] hl = o[2];
                    double[][] hp = o[1];
                    double[][] q = add(GeometryLinalg.multiply(GeometryLinalg.multiply(hl, pt.cov[id]),
                        GeometryLinalg.transpose(hl)), rm);
                    double[][] l = add(GeometryLinalg.multiply(GeometryLinalg.multiply(hp, motion),
                        GeometryLinalg.transpose(hp)), q);
                    double[][] li = GeometryLinalg.inverse(l);
                    double[][] qi = GeometryLinalg.inverse(q);
                    if (li == null || qi == null) {
                        continue;
                    }
                    double det = l[0][0] * l[1][1] - l[0][1] * l[1][0];
                    lw += -0.5 * (v[0] * (li[0][0] * v[0] + li[0][1] * v[1]) + v[1] * (li[1][0] * v[0]
                        + li[1][1] * v[1])) - 0.5 * Math.log(Math.max(det, 1e-300));
                    double[][] gain = GeometryLinalg.multiply(GeometryLinalg.multiply(pt.cov[id],
                        GeometryLinalg.transpose(hl)), qi);
                    double[] dm = GeometryLinalg.multiply(gain, v);
                    pt.mean[id] = new double[] {pt.mean[id][0] + dm[0], pt.mean[id][1] + dm[1]};
                    double[][] kh = GeometryLinalg.multiply(gain, hl);
                    double[][] ikh = {{1 - kh[0][0], -kh[0][1]}, {-kh[1][0], 1 - kh[1][1]}};
                    pt.cov[id] = GeometryLinalg.multiply(ikh, pt.cov[id]);
                }
                pt.pose = sample;
                logw[i] = lw;
            }
            w = normalise(logw);
            double[][] pp = new double[particles][];
            for (int i = 0; i < particles; i++) {
                pp[i] = ps[i].pose;
            }
            poses[step] = mean(pp, w);
            if (effective(w) < particles / 2.0) {
                int[] pick = resample(w, r);
                Particle[] np = new Particle[particles];
                for (int i = 0; i < particles; i++) {
                    np[i] = ps[pick[i]].copy();
                }
                ps = np;
                Arrays.fill(w, 1.0 / particles);
            }
        }
        int best = 0;
        for (int i = 1; i < particles; i++) {
            if (w[i] > w[best]) {
                best = i;
            }
        }
        return new Estimate(poses, allSteps(n), ps[best].mean, (System.nanoTime() - t0) / 1e6 / (n - 1));
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

    private static double[] gaussian(double[] mu, double[][] sigma, SplittableRandom r) {
        double[][] l = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                double s = sigma[i][j];
                for (int p = 0; p < j; p++) {
                    s -= l[i][p] * l[j][p];
                }
                if (i == j) {
                    l[i][i] = Math.sqrt(Math.max(s, 1e-18));
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        double[] e = {r.nextGaussian(), r.nextGaussian(), r.nextGaussian()};
        double[] out = mu.clone();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                out[i] += l[i][j] * e[j];
            }
        }
        out[2] = GeometryLinalg.wrap(out[2]);
        return out;
    }

    // ---- pose graph ------------------------------------------------------------------

    /**
     * A relative-pose constraint between two keyframes.
     *
     * @param from the first keyframe
     * @param to the second
     * @param measurement the measured pose of {@code to} in the frame of {@code from}
     * @param information the 3x3 information matrix
     */
    public record Edge(int from, int to, double[] measurement, double[][] information) {
    }

    /**
     * Pose-graph SLAM over keyframes, with loop closures detected from the true trajectory
     * (a stand-in for scan matching) and measured with noise.
     *
     * @param world the world
     * @param every steps between keyframes
     * @param loopRadius the true distance below which two keyframes far apart in time close a
     *     loop
     * @param loopNoise the loop-closure noise {@code (x, y, theta)} standard deviations
     * @param iterations Gauss-Newton iterations
     * @param seed the seed
     * @return the estimate over keyframes, with landmarks placed from the optimised keyframes
     */
    public static Estimate poseGraph(GeometryScenes.RobotWorld world, int every, double loopRadius,
            double[] loopNoise, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = world.steps();
        List<Integer> keys = new ArrayList<>();
        for (int s = 0; s < n; s += every) {
            keys.add(s);
        }
        int m = keys.size();
        double[][] x = new double[m][];
        Estimate dr = deadReckoning(world);
        for (int i = 0; i < m; i++) {
            x[i] = dr.poses()[keys.get(i)].clone();
        }
        long t0 = System.nanoTime();
        List<Edge> edges = new ArrayList<>();
        double[] q = world.odometryNoise();
        for (int i = 1; i < m; i++) {
            double[] rel = GeometryScenes.relative(dr.poses()[keys.get(i - 1)], dr.poses()[keys.get(i)]);
            int len = keys.get(i) - keys.get(i - 1);
            double[][] info = {{1 / (len * sq(q[0]) + len * len * sq(q[2]) * 0.25), 0, 0},
                {0, 1 / (len * sq(q[1]) + len * len * sq(q[2]) * 0.25), 0}, {0, 0, 1 / (len * sq(q[2]))}};
            edges.add(new Edge(i - 1, i, rel, info));
        }
        double[][] li = {{1 / sq(loopNoise[0]), 0, 0}, {0, 1 / sq(loopNoise[1]), 0}, {0, 0, 1 / sq(loopNoise[2])}};
        for (int i = 0; i < m; i++) {
            for (int j = i + 10; j < m; j++) {
                double[] a = world.poses()[keys.get(i)];
                double[] b = world.poses()[keys.get(j)];
                if (Math.hypot(a[0] - b[0], a[1] - b[1]) < loopRadius) {
                    double[] rel = GeometryScenes.relative(a, b);
                    rel = new double[] {rel[0] + loopNoise[0] * r.nextGaussian(), rel[1] + loopNoise[1] * r.nextGaussian(),
                        GeometryLinalg.wrap(rel[2] + loopNoise[2] * r.nextGaussian())};
                    edges.add(new Edge(i, j, rel, li));
                }
            }
        }
        for (int it = 0; it < iterations; it++) {
            double[][] h = new double[3 * m][3 * m];
            double[] g = new double[3 * m];
            for (Edge e : edges) {
                double[] err = edgeError(x[e.from()], x[e.to()], e.measurement());
                double[][] ja = new double[3][3];
                double[][] jb = new double[3][3];
                double eps = 1e-6;
                for (int c = 0; c < 3; c++) {
                    double[] ap = x[e.from()].clone();
                    double[] am = x[e.from()].clone();
                    ap[c] += eps;
                    am[c] -= eps;
                    double[] ep = edgeError(ap, x[e.to()], e.measurement());
                    double[] em = edgeError(am, x[e.to()], e.measurement());
                    double[] bp = x[e.to()].clone();
                    double[] bm = x[e.to()].clone();
                    bp[c] += eps;
                    bm[c] -= eps;
                    double[] fp = edgeError(x[e.from()], bp, e.measurement());
                    double[] fm = edgeError(x[e.from()], bm, e.measurement());
                    for (int a = 0; a < 3; a++) {
                        ja[a][c] = GeometryLinalg.wrap(ep[a] - em[a]) / (2 * eps);
                        jb[a][c] = GeometryLinalg.wrap(fp[a] - fm[a]) / (2 * eps);
                    }
                }
                double[][][] js = {ja, jb};
                int[] ids = {e.from(), e.to()};
                for (int u = 0; u < 2; u++) {
                    double[][] jtw = GeometryLinalg.multiply(GeometryLinalg.transpose(js[u]), e.information());
                    double[] gu = GeometryLinalg.multiply(jtw, err);
                    for (int a = 0; a < 3; a++) {
                        g[3 * ids[u] + a] += gu[a];
                    }
                    for (int v = 0; v < 2; v++) {
                        double[][] blk = GeometryLinalg.multiply(jtw, js[v]);
                        for (int a = 0; a < 3; a++) {
                            for (int b = 0; b < 3; b++) {
                                h[3 * ids[u] + a][3 * ids[v] + b] += blk[a][b];
                            }
                        }
                    }
                }
            }
            for (int a = 0; a < 3; a++) {
                h[a][a] += 1e9;
            }
            double[] d = GeometryLinalg.solveSpd(h, g);
            double change = 0;
            for (int i = 0; i < m; i++) {
                x[i] = new double[] {x[i][0] - d[3 * i], x[i][1] - d[3 * i + 1],
                    GeometryLinalg.wrap(x[i][2] - d[3 * i + 2])};
                change += Math.abs(d[3 * i]) + Math.abs(d[3 * i + 1]);
            }
            if (change < 1e-9) {
                break;
            }
        }
        int k = world.landmarks().length;
        double[][] sum = new double[k][3];
        for (int i = 0; i < m; i++) {
            for (double[] z : world.measurements().get(keys.get(i))) {
                double[] l = place(x[i], z[1], z[2]);
                int id = (int) z[0];
                sum[id][0] += l[0];
                sum[id][1] += l[1];
                sum[id][2]++;
            }
        }
        double[][] lm = new double[k][];
        for (int i = 0; i < k; i++) {
            if (sum[i][2] > 0) {
                lm[i] = new double[] {sum[i][0] / sum[i][2], sum[i][1] / sum[i][2]};
            }
        }
        int[] steps = keys.stream().mapToInt(Integer::intValue).toArray();
        return new Estimate(x, steps, lm, (System.nanoTime() - t0) / 1e6 / (n - 1));
    }

    private static double[] edgeError(double[] a, double[] b, double[] z) {
        double[] rel = GeometryScenes.relative(a, b);
        return new double[] {rel[0] - z[0], rel[1] - z[1], GeometryLinalg.wrap(rel[2] - z[2])};
    }

    /**
     * The number of loop-closure edges {@link #poseGraph} would add.
     *
     * @param world the world
     * @param every steps between keyframes
     * @param loopRadius the loop radius
     * @return the count
     */
    public static int loopClosures(GeometryScenes.RobotWorld world, int every, double loopRadius) {
        int c = 0;
        List<Integer> keys = new ArrayList<>();
        for (int s = 0; s < world.steps(); s += every) {
            keys.add(s);
        }
        for (int i = 0; i < keys.size(); i++) {
            for (int j = i + 10; j < keys.size(); j++) {
                double[] a = world.poses()[keys.get(i)];
                double[] b = world.poses()[keys.get(j)];
                if (Math.hypot(a[0] - b[0], a[1] - b[1]) < loopRadius) {
                    c++;
                }
            }
        }
        return c;
    }
}
