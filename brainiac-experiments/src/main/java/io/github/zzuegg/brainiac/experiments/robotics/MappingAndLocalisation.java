package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.experimental.data.GeometryScenes;
import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.geometry.OccupancyGrid;
import io.github.zzuegg.brainiac.experimental.geometry.Slam;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 12 items 12.3.9 and 12.3.10.
 *
 * <ul>
 *   <li>{@link #slam}: dead reckoning, EKF localisation and Monte Carlo localisation with the
 *       true map, EKF-SLAM (known and nearest-neighbour association), FastSLAM 2.0 and pose-graph
 *       SLAM on a robot driving two laps through a landmark field, by absolute trajectory
 *       error, landmark RMSE and time per step;</li>
 *   <li>{@link #mapping}: occupancy grids of a room map from a simulated lidar at known poses
 *       (Gaussian range noise, 5% spurious returns, and optionally specular dropouts at grazing
 *       incidence), comparing hit counting, the cone model with and without clamping and a
 *       learned inverse sensor model by accuracy and AUROC on seen cells;
 *       {@link #recovery} counts the scans each map needs to show an obstacle that appears
 *       after the room was mapped empty.</li>
 * </ul>
 */
public final class MappingAndLocalisation {

    private MappingAndLocalisation() {
    }

    /**
     * A SLAM row.
     *
     * @param method the method
     * @param trajectoryError the position RMSE
     * @param landmarkError the landmark RMSE (NaN without a map)
     * @param millisPerStep time per step
     */
    public record SlamRow(String method, double trajectoryError, double landmarkError, double millisPerStep) {
    }

    /**
     * Compares SLAM methods.
     *
     * @param landmarks the number of landmarks
     * @param steps the number of steps (two laps)
     * @param seed the seed
     * @return rows
     */
    public static List<SlamRow> slam(int landmarks, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GeometryScenes.RobotWorld w = GeometryScenes.robotWorld(landmarks, steps, 2, new double[] {0.05, 0.02, 0.01},
            0.1, 0.02, 6, r);
        List<SlamRow> rows = new ArrayList<>();
        rows.add(row("dead reckoning", Slam.deadReckoning(w), w));
        rows.add(row("EKF localisation, known map (oracle)", Slam.ekfLocalisation(w), w));
        rows.add(row("MCL, known map, 100 particles", Slam.particleLocalisation(w, 100, seed + 1), w));
        rows.add(row("EKF-SLAM known association", Slam.ekfSlam(w, true), w));
        rows.add(row("EKF-SLAM nearest neighbour", Slam.ekfSlam(w, false), w));
        rows.add(row("FastSLAM 2.0, 20 particles", Slam.fastSlam2(w, 20, seed + 2), w));
        rows.add(row("pose graph (" + Slam.loopClosures(w, 5, 1.0) + " loop closures)",
            Slam.poseGraph(w, 5, 1.0, new double[] {0.05, 0.05, 0.01}, 10, seed + 3), w));
        return rows;
    }

    private static SlamRow row(String name, Slam.Estimate e, GeometryScenes.RobotWorld w) {
        return new SlamRow(name, e.trajectoryError(w), e.landmarkError(w), e.millisPerStep());
    }

    /** A simulated lidar on a true grid. */
    private record Lidar(int beams, double maxRange, double noise, double spurious, boolean specular) {

        double[] angles() {
            double[] a = new double[beams];
            for (int i = 0; i < beams; i++) {
                a[i] = 2 * Math.PI * i / beams;
            }
            return a;
        }

        double[] scan(PlanningMaps.Grid g, double x, double y, SplittableRandom r) {
            double[] a = angles();
            double[] out = new double[beams];
            for (int i = 0; i < beams; i++) {
                out[i] = cast(g, x, y, a[i], r);
            }
            return out;
        }

        double cast(PlanningMaps.Grid g, double x, double y, double angle, SplittableRandom r) {
            double dx = Math.cos(angle);
            double dy = Math.sin(angle);
            int cx = (int) Math.floor(x);
            int cy = (int) Math.floor(y);
            double step = 0.02;
            for (double t = step; t < maxRange; t += step) {
                int nx = (int) Math.floor(x + t * dx);
                int ny = (int) Math.floor(y + t * dy);
                if (nx == cx && ny == cy) {
                    continue;
                }
                boolean vertical = nx != cx;
                cx = nx;
                cy = ny;
                if (!g.free(nx, ny)) {
                    double incidence = Math.abs(vertical ? dx : dy);
                    if (specular && incidence < 0.4 && r.nextDouble() < 0.8) {
                        return maxRange;
                    }
                    if (r.nextDouble() < spurious) {
                        return maxRange * r.nextDouble();
                    }
                    return Math.max(0.1, Math.min(maxRange, t + noise * r.nextGaussian()));
                }
            }
            return r.nextDouble() < spurious ? maxRange * r.nextDouble() : maxRange;
        }
    }

    private static double[] freeSpot(PlanningMaps.Grid g, SplittableRandom r) {
        while (true) {
            int x = 1 + r.nextInt(g.width() - 2);
            int y = 1 + r.nextInt(g.height() - 2);
            if (g.free(x, y)) {
                return new double[] {x + 0.2 + 0.6 * r.nextDouble(), y + 0.2 + 0.6 * r.nextDouble()};
            }
        }
    }

    /**
     * A mapping row.
     *
     * @param sensor the noise model
     * @param method the inverse model
     * @param accuracy the share of seen cells classified correctly
     * @param auroc the AUROC on seen cells
     */
    public record MappingRow(String sensor, String method, double accuracy, double auroc) {
    }

    private static OccupancyGrid[] maps(int side, SimpleMlp learned) {
        return new OccupancyGrid[] {
            new OccupancyGrid(side, side, OccupancyGrid.Model.HIT_COUNT, 0, 0, -1e9, 1e9),
            new OccupancyGrid(side, side, OccupancyGrid.Model.CONE, 0.85, -0.4, -1e9, 1e9),
            new OccupancyGrid(side, side, OccupancyGrid.Model.CONE, 0.85, -0.4, -2, 3.5),
            new OccupancyGrid(side, side, OccupancyGrid.Model.LEARNED, 0, 0, -2, 3.5).learned(learned)};
    }

    private static final String[] NAMES = {"hit counting", "cone, no clamping", "cone, clamped", "learned, clamped"};

    private static SimpleMlp trainModel(Lidar lidar, int side, int scans, SplittableRandom r, long seed) {
        List<float[]> feats = new ArrayList<>();
        List<Boolean> occ = new ArrayList<>();
        int occupied = 0;
        for (int m = 0; m < 3; m++) {
            PlanningMaps.Grid g = PlanningMaps.rooms(side, side, 2, 2, r);
            double[] angles = lidar.angles();
            for (int s = 0; s < scans; s++) {
                double[] p = freeSpot(g, r);
                double[] z = lidar.scan(g, p[0], p[1], r);
                for (int i = 0; i < angles.length; i++) {
                    double prev = z[(i + angles.length - 1) % angles.length];
                    double next = z[(i + 1) % angles.length];
                    for (Object[] c : OccupancyGrid.beamCells(p[0], p[1], angles[i], z[i], prev, next,
                            lidar.maxRange(), side, side)) {
                        feats.add((float[]) c[2]);
                        boolean o = g.blocked()[(Integer) c[0]];
                        occ.add(o);
                        occupied += o ? 1 : 0;
                    }
                }
            }
        }
        double prior = Math.max(0.01, occupied / (double) occ.size());
        return OccupancyGrid.fitInverseModel(feats, occ, 400, prior, seed);
    }

    /**
     * Maps a room grid with each inverse model under a plain and a specular sensor.
     *
     * @param side the map side in cells
     * @param scans scans per map
     * @param seed the seed
     * @return rows
     */
    public static List<MappingRow> mapping(int side, int scans, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<MappingRow> rows = new ArrayList<>();
        for (boolean specular : new boolean[] {false, true}) {
            Lidar lidar = new Lidar(48, 12, 0.1, 0.05, specular);
            SimpleMlp model = trainModel(lidar, side, scans, r, seed + (specular ? 1 : 0));
            PlanningMaps.Grid g = PlanningMaps.rooms(side, side, 2, 2, r);
            OccupancyGrid[] maps = maps(side, model);
            for (int s = 0; s < scans; s++) {
                double[] p = freeSpot(g, r);
                double[] z = lidar.scan(g, p[0], p[1], r);
                for (OccupancyGrid m : maps) {
                    m.update(p[0], p[1], lidar.angles(), z, lidar.maxRange());
                }
            }
            for (int k = 0; k < maps.length; k++) {
                double[] sc = maps[k].score(g.blocked(), null);
                rows.add(new MappingRow(specular ? "specular" : "plain", NAMES[k], sc[0], sc[1]));
            }
        }
        return rows;
    }

    /**
     * A recovery row.
     *
     * @param method the inverse model
     * @param scans scans after the change until 7 of the 9 new obstacle cells (the centre is occluded) read occupied
     *     (the budget plus one if never)
     */
    public record RecoveryRow(String method, int scans) {
    }

    /**
     * Maps a room empty, then adds a 3x3 obstacle and counts the scans each map needs.
     *
     * @param side the map side
     * @param before scans before the change
     * @param budget the most scans after the change
     * @param seed the seed
     * @return rows
     */
    public static List<RecoveryRow> recovery(int side, int before, int budget, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Lidar lidar = new Lidar(48, 12, 0.1, 0.05, false);
        SimpleMlp model = trainModel(lidar, side, 30, r, seed);
        PlanningMaps.Grid g = PlanningMaps.rooms(side, side, 2, 2, r);
        int bx = side / 4;
        int by = side / 4;
        for (int y = by - 1; y <= by + 1; y++) {
            for (int x = bx - 1; x <= bx + 1; x++) {
                g.blocked()[y * side + x] = false;
            }
        }
        OccupancyGrid[] maps = maps(side, model);
        List<double[]> spots = new ArrayList<>();
        while (spots.size() < 20) {
            double[] p = freeSpot(g, r);
            double d = Math.hypot(p[0] - bx - 0.5, p[1] - by - 0.5);
            if (d > 2.5 && d < side / 4.0) {
                spots.add(p);
            }
        }
        for (int s = 0; s < before; s++) {
            double[] p = spots.get(r.nextInt(spots.size()));
            double[] z = lidar.scan(g, p[0], p[1], r);
            for (OccupancyGrid m : maps) {
                m.update(p[0], p[1], lidar.angles(), z, lidar.maxRange());
            }
        }
        PlanningMaps.Grid changed = g.copy();
        int[] cells = new int[9];
        int c = 0;
        for (int y = by - 1; y <= by + 1; y++) {
            for (int x = bx - 1; x <= bx + 1; x++) {
                changed.blocked()[y * side + x] = true;
                cells[c++] = y * side + x;
            }
        }
        int[] found = new int[maps.length];
        java.util.Arrays.fill(found, budget + 1);
        for (int s = 1; s <= budget; s++) {
            double[] p = spots.get(r.nextInt(spots.size()));
            double[] z = lidar.scan(changed, p[0], p[1], r);
            for (int k = 0; k < maps.length; k++) {
                maps[k].update(p[0], p[1], lidar.angles(), z, lidar.maxRange());
                if (found[k] > budget) {
                    int occupied = 0;
                    for (int cell : cells) {
                        if (maps[k].probability(cell) > 0.5) {
                            occupied++;
                        }
                    }
                    if (occupied >= 7) {
                        found[k] = s;
                    }
                }
            }
        }
        List<RecoveryRow> rows = new ArrayList<>();
        for (int k = 0; k < maps.length; k++) {
            rows.add(new RecoveryRow(NAMES[k], found[k]));
        }
        return rows;
    }
}
