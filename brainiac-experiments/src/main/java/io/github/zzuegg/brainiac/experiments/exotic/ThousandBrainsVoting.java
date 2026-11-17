package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.SensorimotorColumn;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Object recognition by sensorimotor columns that vote (Hawkins et al.,
 * <a href="https://doi.org/10.3389/fncir.2017.00081">Front. Neural Circuits 2017</a>, figure 4):
 * objects are small grids of features drawn from a tiny shared alphabet, so one sensation
 * rarely identifies an object. A single {@link SensorimotorColumn} has to touch the object
 * several times, path-integrating between touches; several columns touching different
 * places at once and voting after every sensation need far fewer movements.
 */
public final class ThousandBrainsVoting {
    private final int side;
    private final int[][][] objects;

    private ThousandBrainsVoting(int count, int side, int features, long seed) {
        this.side = side;
        SplittableRandom r = new SplittableRandom(seed);
        this.objects = new int[count][side][side];
        for (int[][] o : objects) {
            for (int[] row : o) {
                for (int x = 0; x < side; x++) {
                    row[x] = r.nextInt(features);
                }
            }
        }
    }

    /**
     * Creates an object library.
     *
     * @param count the number of objects
     * @param side the side of each object's feature grid
     * @param features the alphabet size
     * @param seed the seed
     * @return the experiment
     */
    public static ThousandBrainsVoting of(int count, int side, int features, long seed) {
        return new ThousandBrainsVoting(count, side, features, seed);
    }

    /**
     * Creates columns that have each learned every object.
     *
     * @param n the number of columns
     * @return the columns
     */
    public List<SensorimotorColumn> columns(int n) {
        List<SensorimotorColumn> out = new ArrayList<>();
        for (int c = 0; c < n; c++) {
            SensorimotorColumn col = SensorimotorColumn.create();
            for (int o = 0; o < objects.length; o++) {
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        col.learn(o, x, y, objects[o][y][x]);
                    }
                }
            }
            out.add(col);
        }
        return out;
    }

    /**
     * Recognizes one object.
     *
     * @param columns the columns
     * @param object the presented object
     * @param maxSteps the sensation limit
     * @param r the random source for sensor placement and movement
     * @return the number of sensations until the columns agree on one object, or
     *         {@code -1} if they settled on a wrong one, or {@code maxSteps + 1} if undecided
     */
    public int recognize(List<SensorimotorColumn> columns, int object, int maxSteps, SplittableRandom r) {
        int n = columns.size();
        int[][] pos = new int[n][2];
        for (int c = 0; c < n; c++) {
            columns.get(c).reset();
            pos[c][0] = r.nextInt(side);
            pos[c][1] = r.nextInt(side);
        }
        for (int step = 1; step <= maxSteps; step++) {
            for (int c = 0; c < n; c++) {
                columns.get(c).sense(objects[object][pos[c][1]][pos[c][0]]);
            }
            Set<Integer> agreed = SensorimotorColumn.vote(columns);
            if (agreed.size() == 1) {
                return agreed.contains(object) ? step : -1;
            }
            if (agreed.isEmpty()) {
                return -1;
            }
            for (int c = 0; c < n; c++) {
                int nx;
                int ny;
                do {
                    nx = r.nextInt(side);
                    ny = r.nextInt(side);
                } while (nx == pos[c][0] && ny == pos[c][1]);
                columns.get(c).move(nx - pos[c][0], ny - pos[c][1]);
                pos[c][0] = nx;
                pos[c][1] = ny;
            }
        }
        return maxSteps + 1;
    }

    /**
     * Average sensations to recognition over every object, several times each.
     *
     * @param columnCount how many voting columns
     * @param repeats presentations per object
     * @param seed the seed
     * @return {@code {mean sensations, share recognized correctly}}
     */
    public double[] meanSensations(int columnCount, int repeats, long seed) {
        List<SensorimotorColumn> columns = columns(columnCount);
        SplittableRandom r = new SplittableRandom(seed);
        double steps = 0;
        int right = 0;
        int total = 0;
        for (int rep = 0; rep < repeats; rep++) {
            for (int o = 0; o < objects.length; o++) {
                int s = recognize(columns, o, 30, r);
                total++;
                if (s > 0 && s <= 30) {
                    right++;
                    steps += s;
                }
            }
        }
        return new double[] {steps / Math.max(1, right), (double) right / total};
    }
}
