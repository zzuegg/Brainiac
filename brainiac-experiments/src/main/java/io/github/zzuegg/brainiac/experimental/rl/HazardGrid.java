package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A grid walk with hazard cells, for constrained RL: reaching {@code G} pays +1 and ends the
 * episode, every step pays a small negative reward, and every step that ends on a hazard
 * {@code X} costs 1 (reported by {@link #lastCost()}) without any reward penalty. The short
 * way usually crosses the hazards, so an unconstrained agent takes them; a constrained one
 * must find the detour. {@code S} is the start, {@code #} a wall. Actions: 0 up, 1 right,
 * 2 down, 3 left. Observations are one-hot cells.
 */
public final class HazardGrid implements CostedEnvironment {

    private static final int[] DR = {-1, 0, 1, 0};
    private static final int[] DC = {0, 1, 0, -1};

    private final String[] map;
    private final int rows;
    private final int cols;
    private final int startRow;
    private final int startCol;
    private float stepReward = -0.02f;
    private int maxSteps = 50;
    private int row;
    private int col;
    private int steps;
    private float cost;

    private HazardGrid(String[] map) {
        this.map = map.clone();
        this.rows = map.length;
        this.cols = map[0].length();
        int sr = -1;
        int sc = -1;
        for (int r = 0; r < rows; r++) {
            if (map[r].length() != cols) {
                throw new IllegalArgumentException("rows must have the same length");
            }
            int c = map[r].indexOf('S');
            if (c >= 0) {
                sr = r;
                sc = c;
            }
        }
        if (sr < 0) {
            throw new IllegalArgumentException("the map needs a start S");
        }
        this.startRow = sr;
        this.startCol = sc;
    }

    /**
     * @param rows the map, all rows the same length
     * @return the environment
     */
    public static HazardGrid of(String... rows) {
        return new HazardGrid(rows);
    }

    /**
     * A 3x5 grid whose straight route to the goal crosses one hazard (4 steps) while the safe
     * detour takes 8.
     *
     * @return the environment
     */
    public static HazardGrid detour() {
        return of(
                "S.X.G",
                "..X..",
                ".....");
    }

    /** @param reward the reward of every step @return this environment */
    public HazardGrid stepReward(float reward) {
        this.stepReward = reward;
        return this;
    }

    /** @param steps the most steps per episode @return this environment */
    public HazardGrid maxSteps(int steps) {
        this.maxSteps = steps;
        return this;
    }

    @Override
    public Space observationSpace() {
        return Space.box(rows * cols, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(4);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        row = startRow;
        col = startCol;
        steps = 0;
        cost = 0;
        return observation();
    }

    private float[] observation() {
        float[] x = new float[rows * cols];
        x[row * cols + col] = 1;
        return x;
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        int r = row + DR[a];
        int c = col + DC[a];
        if (r >= 0 && r < rows && c >= 0 && c < cols && map[r].charAt(c) != '#') {
            row = r;
            col = c;
        }
        steps++;
        char cell = map[row].charAt(col);
        cost = cell == 'X' ? 1 : 0;
        boolean goal = cell == 'G';
        return new Step(observation(), stepReward + (goal ? 1 : 0), goal, !goal && steps >= maxSteps);
    }

    @Override
    public float lastCost() {
        return cost;
    }

    @Override
    public String render() {
        StringBuilder b = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                b.append(r == row && c == col ? 'A' : map[r].charAt(c));
            }
            b.append('\n');
        }
        return b.toString();
    }

    @Override
    public String toString() {
        return "HazardGrid(" + rows + "x" + cols + ")";
    }
}
