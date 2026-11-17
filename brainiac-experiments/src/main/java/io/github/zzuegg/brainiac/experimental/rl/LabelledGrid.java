package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A grid walk whose cells carry event labels, for {@link RewardMachine} tasks such as the
 * office world of Toro Icarte et al.: lowercase letters are labelled cells (the label is the
 * letter), {@code S} the start, {@code #} walls, {@code .} floor. The environment itself
 * gives no reward and never terminates on its own (only by the step limit); the reward
 * machine decides both. Observations are one-hot cells; actions 0 up, 1 right, 2 down,
 * 3 left.
 */
public final class LabelledGrid implements LabelledEnvironment {

    private static final int[] DR = {-1, 0, 1, 0};
    private static final int[] DC = {0, 1, 0, -1};

    private final String[] map;
    private final int rows;
    private final int cols;
    private int startRow;
    private int startCol;
    private int maxSteps = 100;
    private int row;
    private int col;
    private int steps;

    /**
     * @param map the rows, all the same length
     */
    public LabelledGrid(String... map) {
        this.map = map.clone();
        this.rows = map.length;
        this.cols = map[0].length();
        for (int r = 0; r < rows; r++) {
            int c = map[r].indexOf('S');
            if (c >= 0) {
                startRow = r;
                startCol = c;
            }
        }
    }

    /** @param steps the most steps per episode @return this environment */
    public LabelledGrid maxSteps(int steps) {
        this.maxSteps = steps;
        return this;
    }

    /** @return the number of cells */
    public int cells() {
        return rows * cols;
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
        return observation();
    }

    private float[] observation() {
        float[] o = new float[rows * cols];
        o[row * cols + col] = 1;
        return o;
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
        return new Step(observation(), 0, false, steps >= maxSteps);
    }

    @Override
    public String label() {
        char ch = map[row].charAt(col);
        return Character.isLowerCase(ch) ? String.valueOf(ch) : "";
    }

    @Override
    public String toString() {
        return "LabelledGrid(" + rows + "x" + cols + ")";
    }
}
