package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A grid with objects of a few types (Barreto et al.'s transfer setting): stepping onto an
 * object picks it up and ends the episode. The features of a step are a one-hot of the
 * picked object's type plus a constant 1 for the step itself, so a task is a weight per
 * object type plus a step cost: {@code r = phi . w}. Map characters: {@code S} start,
 * {@code #} wall, {@code .} floor, digits {@code 0}–{@code 9} objects of that type.
 * Observations are one-hot cells; actions 0 up, 1 right, 2 down, 3 left.
 */
public final class ObjectGrid implements FeatureEnvironment {

    private static final int[] DR = {-1, 0, 1, 0};
    private static final int[] DC = {0, 1, 0, -1};

    private final String[] map;
    private final int rows;
    private final int cols;
    private final int types;
    private final float[] weights;
    private int startRow;
    private int startCol;
    private int maxSteps = 50;
    private int row;
    private int col;
    private int steps;
    private float[] features;

    /**
     * @param map the map rows
     * @param weights one weight per object type, then the weight of the step feature
     */
    public ObjectGrid(String[] map, float[] weights) {
        this.map = map.clone();
        this.rows = map.length;
        this.cols = map[0].length();
        int maxType = -1;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                char ch = map[r].charAt(c);
                if (ch == 'S') {
                    startRow = r;
                    startCol = c;
                } else if (Character.isDigit(ch)) {
                    maxType = Math.max(maxType, ch - '0');
                }
            }
        }
        this.types = maxType + 1;
        if (weights.length != types + 1) {
            throw new IllegalArgumentException("need " + (types + 1) + " weights (one per type and the step)");
        }
        this.weights = weights.clone();
    }

    /**
     * @param weights the new task's weights
     * @return the same grid with another task
     */
    public ObjectGrid withWeights(float[] weights) {
        return new ObjectGrid(map, weights).maxSteps(maxSteps);
    }

    /** @param steps the most steps per episode @return this environment */
    public ObjectGrid maxSteps(int steps) {
        this.maxSteps = steps;
        return this;
    }

    @Override
    public int features() {
        return types + 1;
    }

    @Override
    public float[] weights() {
        return weights.clone();
    }

    @Override
    public float[] lastFeatures() {
        return features.clone();
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
        features = new float[types + 1];
        features[types] = 1;
        char ch = map[row].charAt(col);
        boolean picked = Character.isDigit(ch);
        if (picked) {
            features[ch - '0'] = 1;
        }
        float reward = 0;
        for (int i = 0; i < features.length; i++) {
            reward += features[i] * weights[i];
        }
        return new Step(observation(), reward, picked, !picked && steps >= maxSteps);
    }

    @Override
    public String toString() {
        return "ObjectGrid(" + rows + "x" + cols + ", " + types + " types)";
    }
}
