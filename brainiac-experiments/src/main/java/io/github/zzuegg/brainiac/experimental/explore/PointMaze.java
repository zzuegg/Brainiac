package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.experimental.rl.GoalEnvironment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A continuous 2-D point-mass maze in {@code [-1, 1]^2}, goal-conditioned: the observation is
 * {@code (x, y, goal x, goal y)}, the action a displacement in {@code [-1, 1]^2} scaled by
 * {@code speed}, and moves into a wall are cancelled. Layouts: {@link Layout#OPEN},
 * {@link Layout#U_MAZE} (a wall across the middle from the left edge, so the goal at the top
 * left must be reached around the right side) and {@link Layout#FOUR_ROOMS}. The reward is
 * sparse: -1 per step until the agent is within {@code tolerance} of the goal, which ends the
 * episode (or, with {@link #dense}, minus the distance to the goal).
 */
public final class PointMaze implements GoalEnvironment {

    /** The wall layout. */
    public enum Layout {
        /** no walls */
        OPEN,
        /** one wall from the left edge to {@code x = 0.4} across the middle */
        U_MAZE,
        /** two crossing walls with four doorways */
        FOUR_ROOMS
    }

    private record Wall(double x0, double y0, double x1, double y1) {
        boolean contains(double x, double y) {
            return x >= x0 && x <= x1 && y >= y0 && y <= y1;
        }
    }

    private final Layout layout;
    private final List<Wall> walls;
    private final int maxSteps;
    private final float[] start;
    private final float[] fixedGoal;
    private float speed = 0.1f;
    private float tolerance = 0.15f;
    private boolean dense;
    private float x;
    private float y;
    private float[] goal;
    private int t;

    /**
     * @param layout the walls
     * @param maxSteps steps per episode
     * @param start the start position
     * @param goal the goal, or null for a random free goal per episode
     */
    public PointMaze(Layout layout, int maxSteps, float[] start, float[] goal) {
        this.layout = layout;
        this.maxSteps = maxSteps;
        this.start = start.clone();
        this.fixedGoal = goal == null ? null : goal.clone();
        this.walls = switch (layout) {
            case OPEN -> List.of();
            case U_MAZE -> List.of(new Wall(-1, -0.15, 0.4, 0.15));
            case FOUR_ROOMS -> List.of(new Wall(-0.05, -1, 0.05, -0.55), new Wall(-0.05, -0.25, 0.05, 0.55),
                    new Wall(-0.05, 0.85, 0.05, 1), new Wall(-1, -0.05, -0.6, 0.05), new Wall(-0.3, -0.05, 0.6, 0.05),
                    new Wall(0.9, -0.05, 1, 0.05));
        };
    }

    /**
     * The U-maze with a start at the bottom left and the goal at the top left, behind the wall.
     *
     * @param maxSteps steps per episode
     * @return the maze
     */
    public static PointMaze uMaze(int maxSteps) {
        return new PointMaze(Layout.U_MAZE, maxSteps, new float[] {-0.7f, -0.7f}, new float[] {-0.7f, 0.7f});
    }

    /** @param on use the dense reward {@code -distance} @return this maze */
    public PointMaze dense(boolean on) {
        this.dense = on;
        return this;
    }

    /** @param step the largest displacement per step @return this maze */
    public PointMaze speed(float step) {
        this.speed = step;
        return this;
    }

    /** @return the layout */
    public Layout layout() {
        return layout;
    }

    /** @return the current position */
    public float[] position() {
        return new float[] {x, y};
    }

    private boolean free(double px, double py) {
        if (px < -1 || px > 1 || py < -1 || py > 1) {
            return false;
        }
        for (Wall w : walls) {
            if (w.contains(px, py)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int goalSize() {
        return 2;
    }

    @Override
    public float[] achievedGoal(float[] observation) {
        return new float[] {observation[0], observation[1]};
    }

    @Override
    public float reward(float[] achieved, float[] desired) {
        double d = Math.hypot(achieved[0] - desired[0], achieved[1] - desired[1]);
        if (dense) {
            return (float) -d;
        }
        return d < tolerance ? 0 : -1;
    }

    @Override
    public boolean success(float[] achieved, float[] desired) {
        return Math.hypot(achieved[0] - desired[0], achieved[1] - desired[1]) < tolerance;
    }

    @Override
    public Space observationSpace() {
        return Space.box(4, -1, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.box(2, -1, 1);
    }

    private float[] observation() {
        return new float[] {x, y, goal[0], goal[1]};
    }

    @Override
    public float[] reset(SplittableRandom random) {
        x = start[0];
        y = start[1];
        t = 0;
        if (fixedGoal != null) {
            goal = fixedGoal.clone();
        } else {
            do {
                goal = new float[] {(float) random.nextDouble(-0.9, 0.9), (float) random.nextDouble(-0.9, 0.9)};
            } while (!free(goal[0], goal[1]));
        }
        return observation();
    }

    @Override
    public Step step(float[] action) {
        t++;
        float ax = Math.max(-1, Math.min(1, action[0])) * speed;
        float ay = Math.max(-1, Math.min(1, action[1])) * speed;
        // move in four sub-steps so thin walls cannot be jumped
        for (int k = 0; k < 4; k++) {
            float nx = x + ax / 4;
            float ny = y + ay / 4;
            if (free(nx, ny)) {
                x = nx;
                y = ny;
            } else if (free(nx, y)) {
                x = nx;
            } else if (free(x, ny)) {
                y = ny;
            }
        }
        float[] obs = observation();
        float[] achieved = {x, y};
        boolean ok = success(achieved, goal);
        return new Step(obs, reward(achieved, goal), ok, !ok && t >= maxSteps);
    }

    @Override
    public String toString() {
        return "PointMaze(" + layout + ")";
    }
}
