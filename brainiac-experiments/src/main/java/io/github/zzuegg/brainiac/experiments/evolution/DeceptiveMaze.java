package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.NoveltySearch;

import java.util.SplittableRandom;

/**
 * A deceptive maze for comparing novelty search with objective search: a point robot
 * starts inside a cup whose opening faces away from the goal, so moving toward the goal
 * runs into the cup's roof. Following the distance to the goal gets stuck there; to
 * succeed the robot must first move away from the goal, out through the opening.
 *
 * <p>A genome is an open-loop plan of {@link #STEPS} moves, each a 2-d step clipped to
 * length 1; a move that would cross a wall is not taken. The behaviour descriptor is where
 * the robot ends up, the fitness the negative distance to the goal.
 */
public final class DeceptiveMaze {

    /** The number of moves in a plan. */
    public static final int STEPS = 24;
    /** The start position. */
    public static final double[] START = {0, 3};
    /** The goal position. */
    public static final double[] GOAL = {0, 9};

    /** Walls as segments {x1, y1, x2, y2}: a cup open at the bottom, around the start. */
    private static final double[][] WALLS = {
        {-4, 5, 4, 5},
        {-4, 1, -4, 5},
        {4, 1, 4, 5},
        {-10, -3, 10, -3},
        {-10, -3, -10, 12},
        {10, -3, 10, 12},
        {-10, 12, 10, 12},
    };

    /**
     * The outcome of one search.
     *
     * @param solved whether the goal was reached (within 1)
     * @param generations generations used
     * @param bestDistance the closest any plan got to the goal
     */
    public record Outcome(boolean solved, int generations, double bestDistance) {
    }

    private DeceptiveMaze() {
    }

    /**
     * Where a plan takes the robot.
     *
     * @param plan {@code 2 * STEPS} values, (dx, dy) per move
     * @return the final position
     */
    public static double[] simulate(double[] plan) {
        double x = START[0];
        double y = START[1];
        for (int s = 0; s < STEPS; s++) {
            double dx = plan[2 * s];
            double dy = plan[2 * s + 1];
            double len = Math.hypot(dx, dy);
            if (len > 1) {
                dx /= len;
                dy /= len;
            }
            if (!blocked(x, y, x + dx, y + dy)) {
                x += dx;
                y += dy;
            }
        }
        return new double[] {x, y};
    }

    /**
     * The distance of a position to the goal.
     *
     * @param p the position
     * @return its distance
     */
    public static double distance(double[] p) {
        return Math.hypot(p[0] - GOAL[0], p[1] - GOAL[1]);
    }

    private static boolean blocked(double x1, double y1, double x2, double y2) {
        for (double[] w : WALLS) {
            if (intersects(x1, y1, x2, y2, w[0], w[1], w[2], w[3])) {
                return true;
            }
        }
        return false;
    }

    private static boolean intersects(double ax, double ay, double bx, double by, double cx, double cy, double dx,
            double dy) {
        double d1 = cross(cx, cy, dx, dy, ax, ay);
        double d2 = cross(cx, cy, dx, dy, bx, by);
        double d3 = cross(ax, ay, bx, by, cx, cy);
        double d4 = cross(ax, ay, bx, by, dx, dy);
        return d1 * d2 <= 0 && d3 * d4 <= 0 && !(d1 == 0 && d2 == 0);
    }

    private static double cross(double ox, double oy, double px, double py, double qx, double qy) {
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox);
    }

    /**
     * Searches the maze with novelty search ({@code noveltyWeight} 1) or plain objective
     * search ({@code noveltyWeight} 0).
     *
     * @param noveltyWeight how much novelty counts against fitness
     * @param generations the budget
     * @param seed the seed
     * @return what happened
     */
    public static Outcome search(double noveltyWeight, int generations, long seed) {
        NoveltySearch<double[]> search = NoveltySearch.<double[]>of(
                r -> randomPlan(r),
                DeceptiveMaze::mutate,
                plan -> {
                    double[] end = simulate(plan);
                    return new NoveltySearch.Evaluation(-distance(end), end);
                })
                .population(40).neighbours(10).noveltyWeight(noveltyWeight).seed(seed);
        NoveltySearch.Individual<double[]> best = search.run(generations, ind -> -ind.fitness() < 1);
        return new Outcome(-best.fitness() < 1, search.history().size(), -best.fitness());
    }

    private static double[] randomPlan(SplittableRandom r) {
        double[] p = new double[2 * STEPS];
        for (int i = 0; i < p.length; i++) {
            p[i] = r.nextDouble(-0.3, 0.3);
        }
        return p;
    }

    private static double[] mutate(double[] plan, SplittableRandom r) {
        double[] p = plan.clone();
        for (int i = 0; i < p.length; i++) {
            if (r.nextDouble() < 0.2) {
                p[i] = Math.max(-1, Math.min(1, p[i] + 0.4 * r.nextGaussian()));
            }
        }
        return p;
    }
}
