package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A 2-D point robot that must reach a goal straight ahead with a round obstacle in the way,
 * the toy task robot-policy papers use to show multimodal demonstrations (Diffusion Policy,
 * Chi et al., <a href="https://arxiv.org/abs/2303.04137">2303.04137</a>, Fig. 3). The robot
 * starts at {@code (0, -1 +- 0.05)}, the goal is {@code (0, 1)}, and an obstacle of radius
 * 0.3 sits at {@code (0, -0.45)}, right in front of the start. Demonstrators go around it on
 * the left or on the right, half each, so the mean demonstrated action at the start points straight into the obstacle: a policy that
 * regresses the mean (MSE behaviour cloning) crashes; one that models the action
 * distribution (diffusion or flow policies) commits to one side.
 *
 * <p>Observations are the position {@code (x, y)}; actions are a velocity in
 * {@code [-1, 1]^2}, moving the robot by a tenth of it per step. Reaching within 0.15 of the
 * goal gives reward 1 and ends the episode; touching the obstacle or leaving the arena ends
 * it too. Every step costs 0.01, so faster successes return more.
 */
public final class MultimodalReach implements Environment {
    /** The obstacle radius. */
    public static final double OBSTACLE = 0.3;
    /** The height of the obstacle's centre (it sits on the line from start to goal). */
    public static final double OBSTACLE_Y = -0.45;
    /** How close counts as reaching the goal. */
    public static final double REACH = 0.15;
    private static final double SPEED = 0.1;
    private final int horizon;
    private double x;
    private double y;
    private int t;
    private boolean reached;
    private boolean crashed;

    /**
     * Creates the task.
     *
     * @param horizon the step limit
     */
    public MultimodalReach(int horizon) {
        if (horizon < 1) {
            throw new IllegalArgumentException("horizon must be positive");
        }
        this.horizon = horizon;
    }

    @Override
    public Space observationSpace() {
        return new Space.Box(new float[] {-2, -2}, new float[] {2, 2});
    }

    @Override
    public Space actionSpace() {
        return new Space.Box(new float[] {-1, -1}, new float[] {1, 1});
    }

    @Override
    public float[] reset(SplittableRandom random) {
        x = 0;
        y = -1 + random.nextDouble(-0.05, 0.05);
        t = 0;
        reached = false;
        crashed = false;
        return observation();
    }

    @Override
    public Step step(float[] action) {
        double ax = Math.max(-1, Math.min(1, action[0]));
        double ay = Math.max(-1, Math.min(1, action[1]));
        x += SPEED * ax;
        y += SPEED * ay;
        t++;
        reached = Math.hypot(x, y - 1) < REACH;
        crashed = Math.hypot(x, y - OBSTACLE_Y) < OBSTACLE || Math.abs(x) > 2 || Math.abs(y) > 2;
        float reward = (reached ? 1f : 0f) - 0.01f;
        return new Step(observation(), reward, reached || crashed, t >= horizon);
    }

    private float[] observation() {
        return new float[] {(float) x, (float) y};
    }

    /** @return whether the last episode reached the goal */
    public boolean reached() {
        return reached;
    }

    /** @return whether the last episode hit the obstacle or left the arena */
    public boolean crashed() {
        return crashed;
    }

    /**
     * A scripted demonstrator: at the start of each episode it picks the left or the right
     * way round with equal odds. It flies at full speed through a waypoint beside the obstacle,
     * {@code (+-0.6, -0.45)}, to the goal, with Gaussian noise on each action. Both modes start
     * at the same point, so the mean action there points into the obstacle.
     *
     * @param noise the action noise standard deviation
     * @param seed the seed for the side and the noise
     * @return the demonstrator
     */
    public static Agent expert(double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return new Agent() {
            private int side;
            private int waypoint;
            private float[] last;

            @Override
            public void startEpisode() {
                side = r.nextBoolean() ? 1 : -1;
                waypoint = 0;
            }

            @Override
            public float[] act(float[] obs, boolean explore) {
                // a jump in position means a new episode, for callers that never call startEpisode
                if (last == null || Math.hypot(obs[0] - last[0], obs[1] - last[1]) > 0.3) {
                    startEpisode();
                }
                last = obs.clone();
                double wx = waypoint == 0 ? side * 0.6 : 0;
                double wy = waypoint == 0 ? OBSTACLE_Y : 1;
                double dx = wx - obs[0];
                double dy = wy - obs[1];
                double d = Math.hypot(dx, dy);
                if (waypoint == 0 && d < 0.15) {
                    waypoint = 1;
                    dx = -obs[0];
                    dy = 1 - obs[1];
                    d = Math.hypot(dx, dy);
                }
                double s = Math.min(1, d / SPEED) / Math.max(1e-9, d);
                float ax = (float) Math.max(-1, Math.min(1, dx * s + noise * r.nextGaussian()));
                float ay = (float) Math.max(-1, Math.min(1, dy * s + noise * r.nextGaussian()));
                return new float[] {ax, ay};
            }
        };
    }

    /**
     * Plays greedy episodes and counts successes, calling {@link Agent#startEpisode} first.
     *
     * @param agent the policy
     * @param episodes how many episodes
     * @param seed the seed for the starts
     * @return the share of episodes that reached the goal
     */
    public double successRate(Agent agent, int episodes, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int wins = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = reset(random.split());
            agent.startEpisode();
            Step s;
            do {
                s = step(agent.act(obs, false));
                obs = s.observation();
            } while (!s.done());
            wins += reached ? 1 : 0;
        }
        return (double) wins / episodes;
    }

    @Override
    public String toString() {
        return String.format("MultimodalReach(%.2f, %.2f)", x, y);
    }
}
