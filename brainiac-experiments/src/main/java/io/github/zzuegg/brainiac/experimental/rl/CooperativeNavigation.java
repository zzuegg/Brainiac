package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * The cooperative-navigation particle world of the multi-agent particle environments
 * (Lowe et al. 2017): {@code n} agents and {@code n} landmarks in the square
 * {@code [-1, 1]^2}; the team is rewarded for covering every landmark, each step
 * {@code -sum over landmarks of the distance to the nearest agent}, minus a penalty for each
 * pair of agents that collide. The agents must spread out and agree on who goes where without
 * communication.
 *
 * <p>Agents are point masses with damped velocity: {@code v <- 0.75 v + 0.25 a},
 * {@code p <- p + 0.1 v}, with the force {@code a} clipped to the unit box. Discrete actions
 * (the {@link MultiAgentEnvironment} interface) are 0 stay, 1 left, 2 right, 3 down, 4 up;
 * {@link #stepContinuous} takes 2-D forces instead (for MADDPG). An agent observes its own
 * position and velocity, the landmarks and the other agents relative to itself, and a one-hot
 * agent id ({@code 4 + 4n + n} values); the global state holds every position and velocity
 * and the landmarks ({@code 6n} values). Episodes last 25 steps.
 */
public final class CooperativeNavigation implements MultiAgentEnvironment {

    private static final float[][] DIRECTIONS = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private final int n;
    private final int length;
    private final float[][] pos;
    private final float[][] vel;
    private final float[][] landmarks;
    private int t;

    /**
     * @param agents agents and landmarks, 3 in the paper
     * @param length steps per episode
     */
    public CooperativeNavigation(int agents, int length) {
        this.n = agents;
        this.length = length;
        this.pos = new float[agents][2];
        this.vel = new float[agents][2];
        this.landmarks = new float[agents][2];
    }

    /** Creates the standard world: 3 agents, 3 landmarks, 25 steps. */
    public CooperativeNavigation() {
        this(3, 25);
    }

    @Override
    public int agents() {
        return n;
    }

    @Override
    public int observationSize() {
        return 4 + 4 * n + n;
    }

    @Override
    public int stateSize() {
        return 6 * n;
    }

    @Override
    public int actions() {
        return 5;
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        for (int i = 0; i < n; i++) {
            pos[i][0] = (float) random.nextDouble(-1, 1);
            pos[i][1] = (float) random.nextDouble(-1, 1);
            vel[i][0] = 0;
            vel[i][1] = 0;
            landmarks[i][0] = (float) random.nextDouble(-0.8, 0.8);
            landmarks[i][1] = (float) random.nextDouble(-0.8, 0.8);
        }
        t = 0;
        return observations();
    }

    @Override
    public float[] state() {
        float[] s = new float[6 * n];
        for (int i = 0; i < n; i++) {
            s[4 * i] = pos[i][0];
            s[4 * i + 1] = pos[i][1];
            s[4 * i + 2] = vel[i][0];
            s[4 * i + 3] = vel[i][1];
            s[4 * n + 2 * i] = landmarks[i][0];
            s[4 * n + 2 * i + 1] = landmarks[i][1];
        }
        return s;
    }

    private float[][] observations() {
        float[][] out = new float[n][observationSize()];
        for (int i = 0; i < n; i++) {
            float[] o = out[i];
            o[0] = pos[i][0];
            o[1] = pos[i][1];
            o[2] = vel[i][0];
            o[3] = vel[i][1];
            int k = 4;
            for (int l = 0; l < n; l++) {
                o[k++] = landmarks[l][0] - pos[i][0];
                o[k++] = landmarks[l][1] - pos[i][1];
            }
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    o[k++] = pos[j][0] - pos[i][0];
                    o[k++] = pos[j][1] - pos[i][1];
                }
            }
            o[k + i] = 1;
        }
        return out;
    }

    @Override
    public Step step(int[] actions) {
        float[][] forces = new float[n][];
        for (int i = 0; i < n; i++) {
            forces[i] = DIRECTIONS[actions[i]];
        }
        return stepContinuous(forces);
    }

    /**
     * Applies 2-D forces.
     *
     * @param forces one force per agent, clipped to {@code [-1, 1]^2}
     * @return what happened
     */
    public Step stepContinuous(float[][] forces) {
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 2; d++) {
                float a = Math.max(-1, Math.min(1, forces[i][d]));
                vel[i][d] = 0.75f * vel[i][d] + 0.25f * a;
                pos[i][d] = Math.max(-1.5f, Math.min(1.5f, pos[i][d] + 0.1f * vel[i][d]));
            }
        }
        t++;
        float reward = 0;
        for (float[] l : landmarks) {
            reward -= nearest(l);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) < 0.1) {
                    reward -= 0.5f;
                }
            }
        }
        return new Step(observations(), reward, false, t >= length);
    }

    private float nearest(float[] l) {
        double best = Double.MAX_VALUE;
        for (float[] p : pos) {
            best = Math.min(best, Math.hypot(p[0] - l[0], p[1] - l[1]));
        }
        return (float) best;
    }

    /** @return the fraction of landmarks with an agent within 0.15 */
    public double coverage() {
        int covered = 0;
        for (float[] l : landmarks) {
            covered += nearest(l) < 0.15 ? 1 : 0;
        }
        return covered / (double) n;
    }

    @Override
    public String toString() {
        return "cooperative navigation (" + n + " agents)";
    }
}
