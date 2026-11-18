package io.github.zzuegg.brainiac.experimental.marl;

import java.util.SplittableRandom;

/**
 * A 2-D Ising-style coordination lattice (Yang et al., 2018): {@code side x side} agents on a
 * torus each choose a spin {@code -1} (action 0) or {@code +1} (action 1) every step and receive
 * {@code J/4 sum_j s_i s_j} over their four neighbours. With {@code J = +1} (ferromagnetic) the
 * ground state is all spins aligned; with {@code J = -1} (antiferromagnetic) it is the
 * checkerboard, which a shared constant policy cannot reach.
 *
 * <p>Agent {@code i} observes its own last spin, the mean of its neighbours' last spins and a
 * constant; the global state is every spin. Spins start at random and episodes are truncated
 * after a fixed number of steps.
 */
public final class IsingCoordination implements LocalRewardEnvironment {

    private final int side;
    private final int n;
    private final int maxSteps;
    private final float coupling;
    private final int[] spins;
    private final float[] rewards;
    private int t;

    /**
     * @param side lattice side, so {@code side * side} agents
     * @param maxSteps steps per episode
     * @param coupling {@code J}: +1 to align, -1 to anti-align
     */
    public IsingCoordination(int side, int maxSteps, float coupling) {
        if (side < 2) {
            throw new IllegalArgumentException("side must be at least 2");
        }
        this.side = side;
        this.n = side * side;
        this.maxSteps = maxSteps;
        this.coupling = coupling;
        this.spins = new int[n];
        this.rewards = new float[n];
    }

    @Override
    public int agents() {
        return n;
    }

    @Override
    public int observationSize() {
        return 3;
    }

    @Override
    public int stateSize() {
        return n;
    }

    @Override
    public int actions() {
        return 2;
    }

    @Override
    public int[] neighbours(int agent) {
        int r = agent / side;
        int c = agent % side;
        return new int[] {((r + side - 1) % side) * side + c, ((r + 1) % side) * side + c,
            r * side + (c + side - 1) % side, r * side + (c + 1) % side};
    }

    private float[][] observations() {
        float[][] o = new float[n][];
        for (int i = 0; i < n; i++) {
            float m = 0;
            for (int j : neighbours(i)) {
                m += spins[j];
            }
            o[i] = new float[] {spins[i], m / 4f, 1};
        }
        return o;
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        for (int i = 0; i < n; i++) {
            spins[i] = random.nextBoolean() ? 1 : -1;
        }
        t = 0;
        return observations();
    }

    @Override
    public float[] state() {
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = spins[i];
        }
        return s;
    }

    @Override
    public Step step(int[] actions) {
        for (int i = 0; i < n; i++) {
            spins[i] = actions[i] == 1 ? 1 : -1;
        }
        float team = 0;
        for (int i = 0; i < n; i++) {
            float e = 0;
            for (int j : neighbours(i)) {
                e += spins[i] * spins[j];
            }
            rewards[i] = coupling * e / 4f;
            team += rewards[i];
        }
        t++;
        return new Step(observations(), team / n, false, t >= maxSteps);
    }

    @Override
    public float[] agentRewards() {
        return rewards.clone();
    }

    /**
     * The order parameter of the current spins: the magnetisation {@code |mean s|} for a
     * ferromagnet and the staggered magnetisation {@code |mean (-1)^(r+c) s|} for an
     * antiferromagnet; 1 at the ground state.
     *
     * @return the order parameter in [0, 1]
     */
    public double orderParameter() {
        double m = 0;
        for (int i = 0; i < n; i++) {
            int sign = coupling >= 0 || (i / side + i % side) % 2 == 0 ? 1 : -1;
            m += sign * spins[i];
        }
        return Math.abs(m / n);
    }

    @Override
    public String toString() {
        return "Ising(" + side + "x" + side + ", J=" + coupling + ")";
    }
}
