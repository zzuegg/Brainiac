package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * Hard-exploration chains where dithering fails:
 *
 * <ul>
 *   <li>{@link #deepSea} (bsuite, Osband et al. 2019,
 *       <a href="https://arxiv.org/abs/1908.03568">arXiv:1908.03568</a>): an {@code N x N} grid
 *       entered at the top-left; every step moves one row down and one column left or right,
 *       and which action index means "right" is drawn at random per cell. Moving right costs
 *       {@code 0.01 / N}; only reaching the bottom-right corner pays 1, so a dithering agent
 *       finds it with probability about {@code 2^-N} per episode. Observations are the one-hot
 *       position ({@code N^2} values); episodes last {@code N} steps.</li>
 *   <li>{@link #riverSwim} (Strehl &amp; Littman, 2008): a chain of states; swimming left always
 *       works and pays 0.005 at the left bank, swimming right succeeds with probability 0.35
 *       (0.6 stay, 0.05 drift back) and pays 1 at the right end. Observations are one-hot;
 *       episodes last {@code length} steps.</li>
 * </ul>
 */
public final class DeepSeaChain implements Environment {

    private final boolean sea;
    private final int n;
    private final int length;
    private final boolean[][] rightIsOne;
    private final double cost;
    private SplittableRandom random = new SplittableRandom(0);
    private int row;
    private int col;
    private int t;

    private DeepSeaChain(boolean sea, int n, int length, long seed, double cost) {
        this.sea = sea;
        this.n = n;
        this.length = length;
        this.cost = cost;
        this.rightIsOne = new boolean[n][n];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rightIsOne[i][j] = r.nextBoolean();
            }
        }
    }

    /**
     * @param size the side {@code N}, 10 to 30 in the experiments
     * @param mappingSeed the seed of the per-cell action mapping
     * @return the deep sea
     */
    public static DeepSeaChain deepSea(int size, long mappingSeed) {
        return new DeepSeaChain(true, size, size, mappingSeed, 0.01 / size);
    }

    /**
     * @param states the chain length (6 in the paper)
     * @param length steps per episode
     * @return the river swim
     */
    public static DeepSeaChain riverSwim(int states, int length) {
        return new DeepSeaChain(false, states, length, 0, 0);
    }

    /** @return the return of the optimal policy (deep sea; for river swim an upper bound of 1 per step) */
    public double optimalReturn() {
        return sea ? 1 - cost * (n - 1) : length;
    }

    /** @return whether the agent sits at the rewarding end (deep sea: bottom-right corner) */
    public boolean atGoal() {
        return sea ? row == n - 1 && col == n - 1 : col == n - 1;
    }

    /** @return the side or chain length */
    public int size() {
        return n;
    }

    @Override
    public Space observationSpace() {
        return Space.box(sea ? n * n : n, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(2);
    }

    private float[] observation() {
        float[] o = new float[sea ? n * n : n];
        o[sea ? Math.min(row, n - 1) * n + col : col] = 1;
        return o;
    }

    @Override
    public float[] reset(SplittableRandom random) {
        this.random = random;
        row = 0;
        col = 0;
        t = 0;
        return observation();
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        t++;
        float reward = 0;
        if (sea) {
            boolean right = (a == 1) == rightIsOne[row][col];
            if (right) {
                reward -= (float) cost;
                col = Math.min(n - 1, col + 1);
            } else {
                col = Math.max(0, col - 1);
            }
            row++;
            boolean done = row >= n - 1 || t >= length;
            if (done && col == n - 1) {
                reward += 1;
            }
            row = Math.min(row, n - 1);
            return new Step(observation(), reward, done, false);
        }
        if (a == 0) {
            if (col == 0) {
                reward = 0.005f;
            }
            col = Math.max(0, col - 1);
        } else {
            double u = random.nextDouble();
            if (col == n - 1) {
                reward = u < 0.95 ? 1 : 0;
                if (u >= 0.95) {
                    col--;
                }
            } else if (u < 0.35) {
                col++;
            } else if (u >= 0.95) {
                col = Math.max(0, col - 1);
            }
        }
        return new Step(observation(), reward, false, t >= length);
    }

    @Override
    public String toString() {
        return sea ? "DeepSea(" + n + ")" : "RiverSwim(" + n + ")";
    }
}
