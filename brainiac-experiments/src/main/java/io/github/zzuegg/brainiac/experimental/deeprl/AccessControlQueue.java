package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * The access-control queuing task of Sutton &amp; Barto (2018, example 10.2), the classic
 * continuing, average-reward problem: {@code k} servers; each step a customer of priority
 * 1, 2, 4 or 8 (equally likely) is at the head of the queue and is accepted (paying its
 * priority, if a server is free) or rejected (paying 0); every busy server becomes free with
 * probability {@code p} per step. The task never terminates. The observation is
 * {@code (free servers / k, priority index / 3)}; actions: 0 reject, 1 accept.
 * {@link #optimalRewardRate} solves it exactly by relative value iteration.
 */
public final class AccessControlQueue implements Environment {

    private static final int[] PRIORITY = {1, 2, 4, 8};

    private final int servers;
    private final double freeProbability;
    private SplittableRandom random = new SplittableRandom(0);
    private int free;
    private int priority;

    /**
     * @param servers servers {@code k} (10 in the book)
     * @param freeProbability the per-step chance a busy server frees (0.06 in the book)
     */
    public AccessControlQueue(int servers, double freeProbability) {
        this.servers = servers;
        this.freeProbability = freeProbability;
    }

    /** Creates the book's task: 10 servers, p = 0.06. */
    public AccessControlQueue() {
        this(10, 0.06);
    }

    @Override
    public Space observationSpace() {
        return Space.box(2, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(2);
    }

    private float[] observation() {
        return new float[] {free / (float) servers, priority / 3f};
    }

    @Override
    public float[] reset(SplittableRandom r) {
        this.random = r;
        free = servers;
        priority = r.nextInt(4);
        return observation();
    }

    @Override
    public Step step(float[] action) {
        float reward = 0;
        if (Math.round(action[0]) == 1 && free > 0) {
            free--;
            reward = PRIORITY[priority];
        }
        int busy = servers - free;
        for (int i = 0; i < busy; i++) {
            if (random.nextDouble() < freeProbability) {
                free++;
            }
        }
        priority = random.nextInt(4);
        return new Step(observation(), reward, false, false);
    }

    /**
     * The optimal long-run reward per step, by relative value iteration on the exact model.
     *
     * @return the optimal average reward
     */
    public double optimalRewardRate() {
        int k = servers;
        double[][] binom = new double[k + 1][k + 1];
        for (int b = 0; b <= k; b++) {
            for (int j = 0; j <= b; j++) {
                binom[b][j] = choose(b, j) * Math.pow(freeProbability, j) * Math.pow(1 - freeProbability, b - j);
            }
        }
        double[][] h = new double[k + 1][4];
        double gain = 0;
        for (int it = 0; it < 5_000; it++) {
            double[] expected = new double[k + 1];
            for (int f = 0; f <= k; f++) {
                for (int p = 0; p < 4; p++) {
                    expected[f] += h[f][p] / 4;
                }
            }
            double[][] nh = new double[k + 1][4];
            for (int f = 0; f <= k; f++) {
                for (int p = 0; p < 4; p++) {
                    double best = Double.NEGATIVE_INFINITY;
                    for (int a = 0; a < 2; a++) {
                        int f1 = f;
                        double r = 0;
                        if (a == 1 && f > 0) {
                            f1 = f - 1;
                            r = PRIORITY[p];
                        }
                        int busy = k - f1;
                        double v = r;
                        for (int j = 0; j <= busy; j++) {
                            v += binom[busy][j] * expected[f1 + j];
                        }
                        best = Math.max(best, v);
                    }
                    nh[f][p] = best;
                }
            }
            gain = nh[0][0] - h[0][0];
            double ref = nh[0][0];
            for (int f = 0; f <= k; f++) {
                for (int p = 0; p < 4; p++) {
                    nh[f][p] -= ref;
                }
            }
            h = nh;
        }
        return gain;
    }

    private static double choose(int n, int r) {
        double c = 1;
        for (int i = 0; i < r; i++) {
            c = c * (n - i) / (i + 1);
        }
        return c;
    }

    /**
     * The long-run reward per step of a greedy agent.
     *
     * @param agent the agent
     * @param steps steps to average over
     * @param seed the seed
     * @return the average reward
     */
    public double rewardRate(Agent agent, int steps, long seed) {
        float[] obs = reset(new SplittableRandom(seed));
        double total = 0;
        for (int t = 0; t < steps; t++) {
            Step s = step(agent.act(obs, false));
            total += s.reward();
            obs = s.observation();
        }
        return total / steps;
    }

    @Override
    public String toString() {
        return "AccessControlQueue(" + servers + ")";
    }
}
