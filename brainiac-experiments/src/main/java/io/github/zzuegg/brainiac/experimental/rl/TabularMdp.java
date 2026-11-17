package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A finite Markov decision process with known dynamics: transition probabilities
 * {@code P[s][a][s']}, a state reward, a start distribution and terminal states. Methods
 * that need the model, such as value iteration, soft (maximum-entropy) value iteration and
 * {@link MaxEntIrl}, read it directly; as an {@link Environment} it samples episodes with
 * one-hot state observations, so model-free agents can run on the same problem.
 */
public final class TabularMdp implements Environment {

    private final int states;
    private final int actions;
    private final double[][][] transitions;
    private final double[] reward;
    private final double[] start;
    private final boolean[] terminal;
    private final int maxSteps;
    private SplittableRandom random = new SplittableRandom(0);
    private int state;
    private int steps;
    private boolean over = true;

    private TabularMdp(double[][][] transitions, double[] reward, double[] start, boolean[] terminal, int maxSteps) {
        this.states = reward.length;
        this.actions = transitions[0].length;
        this.transitions = transitions;
        this.reward = reward;
        this.start = start;
        this.terminal = terminal;
        this.maxSteps = maxSteps;
    }

    /**
     * @param transitions {@code P[s][a][s']}
     * @param reward the reward for entering each state
     * @param start the start distribution
     * @param terminal which states end the episode
     * @param maxSteps the episode length limit
     * @return the MDP
     */
    public static TabularMdp of(double[][][] transitions, double[] reward, double[] start, boolean[] terminal,
            int maxSteps) {
        return new TabularMdp(transitions, reward.clone(), start.clone(), terminal.clone(), maxSteps);
    }

    /**
     * An open grid with the four moves (0 up, 1 right, 2 down, 3 left), a chance of slipping
     * to a random move, the start in the top-left corner and a rewarding terminal goal.
     *
     * @param rows grid rows
     * @param cols grid columns
     * @param goalRow the goal's row
     * @param goalCol the goal's column
     * @param slip the chance a move goes in a random direction
     * @param maxSteps the episode length limit
     * @return the MDP, with reward 1 for entering the goal and -0.01 elsewhere
     */
    public static TabularMdp grid(int rows, int cols, int goalRow, int goalCol, double slip, int maxSteps) {
        int n = rows * cols;
        int[][] moves = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        double[][][] p = new double[n][4][n];
        for (int s = 0; s < n; s++) {
            for (int a = 0; a < 4; a++) {
                for (int m = 0; m < 4; m++) {
                    double prob = (m == a ? 1 - slip : 0) + slip / 4;
                    int r = Math.max(0, Math.min(rows - 1, s / cols + moves[m][0]));
                    int c = Math.max(0, Math.min(cols - 1, s % cols + moves[m][1]));
                    p[s][a][r * cols + c] += prob;
                }
            }
        }
        double[] reward = new double[n];
        java.util.Arrays.fill(reward, -0.01);
        int goal = goalRow * cols + goalCol;
        reward[goal] = 1;
        double[] start = new double[n];
        start[0] = 1;
        boolean[] terminal = new boolean[n];
        terminal[goal] = true;
        return new TabularMdp(p, reward, start, terminal, maxSteps);
    }

    /** @return the number of states */
    public int states() {
        return states;
    }

    /** @return the number of actions */
    public int actions() {
        return actions;
    }

    /**
     * @param s a state
     * @param a an action
     * @return the next-state distribution
     */
    public double[] transition(int s, int a) {
        return transitions[s][a].clone();
    }

    /** @return the reward for entering each state */
    public double[] reward() {
        return reward.clone();
    }

    /** @return the start distribution */
    public double[] start() {
        return start.clone();
    }

    /**
     * @param s a state
     * @return whether it ends the episode
     */
    public boolean terminal(int s) {
        return terminal[s];
    }

    /** @return the episode length limit */
    public int maxSteps() {
        return maxSteps;
    }

    /**
     * @param newReward a reward for entering each state
     * @return the same dynamics with another reward
     */
    public TabularMdp withReward(double[] newReward) {
        return new TabularMdp(transitions, newReward.clone(), start, terminal, maxSteps);
    }

    /**
     * Value iteration for the optimal action values.
     *
     * @param gamma the discount
     * @param iterations sweeps
     * @return {@code Q[s][a]}
     */
    public double[][] optimalQ(double gamma, int iterations) {
        return softQ(gamma, iterations, 0);
    }

    /**
     * Soft (maximum-entropy) value iteration: {@code V(s) = temperature * log sum_a exp(Q(s,a) / temperature)};
     * temperature 0 gives ordinary value iteration.
     *
     * @param gamma the discount
     * @param iterations sweeps
     * @param temperature the entropy temperature
     * @return {@code Q[s][a]}
     */
    public double[][] softQ(double gamma, int iterations, double temperature) {
        double[] v = new double[states];
        double[][] q = new double[states][actions];
        for (int it = 0; it < iterations; it++) {
            for (int s = 0; s < states; s++) {
                for (int a = 0; a < actions; a++) {
                    double sum = 0;
                    for (int t = 0; t < states; t++) {
                        double p = transitions[s][a][t];
                        if (p > 0) {
                            sum += p * (reward[t] + (terminal[t] ? 0 : gamma * v[t]));
                        }
                    }
                    q[s][a] = sum;
                }
            }
            for (int s = 0; s < states; s++) {
                double max = Double.NEGATIVE_INFINITY;
                for (double x : q[s]) {
                    max = Math.max(max, x);
                }
                if (temperature <= 0) {
                    v[s] = max;
                } else {
                    double z = 0;
                    for (double x : q[s]) {
                        z += Math.exp((x - max) / temperature);
                    }
                    v[s] = max + temperature * Math.log(z);
                }
            }
        }
        return q;
    }

    /**
     * @param q action values
     * @return the greedy policy as an agent on one-hot observations
     */
    public static Agent greedy(double[][] q) {
        return (obs, explore) -> {
            int s = RlTensors.argmax(obs);
            return new float[] {RlTensors.argmax(q[s])};
        };
    }

    @Override
    public Space observationSpace() {
        return Space.box(states, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(actions);
    }

    private int sample(double[] p) {
        double u = random.nextDouble();
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    private float[] observation() {
        float[] x = new float[states];
        x[state] = 1;
        return x;
    }

    @Override
    public float[] reset(SplittableRandom r) {
        this.random = r;
        state = sample(start);
        steps = 0;
        over = false;
        return observation();
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        state = sample(transitions[state][Math.round(action[0])]);
        steps++;
        boolean done = terminal[state];
        boolean cut = !done && steps >= maxSteps;
        over = done || cut;
        return new Step(observation(), (float) reward[state], done, cut);
    }

    @Override
    public String toString() {
        return "TabularMdp(" + states + " states, " + actions + " actions)";
    }
}
