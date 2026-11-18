package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A generative flow network trained with <em>trajectory balance</em> (Malkin et al.,
 * "Trajectory balance: improved credit assignment in GFlowNets",
 * <a href="https://arxiv.org/abs/2201.13259">arXiv:2201.13259</a>; Bengio et al. 2021). Objects
 * are built step by step in a directed acyclic graph of states ({@link Problem}); a forward
 * policy {@code P_F(s' | s)} picks the next step or stops. Instead of maximizing the reward
 * (which collapses onto one mode, like RL does), the sampler is trained so that it ends in
 * {@code x} with probability <em>proportional</em> to {@code R(x)}: for every complete
 * trajectory {@code s_0 -> ... -> s_n = x},
 *
 * <pre>
 * L(tau) = ( log Z + sum_t log P_F(s_(t+1) | s_t) - log R(x) - sum_t log P_B(s_t | s_(t+1)) )^2
 * </pre>
 *
 * <p>with a learned scalar {@code log Z} and a fixed uniform backward policy {@code P_B} over the
 * parents of each state. At the optimum every high-reward mode is sampled in proportion, which is
 * what diverse candidate generation (molecules, sequences, designs) needs. Training samples its
 * own trajectories, optionally mixing in uniform exploration (off-policy is allowed: the loss
 * holds for any trajectory).
 */
public final class GFlowNet {

    /** A constructive generation problem: a DAG of states with a stop action. */
    public interface Problem {
        /** @return the initial (empty) state */
        int[] initial();

        /** @return the number of actions, including stop */
        int actions();

        /** @return the index of the stop action */
        int stopAction();

        /**
         * @param state a state
         * @return which actions are legal (stop must be legal wherever the state is a valid object)
         */
        boolean[] legal(int[] state);

        /**
         * @param state a state
         * @param action a legal non-stop action
         * @return the child state
         */
        int[] step(int[] state, int action);

        /**
         * @param state a non-initial state
         * @return how many parents it has in the DAG (the uniform backward policy is 1 / this)
         */
        int parents(int[] state);

        /**
         * @param state a state
         * @return its features for the policy network
         */
        float[] features(int[] state);

        /**
         * @param terminal a finished object
         * @return {@code log R(x)}
         */
        double logReward(int[] terminal);
    }

    /**
     * One sampled trajectory.
     *
     * @param states the visited states, initial first, terminal last
     * @param actions the actions taken, stop last
     */
    public record Trajectory(List<int[]> states, int[] actions) {
        /** @return the finished object */
        public int[] terminal() {
            return states.get(states.size() - 1);
        }
    }

    private final Problem problem;
    private final Model policy;
    private final Var logZ;
    private final Optimizer optimizer;
    private final Optimizer zOptimizer;
    private double exploration;
    private long rewardCalls;

    private GFlowNet(Problem problem, int hidden, float learningRate, long seed) {
        this.problem = Objects.requireNonNull(problem);
        int in = problem.features(problem.initial()).length;
        this.policy = Model.sequential().input(in).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                .dense(problem.actions()).seed(seed).build();
        this.logZ = Var.param(Tensor.zeros(Shape.of(1)));
        this.optimizer = Optimizer.adam(learningRate);
        this.zOptimizer = Optimizer.adam(learningRate * 10);
    }

    /**
     * Creates a GFlowNet with an MLP forward policy.
     *
     * @param problem the problem
     * @param hidden the width of the two hidden layers
     * @param learningRate the policy's Adam rate ({@code log Z} learns ten times faster, as usual)
     * @param seed the seed
     * @return the sampler
     */
    public static GFlowNet of(Problem problem, int hidden, float learningRate, long seed) {
        return new GFlowNet(problem, hidden, learningRate, seed);
    }

    /**
     * Mixes uniform random legal actions into the training trajectories.
     *
     * @param epsilon the probability of a uniform action
     * @return this sampler
     */
    public GFlowNet exploration(double epsilon) {
        this.exploration = epsilon;
        return this;
    }

    /** @return the learned {@code log Z}, an estimate of {@code log sum_x R(x)} */
    public double logPartition() {
        return logZ.value().item();
    }

    /** @return how many rewards were computed in training */
    public long rewardCalls() {
        return rewardCalls;
    }

    private static double[] maskedSoftmax(float[] logits, boolean[] legal) {
        double max = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < logits.length; a++) {
            if (legal[a]) {
                max = Math.max(max, logits[a]);
            }
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int a = 0; a < logits.length; a++) {
            if (legal[a]) {
                p[a] = Math.exp(logits[a] - max);
                sum += p[a];
            }
        }
        for (int a = 0; a < p.length; a++) {
            p[a] /= sum;
        }
        return p;
    }

    /**
     * Samples trajectories from the forward policy, all in step.
     *
     * @param n how many
     * @param epsilon uniform-action mixing
     * @param random the random source
     * @return the trajectories
     */
    public List<Trajectory> sample(int n, double epsilon, SplittableRandom random) {
        List<List<int[]>> states = new ArrayList<>();
        List<List<Integer>> actions = new ArrayList<>();
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            states.add(new ArrayList<>(List.of(problem.initial())));
            actions.add(new ArrayList<>());
        }
        int live = n;
        while (live > 0) {
            List<Integer> who = new ArrayList<>();
            List<float[]> rows = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    who.add(i);
                    rows.add(problem.features(last(states.get(i))));
                }
            }
            float[] logits = policy.forward(Var.of(rowsTensor(rows)), false).value().toArray();
            int a = problem.actions();
            for (int k = 0; k < who.size(); k++) {
                int i = who.get(k);
                int[] s = last(states.get(i));
                boolean[] legal = problem.legal(s);
                double[] p = maskedSoftmax(Arrays.copyOfRange(logits, k * a, (k + 1) * a), legal);
                int act;
                if (random.nextDouble() < epsilon) {
                    List<Integer> options = new ArrayList<>();
                    for (int j = 0; j < a; j++) {
                        if (legal[j]) {
                            options.add(j);
                        }
                    }
                    act = options.get(random.nextInt(options.size()));
                } else {
                    double u = random.nextDouble();
                    act = -1;
                    for (int j = 0; j < a; j++) {
                        if (legal[j]) {
                            act = j;
                            u -= p[j];
                            if (u <= 0) {
                                break;
                            }
                        }
                    }
                }
                actions.get(i).add(act);
                if (act == problem.stopAction()) {
                    done[i] = true;
                    live--;
                } else {
                    states.get(i).add(problem.step(s, act));
                }
            }
        }
        List<Trajectory> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(new Trajectory(states.get(i), actions.get(i).stream().mapToInt(Integer::intValue).toArray()));
        }
        return out;
    }

    private static int[] last(List<int[]> l) {
        return l.get(l.size() - 1);
    }

    private static Tensor rowsTensor(List<float[]> rows) {
        int f = rows.get(0).length;
        float[] v = new float[rows.size() * f];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, v, i * f, f);
        }
        return Tensor.of(Shape.of(rows.size(), f), v);
    }

    /**
     * The trajectory-balance loss of trajectories, averaged.
     *
     * @param batch the trajectories
     * @return the loss
     */
    public Var trajectoryBalance(List<Trajectory> batch) {
        List<float[]> rows = new ArrayList<>();
        List<boolean[]> legal = new ArrayList<>();
        List<Integer> taken = new ArrayList<>();
        int[] owner;
        List<Integer> owners = new ArrayList<>();
        float[] target = new float[batch.size()];
        for (int b = 0; b < batch.size(); b++) {
            Trajectory t = batch.get(b);
            double logPb = 0;
            for (int k = 0; k < t.actions().length; k++) {
                int[] s = t.states().get(k);
                rows.add(problem.features(s));
                legal.add(problem.legal(s));
                taken.add(t.actions()[k]);
                owners.add(b);
                if (t.actions()[k] != problem.stopAction()) {
                    logPb -= Math.log(problem.parents(t.states().get(k + 1)));
                }
            }
            rewardCalls++;
            target[b] = (float) (problem.logReward(t.terminal()) + logPb);
        }
        owner = owners.stream().mapToInt(Integer::intValue).toArray();
        int a = problem.actions();
        int n = rows.size();
        float[] maskAdd = new float[n * a];
        float[] pick = new float[n * a];
        float[] sumTo = new float[batch.size() * n];
        for (int i = 0; i < n; i++) {
            boolean[] l = legal.get(i);
            for (int j = 0; j < a; j++) {
                maskAdd[i * a + j] = l[j] ? 0 : -1e9f;
            }
            pick[i * a + taken.get(i)] = 1;
            sumTo[owner[i] * n + i] = 1;
        }
        Var logits = policy.forward(Var.of(rowsTensor(rows)), true).add(Tensor.of(Shape.of(n, a), maskAdd));
        Var max = Var.of(logits.value().max(1, true));
        Var shifted = logits.sub(max);
        Var logSoftmax = shifted.sub(shifted.exp().sum(1, true).log());
        Var logPf = logSoftmax.mul(Tensor.of(Shape.of(n, a), pick)).sum(1, true);          // [n, 1]
        Var perTrajectory = Var.of(Tensor.of(Shape.of(batch.size(), n), sumTo)).matmul(logPf);   // [b, 1]
        Var delta = perTrajectory.add(logZ).sub(Tensor.of(Shape.of(batch.size(), 1), target));
        return delta.square().mean();
    }

    /**
     * One training step on freshly sampled trajectories.
     *
     * @param batch trajectories per step
     * @param random the random source
     * @return the loss
     */
    public float trainStep(int batch, SplittableRandom random) {
        Var loss = trajectoryBalance(sample(batch, exploration, random));
        float v = loss.value().item();
        loss.backward();
        optimizer.step(policy.parameters());
        zOptimizer.step(List.of(logZ));
        return v;
    }

    /** @return the policy network */
    public Model policy() {
        return policy;
    }
}
