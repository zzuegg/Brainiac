package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.GFlowNet;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * GFlowNets for combinatorial optimisation (Zhang et al.): a sampler that builds solutions
 * one decision at a time with probability proportional to {@code exp(-E(x) / T)}, giving many
 * diverse good solutions instead of one.
 *
 * <p>Two set-building problems implement the wave 6 {@link GFlowNet.Problem}, so both the
 * plain trajectory-balance {@link GFlowNet} and this class train on them:
 * <ul>
 *   <li>{@link #independentSet}: a state marks every node undecided, in the set or excluded;
 *       an action adds an undecided node (its neighbours become excluded); stop is legal once
 *       nothing is undecided. Energy {@code -|S|}. A state with {@code k} set nodes has
 *       {@code k} parents (any of them could have been added last).</li>
 *   <li>{@link #maxCut}: an action puts an undecided node on side 0 or 1; energy is minus the
 *       cut among decided nodes.</li>
 * </ul>
 *
 * <p>The forward-looking detailed-balance loss (FL-DB) uses the energy of partial states:
 * with {@code F(s) = F~(s) exp(-E(s)/T)} each transition must satisfy
 * {@code log F~(s) + log P_F(s'|s) = log F~(s') + log P_B(s|s') + (E(s) - E(s')) / T}, and
 * terminal states have {@code F~ = 1}. Every transition is a training signal with a dense,
 * local credit for the energy change, unlike trajectory balance, which sees one number per
 * trajectory. One MLP outputs the action logits and {@code log F~}. The temperature can be
 * annealed ({@link #temperature}) to sharpen the distribution.
 *
 * <p>Paper: Zhang, Dai, Malkin, Courville, Bengio and Pan, "Let the Flows Tell: Solving Graph
 * Combinatorial Optimization Problems with GFlowNets" (NeurIPS 2023),
 * <a href="https://arxiv.org/abs/2305.17010">arXiv:2305.17010</a>.
 */
public final class GFlowNetCo {

    /** A set-building problem with an energy defined on partial states. */
    public interface EnergyProblem extends GFlowNet.Problem {
        /**
         * @param state any state
         * @return its energy (lower is better)
         */
        double energy(int[] state);

        /** @return the temperature of {@link #logReward} */
        double temperature();

        @Override
        default double logReward(int[] terminal) {
            return -energy(terminal) / temperature();
        }
    }

    /**
     * Maximum independent set on a graph.
     *
     * @param adjacency 0/1 symmetric
     * @param temperature the reward temperature
     * @return the problem (state entries 0 undecided, 1 in, 2 excluded)
     */
    public static EnergyProblem independentSet(double[][] adjacency, double temperature) {
        int n = adjacency.length;
        return new EnergyProblem() {
            @Override
            public int[] initial() {
                return new int[n];
            }

            @Override
            public int actions() {
                return n + 1;
            }

            @Override
            public int stopAction() {
                return n;
            }

            @Override
            public boolean[] legal(int[] state) {
                boolean[] l = new boolean[n + 1];
                boolean any = false;
                for (int i = 0; i < n; i++) {
                    l[i] = state[i] == 0;
                    any |= l[i];
                }
                l[n] = !any;
                return l;
            }

            @Override
            public int[] step(int[] state, int action) {
                int[] s = state.clone();
                s[action] = 1;
                for (int j = 0; j < n; j++) {
                    if (adjacency[action][j] != 0 && s[j] == 0) {
                        s[j] = 2;
                    }
                }
                return s;
            }

            @Override
            public int parents(int[] state) {
                int k = 0;
                for (int v : state) {
                    k += v == 1 ? 1 : 0;
                }
                return Math.max(1, k);
            }

            @Override
            public float[] features(int[] state) {
                float[] f = new float[3 * n];
                for (int i = 0; i < n; i++) {
                    f[3 * i + state[i]] = 1;
                }
                return f;
            }

            @Override
            public double energy(int[] state) {
                int k = 0;
                for (int v : state) {
                    k += v == 1 ? 1 : 0;
                }
                return -k;
            }

            @Override
            public double temperature() {
                return temperature;
            }
        };
    }

    /**
     * MaxCut on a graph.
     *
     * @param w weights
     * @param temperature the reward temperature
     * @return the problem (state entries 0 undecided, 1 side A, 2 side B)
     */
    public static EnergyProblem maxCut(double[][] w, double temperature) {
        int n = w.length;
        return new EnergyProblem() {
            @Override
            public int[] initial() {
                return new int[n];
            }

            @Override
            public int actions() {
                return 2 * n + 1;
            }

            @Override
            public int stopAction() {
                return 2 * n;
            }

            @Override
            public boolean[] legal(int[] state) {
                boolean[] l = new boolean[2 * n + 1];
                boolean any = false;
                for (int i = 0; i < n; i++) {
                    l[2 * i] = l[2 * i + 1] = state[i] == 0;
                    any |= state[i] == 0;
                }
                l[2 * n] = !any;
                return l;
            }

            @Override
            public int[] step(int[] state, int action) {
                int[] s = state.clone();
                s[action / 2] = 1 + action % 2;
                return s;
            }

            @Override
            public int parents(int[] state) {
                int k = 0;
                for (int v : state) {
                    k += v != 0 ? 1 : 0;
                }
                return Math.max(1, k);
            }

            @Override
            public float[] features(int[] state) {
                float[] f = new float[3 * n];
                for (int i = 0; i < n; i++) {
                    f[3 * i + state[i]] = 1;
                }
                return f;
            }

            @Override
            public double energy(int[] state) {
                double c = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        if (state[i] != 0 && state[j] != 0 && state[i] != state[j]) {
                            c += w[i][j];
                        }
                    }
                }
                return -c;
            }

            @Override
            public double temperature() {
                return temperature;
            }
        };
    }

    private final EnergyProblem problem;
    private final Model net;
    private final Optimizer optimizer;
    private double temperature;

    private GFlowNetCo(EnergyProblem problem, int hidden, float learningRate, long seed) {
        this.problem = problem;
        this.temperature = problem.temperature();
        int in = problem.features(problem.initial()).length;
        this.net = Model.sequential().input(in).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                .dense(problem.actions() + 1).seed(seed).build();
        this.optimizer = Optimizer.adam(learningRate);
    }

    /**
     * Creates the sampler.
     *
     * @param problem the problem
     * @param hidden MLP width
     * @param learningRate Adam rate
     * @param seed the seed
     * @return the sampler
     */
    public static GFlowNetCo of(EnergyProblem problem, int hidden, float learningRate, long seed) {
        return new GFlowNetCo(problem, hidden, learningRate, seed);
    }

    /**
     * Sets the training temperature (anneal it downwards to sharpen the sampler).
     *
     * @param t the temperature
     */
    public void temperature(double t) {
        this.temperature = t;
    }

    private static Tensor rows(List<float[]> rows) {
        int f = rows.get(0).length;
        float[] v = new float[rows.size() * f];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, v, i * f, f);
        }
        return Tensor.of(Shape.of(rows.size(), f), v);
    }

    /**
     * Samples trajectories, all in step.
     *
     * @param n how many
     * @param epsilon uniform-action mixing
     * @param random the source
     * @return the trajectories
     */
    public List<GFlowNet.Trajectory> sample(int n, double epsilon, SplittableRandom random) {
        List<List<int[]>> states = new ArrayList<>();
        List<List<Integer>> actions = new ArrayList<>();
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            states.add(new ArrayList<>(List.of(problem.initial())));
            actions.add(new ArrayList<>());
        }
        int live = n;
        int a = problem.actions();
        while (live > 0) {
            List<Integer> who = new ArrayList<>();
            List<float[]> feats = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    who.add(i);
                    feats.add(problem.features(last(states.get(i))));
                }
            }
            float[] out = Var.noGrad(() -> net.forward(Var.of(rows(feats)), false)).value().toArray();
            for (int k = 0; k < who.size(); k++) {
                int i = who.get(k);
                int[] s = last(states.get(i));
                boolean[] legal = problem.legal(s);
                int act = pick(out, k * (a + 1), legal, epsilon, random);
                actions.get(i).add(act);
                if (act == problem.stopAction()) {
                    done[i] = true;
                    live--;
                } else {
                    states.get(i).add(problem.step(s, act));
                }
            }
        }
        List<GFlowNet.Trajectory> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            res.add(new GFlowNet.Trajectory(states.get(i), actions.get(i).stream().mapToInt(Integer::intValue).toArray()));
        }
        return res;
    }

    private static int pick(float[] out, int offset, boolean[] legal, double epsilon, SplittableRandom r) {
        int a = legal.length;
        List<Integer> options = new ArrayList<>();
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < a; j++) {
            if (legal[j]) {
                options.add(j);
                max = Math.max(max, out[offset + j]);
            }
        }
        if (r.nextDouble() < epsilon) {
            return options.get(r.nextInt(options.size()));
        }
        double sum = 0;
        for (int j : options) {
            sum += Math.exp(out[offset + j] - max);
        }
        double u = r.nextDouble() * sum;
        for (int j : options) {
            u -= Math.exp(out[offset + j] - max);
            if (u <= 0) {
                return j;
            }
        }
        return options.get(options.size() - 1);
    }

    private static int[] last(List<int[]> l) {
        return l.get(l.size() - 1);
    }

    /**
     * The forward-looking detailed-balance loss over every transition of a batch.
     *
     * @param batch trajectories
     * @return the mean squared residual
     */
    public Var forwardLookingLoss(List<GFlowNet.Trajectory> batch) {
        List<float[]> feats = new ArrayList<>();
        List<boolean[]> legal = new ArrayList<>();
        List<Integer> taken = new ArrayList<>();
        List<Integer> nextRow = new ArrayList<>();
        List<Float> constant = new ArrayList<>();
        for (GFlowNet.Trajectory t : batch) {
            int base = feats.size();
            int len = t.actions().length;
            for (int k = 0; k < len; k++) {
                feats.add(problem.features(t.states().get(k)));
                legal.add(problem.legal(t.states().get(k)));
            }
            for (int k = 0; k < len; k++) {
                int act = t.actions()[k];
                taken.add(act);
                int[] s = t.states().get(k);
                if (act == problem.stopAction()) {
                    nextRow.add(-1);
                    constant.add(0f);
                } else {
                    int[] s2 = t.states().get(k + 1);
                    nextRow.add(k + 1 < len ? base + k + 1 : -1);
                    // - log P_B(s | s') = log parents(s'), plus the energy change
                    double c = Math.log(problem.parents(s2)) + (problem.energy(s2) - problem.energy(s)) / temperature;
                    constant.add((float) c);
                }
            }
        }
        int m = feats.size();
        int a = problem.actions();
        Var out = net.forward(Var.of(rows(feats)), true);                     // [m, a + 1]
        float[] mask = new float[m * (a + 1)];
        float[] pickA = new float[m * (a + 1)];
        float[] pickF = new float[m * (a + 1)];
        float[] next = new float[m * m];
        float[] hasNext = new float[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < a; j++) {
                mask[i * (a + 1) + j] = legal.get(i)[j] ? 0 : -1e9f;
            }
            mask[i * (a + 1) + a] = -1e9f;
            pickA[i * (a + 1) + taken.get(i)] = 1;
            pickF[i * (a + 1) + a] = 1;
            if (nextRow.get(i) >= 0) {
                next[i * m + nextRow.get(i)] = 1;
                hasNext[i] = 1;
            }
        }
        Var logits = out.add(Tensor.of(Shape.of(m, a + 1), mask));
        Var logPf = logits.logSoftmax(1).mul(Tensor.of(Shape.of(m, a + 1), pickA)).sum(1, true);   // [m, 1]
        Var logF = out.mul(Tensor.of(Shape.of(m, a + 1), pickF)).sum(1, true);                      // [m, 1]
        Var logFNext = Var.of(Tensor.of(Shape.of(m, m), next)).matmul(logF);                        // 0 at terminals
        float[] c = new float[m];
        for (int i = 0; i < m; i++) {
            c[i] = constant.get(i);
        }
        Var delta = logF.add(logPf).sub(logFNext).add(Tensor.of(Shape.of(m, 1), c));
        return delta.square().mean();
    }

    /**
     * One FL-DB step on fresh trajectories.
     *
     * @param batch trajectories
     * @param epsilon exploration
     * @param random the source
     * @return the loss
     */
    public float trainStep(int batch, double epsilon, SplittableRandom random) {
        Var loss = forwardLookingLoss(sample(batch, epsilon, random));
        float v = loss.value().item();
        loss.backward();
        optimizer.step(net.parameters());
        return v;
    }

    /**
     * The energies of sampled terminal objects.
     *
     * @param trajectories trajectories
     * @param problem the problem
     * @return the terminal energies
     */
    public static double[] energies(List<GFlowNet.Trajectory> trajectories, EnergyProblem problem) {
        return trajectories.stream().mapToDouble(t -> problem.energy(t.terminal())).toArray();
    }

    /**
     * The number of distinct terminal objects within a share of the best known value.
     *
     * @param trajectories samples
     * @param problem the problem
     * @param bestEnergy the best known energy (negative)
     * @param share e.g. 0.95
     * @return distinct near-optimal objects
     */
    public static int distinctNearOptimal(List<GFlowNet.Trajectory> trajectories, EnergyProblem problem,
            double bestEnergy, double share) {
        java.util.Set<String> seen = new java.util.HashSet<>();
        for (GFlowNet.Trajectory t : trajectories) {
            if (problem.energy(t.terminal()) <= share * bestEnergy + 1e-9) {
                seen.add(Arrays.toString(t.terminal()));
            }
        }
        return seen.size();
    }
}
