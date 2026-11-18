package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Unsupervised graph neural solvers for problems with a QUBO form,
 * {@code min_x x^T Q x} over {@code x in {0,1}^n} (linear terms on the diagonal): a GCN
 * ({@link GcnLayer}s) outputs node probabilities {@code p = sigmoid(.)} and minimises the
 * relaxed objective {@code p^T Q p} directly. The output logit also gets a linear skip
 * connection from the node input (our addition: without it the GCN's neighbourhood averaging
 * pulls adjacent nodes together, and at a test budget of a few hundred steps PI-GNN's cuts
 * stayed below a random partition's).
 *
 * <ul>
 *   <li>PI-GNN (Schuetz et al.): {@link #trainInstance} trains on one instance, with a
 *       learnable embedding per node as the input, and rounds {@code p > 1/2}
 *       ({@link #threshold}).</li>
 *   <li>Erdős goes neural (Karalias and Loukas): {@link #trainAcross} trains one network on
 *       many graphs with fixed node features (a constant, the scaled degree and random
 *       features), on the probabilistic penalty loss, which for MIS is exactly the QUBO
 *       {@code -sum p_i + beta sum_(ij in E) p_i p_j}; a small expected loss certifies by the
 *       probabilistic method that a good integral solution exists, and
 *       {@link #conditionalExpectation} finds one by fixing nodes one at a time (most likely
 *       first) to whichever value lowers the expected loss given the others.</li>
 * </ul>
 * {@link #qubo} builds MaxCut ({@code -cut}) and MIS QUBOs; {@link #toIsing} converts a QUBO
 * to Ising couplings for the {@code SimulatedBifurcation} baselines.
 *
 * <p>Papers: Karalias and Loukas, "Erdős Goes Neural: an Unsupervised Learning Framework for
 * Combinatorial Optimization on Graphs" (NeurIPS 2020),
 * <a href="https://arxiv.org/abs/2006.10643">arXiv:2006.10643</a>; Schuetz, Brubaker and
 * Katzgraber, "Combinatorial optimization with physics-inspired graph neural networks" (Nature
 * Machine Intelligence 2022), <a href="https://arxiv.org/abs/2107.01188">arXiv:2107.01188</a>.
 */
public final class QuboGnn {

    /** The graph problems. */
    public enum Problem {
        /** maximum cut */
        MAX_CUT,
        /** maximum independent set */
        MIS
    }

    private static final int RANDOM_FEATURES = 6;

    private final Problem problem;
    private final double penalty;
    private final int hidden;
    private final long seed;
    private GcnLayer first;
    private GcnLayer second;
    private GcnLayer third;
    private Var skip;
    private Var embedding;

    private QuboGnn(Problem problem, double penalty, int hidden, long seed) {
        this.problem = problem;
        this.penalty = penalty;
        this.hidden = hidden;
        this.seed = seed;
    }

    /**
     * Creates a solver.
     *
     * @param problem the problem
     * @param penalty the MIS edge penalty {@code beta} (ignored for MaxCut)
     * @param hidden the GCN width
     * @param seed the seed
     * @return the solver
     */
    public static QuboGnn of(Problem problem, double penalty, int hidden, long seed) {
        return new QuboGnn(problem, penalty, hidden, seed);
    }

    /**
     * The QUBO matrix of a problem on a graph.
     *
     * @param problem the problem
     * @param adjacency 0/1 symmetric
     * @param penalty the MIS edge penalty
     * @return {@code Q} with {@code x^T Q x} equal to {@code -cut} (MaxCut) or
     *     {@code -|S| + penalty * edges inside S} (MIS)
     */
    public static double[][] qubo(Problem problem, double[][] adjacency, double penalty) {
        int n = adjacency.length;
        double[][] q = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && adjacency[i][j] != 0) {
                    if (problem == Problem.MAX_CUT) {
                        q[i][i] -= adjacency[i][j];
                        q[i][j] = adjacency[i][j];
                    } else {
                        q[i][j] = penalty / 2;
                    }
                }
            }
            if (problem == Problem.MIS) {
                q[i][i] = -1;
            }
        }
        return q;
    }

    /**
     * {@code x^T Q x} for a 0/1 vector.
     *
     * @param q the QUBO
     * @param x the assignment
     * @return the energy
     */
    public static double energy(double[][] q, int[] x) {
        double e = 0;
        for (int i = 0; i < x.length; i++) {
            if (x[i] == 0) {
                continue;
            }
            for (int j = 0; j < x.length; j++) {
                e += x[j] == 1 ? q[i][j] : 0;
            }
        }
        return e;
    }

    /**
     * Converts a QUBO to Ising form with {@code x = (1 + s) / 2}: couplings
     * {@code J_ij = -Q_ij / 2} and fields {@code h_i = -sum_j Q_ij / 2}, so the Ising energy
     * {@code -sum_(i<j) J s_i s_j - h.s} equals {@code x^T Q x} up to a constant.
     *
     * @param q the QUBO
     * @return {@code {J (n rows), h (one more row)}}
     */
    public static double[][] toIsing(double[][] q) {
        int n = q.length;
        double[][] out = new double[n + 1][n];
        for (int i = 0; i < n; i++) {
            double row = 0;
            for (int j = 0; j < n; j++) {
                row += q[i][j];
                if (i != j) {
                    out[i][j] = -(q[i][j] + q[j][i]) / 4;
                }
            }
            out[n][i] = -row / 2;
        }
        return out;
    }

    private void build(int in) {
        SplittableRandom r = new SplittableRandom(seed);
        first = new GcnLayer(in, hidden, Activation.RELU, r);
        second = new GcnLayer(hidden, hidden, Activation.RELU, r);
        third = new GcnLayer(hidden, 1, Activation.IDENTITY, r);
        skip = AttentionModel.matrix(r, in, 1);
    }

    private Var forward(Var x, Var adjacency) {
        Var h = first.forward(List.of(x, adjacency), true);
        h = second.forward(List.of(h, adjacency), true);
        return third.forward(List.of(h, adjacency), true).add(x.matmul(skip)).sigmoid();
    }

    /**
     * The fixed node features of the across-instance model: a constant, the degree over the
     * maximum degree, and random features from a seeded generator.
     *
     * @param adjacency the graph
     * @param seed the seed of the random features
     * @return {@code [n, 2 + 6]}
     */
    public static Tensor features(double[][] adjacency, long seed) {
        int n = adjacency.length;
        int f = 2 + RANDOM_FEATURES;
        float[] x = new float[n * f];
        double maxDeg = 1;
        double[] deg = new double[n];
        for (int i = 0; i < n; i++) {
            for (double v : adjacency[i]) {
                deg[i] += v;
            }
            maxDeg = Math.max(maxDeg, deg[i]);
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < n; i++) {
            x[i * f] = 1;
            x[i * f + 1] = (float) (deg[i] / maxDeg);
            for (int k = 0; k < RANDOM_FEATURES; k++) {
                x[i * f + 2 + k] = (float) (2 * r.nextDouble() - 1);
            }
        }
        return Tensor.of(Shape.of(n, f), x);
    }

    private static Var adjacencyVar(double[][] a) {
        int n = a.length;
        float[] v = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                v[i * n + j] = (float) a[i][j];
            }
        }
        return Var.of(Tensor.of(Shape.of(n, n), v));
    }

    private static Var relaxed(Var p, double[][] q) {
        int n = q.length;
        float[] v = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                v[i * n + j] = (float) q[i][j];
            }
        }
        return p.mul(Var.of(Tensor.of(Shape.of(n, n), v)).matmul(p)).sum();
    }

    /**
     * PI-GNN: trains a fresh network with learnable node embeddings on one instance.
     *
     * @param adjacency the graph
     * @param embeddingSize the input embedding width
     * @param steps Adam steps
     * @param learningRate the rate
     * @return the node probabilities after training
     */
    public double[] trainInstance(double[][] adjacency, int embeddingSize, int steps, float learningRate) {
        int n = adjacency.length;
        build(embeddingSize);
        embedding = Var.param(Tensor.normal(Shape.of(n, embeddingSize), 0, 1, new SplittableRandom(seed + 1)));
        double[][] q = qubo(problem, adjacency, penalty);
        Var a = adjacencyVar(adjacency);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = parameters();
        params.add(embedding);
        for (int s = 0; s < steps; s++) {
            Var loss = relaxed(forward(embedding, a), q);
            loss.backward();
            opt.step(params);
        }
        return toDoubles(Var.noGrad(() -> forward(embedding, a)));
    }

    /**
     * Erdős goes neural: trains one network on many graphs with fixed features.
     *
     * @param graphs training graphs
     * @param epochs passes over the graphs
     * @param learningRate the rate
     */
    public void trainAcross(List<double[][]> graphs, int epochs, float learningRate) {
        build(2 + RANDOM_FEATURES);
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = parameters();
        for (int e = 0; e < epochs; e++) {
            for (int g = 0; g < graphs.size(); g++) {
                double[][] adj = graphs.get(g);
                Var loss = relaxed(forward(Var.of(features(adj, g)), adjacencyVar(adj)), qubo(problem, adj, penalty))
                        .mul(1f / adj.length);
                loss.backward();
                opt.step(params);
            }
        }
    }

    /**
     * The across-instance model's probabilities on a new graph.
     *
     * @param adjacency the graph
     * @param featureSeed the random-feature seed
     * @return node probabilities
     */
    public double[] predict(double[][] adjacency, long featureSeed) {
        return toDoubles(Var.noGrad(() -> forward(Var.of(features(adjacency, featureSeed)), adjacencyVar(adjacency))));
    }

    private static double[] toDoubles(Var p) {
        float[] f = p.value().toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /**
     * Rounds {@code p > 1/2}; for MIS, conflicts are then removed greedily (the endpoint with
     * more conflicts first), so the result is independent.
     *
     * @param problem the problem
     * @param adjacency the graph
     * @param p probabilities
     * @return the 0/1 assignment
     */
    public static int[] threshold(Problem problem, double[][] adjacency, double[] p) {
        int n = p.length;
        int[] x = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = p[i] > 0.5 ? 1 : 0;
        }
        if (problem == Problem.MIS) {
            while (true) {
                int worst = -1;
                int worstCount = 0;
                for (int i = 0; i < n; i++) {
                    if (x[i] == 0) {
                        continue;
                    }
                    int c = 0;
                    for (int j = 0; j < n; j++) {
                        c += x[j] == 1 && adjacency[i][j] != 0 ? 1 : 0;
                    }
                    if (c > worstCount) {
                        worstCount = c;
                        worst = i;
                    }
                }
                if (worst < 0) {
                    break;
                }
                x[worst] = 0;
            }
        }
        return x;
    }

    /**
     * The method of conditional expectation: visit nodes by decreasing probability and fix
     * each to the value with the lower expected QUBO energy, the others at their current
     * probabilities (fixed ones at 0/1).
     *
     * @param q the QUBO
     * @param p probabilities
     * @return the 0/1 assignment
     */
    public static int[] conditionalExpectation(double[][] q, double[] p) {
        int n = p.length;
        double[] v = p.clone();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));
        for (int i : order) {
            // E(x_i = 1) - E(x_i = 0) = Q_ii + sum_{j != i} (Q_ij + Q_ji) v_j
            double delta = q[i][i];
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    delta += (q[i][j] + q[j][i]) * v[j];
                }
            }
            v[i] = delta < 0 ? 1 : 0;
        }
        int[] x = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = (int) v[i];
        }
        return x;
    }

    /** @return the GCN's trainable values (after training has built it) */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (first != null) {
            p.addAll(first.parameters());
            p.addAll(second.parameters());
            p.addAll(third.parameters());
            p.add(skip);
        }
        return p;
    }
}
