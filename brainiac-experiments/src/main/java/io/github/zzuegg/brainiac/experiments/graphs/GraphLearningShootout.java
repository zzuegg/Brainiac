package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.graph.CorrectAndSmooth;
import io.github.zzuegg.brainiac.experimental.graph.DecoupledPropagation;
import io.github.zzuegg.brainiac.experimental.graph.DropEdge;
import io.github.zzuegg.brainiac.experimental.graph.Gcn2Layer;
import io.github.zzuegg.brainiac.experimental.graph.GeneralizedPageRank;
import io.github.zzuegg.brainiac.experimental.graph.GraphDiffusionOde;
import io.github.zzuegg.brainiac.experimental.graph.H2GcnLayer;
import io.github.zzuegg.brainiac.experimental.graph.PairNorm;
import io.github.zzuegg.brainiac.experimental.graph.RandomWalkEmbedding;
import io.github.zzuegg.brainiac.experimental.graph.SheafDiffusionLayer;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Node-level graph learning beyond GCN/GAT/GraphSAGE, each against the
 * {@link GnnBenchmarks} baselines on stochastic block models (wave 7, WP1):
 * <ul>
 *   <li>{@link #decoupledPropagation} (7.1.2): SGC and APPNP at a 5% label rate;</li>
 *   <li>{@link #depthSweep} (7.1.3): over-smoothing of deep GCN stacks against GCNII,
 *       PairNorm and DropEdge, with the mean pairwise embedding distance;</li>
 *   <li>{@link #heterophilySweep} (7.1.4, 7.1.5): H2GCN, GPR-GNN and sheaf diffusion as
 *       edge homophily falls from 0.9 to 0.1;</li>
 *   <li>{@link #graphDiffusion} (7.1.6): GRAND against integration time and solver;</li>
 *   <li>{@link #randomWalkEmbeddings} (7.1.12): DeepWalk and node2vec for link prediction
 *       and node classification;</li>
 *   <li>{@link #correctAndSmooth} (7.1.18): label propagation and Correct and Smooth on an
 *       MLP.</li>
 * </ul>
 */
public final class GraphLearningShootout {

    private GraphLearningShootout() {
    }

    private static SyntheticGraphs.Graph sbm(double noise, long seed) {
        return SyntheticGraphs.communities(3, 40, 0.25, 0.02, noise, seed);
    }

    /**
     * SGC and APPNP against GCN, GAT and an MLP at a low label rate.
     *
     * @param labelRate the fraction of labelled nodes
     * @param seeds graphs and initialisations averaged
     * @param steps training steps
     * @param seed the base seed
     * @return model to mean test accuracy, plus "model ms" training times
     */
    public static Map<String, Double> decoupledPropagation(double labelRate, int seeds, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        String[] names = {"MLP", "GCN", "GAT", "SGC", "APPNP"};
        for (String name : names) {
            double[] acc = new double[seeds];
            long start = System.nanoTime();
            for (int s = 0; s < seeds; s++) {
                SyntheticGraphs.Graph g = sbm(2.0, seed + s);
                boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, labelRate, seed + s);
                SplittableRandom r = new SplittableRandom(seed + s);
                GraphTraining.NodeModel m = switch (name) {
                    case "MLP" -> GraphTraining.mlp(3, 16, 3, r);
                    case "GCN" -> GraphTraining.gcn(3, 16, 3, r);
                    case "GAT" -> GraphTraining.gat(3, 3, r);
                    case "SGC" -> GraphTraining.single(DecoupledPropagation.sgc(3, 3, 2, r));
                    default -> GraphTraining.single(DecoupledPropagation.appnp(3, 16, 3, 10, 0.1f, 0.5f, r));
                };
                acc[s] = GraphTraining.trainNodes(m, g, g.adjacencyTensor(), train, 3, steps, 0.02f);
            }
            out.put(name, GraphTraining.round(GraphTraining.mean(acc)));
            out.put(name + " ms", (double) ((System.nanoTime() - start) / 1_000_000 / seeds));
        }
        return out;
    }

    /** A deep GCN stack with optional residuals, PairNorm or DropEdge, or a GCNII stack. */
    private static final class DeepGcn implements GraphTraining.NodeModel {
        private final String kind;
        private final int depth;
        private final List<Var> params = new ArrayList<>();
        private final List<Gcn2Layer> gcn2 = new ArrayList<>();
        private final PairNorm norm = new PairNorm(1f);
        private final DropEdge drop;
        private Tensor last;

        DeepGcn(String kind, int depth, int in, int width, int out, SplittableRandom r) {
            this.kind = kind;
            this.depth = depth;
            this.drop = new DropEdge(0.3, r.split());
            params.add(weight(in, width, r));
            params.add(Var.param(Tensor.zeros(Shape.of(1, width))));
            for (int l = 0; l < depth; l++) {
                if (kind.equals("GCNII")) {
                    Gcn2Layer layer = new Gcn2Layer(width, 0.1f, 0.5f, l + 1, r);
                    gcn2.add(layer);
                    params.addAll(layer.parameters());
                } else {
                    params.add(weight(width, width, r));
                    params.add(Var.param(Tensor.zeros(Shape.of(1, width))));
                }
            }
            params.add(weight(width, out, r));
            params.add(Var.param(Tensor.zeros(Shape.of(1, out))));
        }

        private static Var weight(int in, int out, SplittableRandom r) {
            float b = (float) Math.sqrt(6.0 / (in + out));
            return Var.param(Tensor.uniform(Shape.of(in, out), -b, b, r));
        }

        @Override
        public Var forward(Var x, Tensor adjacency, boolean training) {
            Tensor adj = kind.equals("GCN + DropEdge") ? drop.apply(adjacency, training) : adjacency;
            Var s = Var.of(GraphOps.gcnNormalize(adj));
            Var h0 = x.matmul(params.get(0)).add(params.get(1)).relu();
            Var h = h0;
            for (int l = 0; l < depth; l++) {
                if (kind.equals("GCNII")) {
                    h = gcn2.get(l).forward(List.of(h, Var.of(adj), h0), training);
                } else {
                    Var next = s.matmul(h).matmul(params.get(2 + 2 * l)).add(params.get(3 + 2 * l)).relu();
                    if (kind.equals("GCN + PairNorm")) {
                        next = norm.forward(next);
                    }
                    h = kind.equals("GCN + residual") ? h.add(next) : next;
                }
            }
            last = h.value();
            int k = params.size();
            return h.matmul(params.get(k - 2)).add(params.get(k - 1));
        }

        @Override
        public List<Var> parameters() {
            return params;
        }
    }

    /**
     * The mean pairwise Euclidean distance between row-normalised embeddings (0 = every
     * node collapsed onto one direction).
     */
    static double meanPairwiseDistance(Tensor h) {
        int n = h.shape().dim(0);
        int f = h.shape().dim(1);
        float[] v = h.toArray();
        double[][] u = new double[n][f];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int k = 0; k < f; k++) {
                norm += v[i * f + k] * v[i * f + k];
            }
            norm = Math.sqrt(Math.max(norm, 1e-12));
            for (int k = 0; k < f; k++) {
                u[i][k] = v[i * f + k] / norm;
            }
        }
        double total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = 0;
                for (int k = 0; k < f; k++) {
                    d += (u[i][k] - u[j][k]) * (u[i][k] - u[j][k]);
                }
                total += Math.sqrt(d);
            }
        }
        return total / (n * (n - 1) / 2.0);
    }

    /**
     * Accuracy and smoothing against depth.
     *
     * @param models model names: "GCN", "GCN + residual", "GCN + PairNorm", "GCN + DropEdge",
     *     "GCNII"
     * @param depths hidden graph-convolution layers
     * @param steps training steps
     * @param seed the seed
     * @return "model @ depth" to test accuracy, and "model @ depth dist" to the mean
     *     pairwise distance of the last hidden layer
     */
    public static Map<String, Double> depthSweep(List<String> models, int[] depths, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 30, 0.25, 0.02, 1.0, seed);
        boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.1, seed);
        for (int depth : depths) {
            for (String name : models) {
                DeepGcn m = new DeepGcn(name, depth, 3, 16, 3, new SplittableRandom(seed));
                double acc = GraphTraining.trainNodes(m, g, g.adjacencyTensor(), train, 3, steps, 0.01f);
                out.put(name + " @ " + depth, GraphTraining.round(acc));
                out.put(name + " @ " + depth + " dist", GraphTraining.round(meanPairwiseDistance(m.last)));
            }
        }
        return out;
    }

    /** Input projection, four sheaf diffusion layers, output projection. */
    private static GraphTraining.NodeModel sheaf(SheafDiffusionLayer.Maps maps, int in, int out, SplittableRandom r) {
        int d = 2;
        int c = 8;
        Var w1 = Var.param(Tensor.normal(Shape.of(in, d * c), 0, (float) Math.sqrt(2.0 / in), r));
        Var w2 = Var.param(Tensor.normal(Shape.of(d * c, out), 0, (float) Math.sqrt(1.0 / (d * c)), r));
        Var b2 = Var.param(Tensor.zeros(Shape.of(1, out)));
        List<SheafDiffusionLayer> layers = new ArrayList<>();
        for (int l = 0; l < 4; l++) {
            layers.add(new SheafDiffusionLayer(d, c, maps, r));
        }
        return new GraphTraining.NodeModel() {
            @Override
            public Var forward(Var x, Tensor adjacency, boolean training) {
                Var adj = Var.of(adjacency);
                Var h = x.matmul(w1).relu();
                for (SheafDiffusionLayer l : layers) {
                    h = l.forward(List.of(h, adj), training);
                }
                return h.matmul(w2).add(b2);
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(List.of(w1, w2, b2));
                for (SheafDiffusionLayer l : layers) {
                    p.addAll(l.parameters());
                }
                return p;
            }
        };
    }

    /**
     * Accuracy against edge homophily.
     *
     * @param models model names: "MLP", "GCN", "SAGE", "H2GCN", "GPR-GNN", "Sheaf diag",
     *     "Sheaf orth", "Sheaf general"
     * @param homophily edge homophily levels
     * @param seeds graphs averaged per level
     * @param steps training steps
     * @param seed the base seed
     * @return "model @ h" to mean test accuracy, and "measured h @ h"
     */
    public static Map<String, Double> heterophilySweep(List<String> models, double[] homophily, int seeds,
            int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (double h : homophily) {
            double measured = 0;
            for (int s = 0; s < seeds; s++) {
                measured += SyntheticGraphs.edgeHomophily(SyntheticGraphs.homophilous(3, 40, 6, h, 1.5, seed + s));
            }
            out.put("measured h @ " + h, GraphTraining.round(measured / seeds));
            for (String name : models) {
                double acc = GraphTraining.overSeeds(seeds, seed, sd -> {
                    long graphSeed = seed + (sd - seed) / 1000;
                    SyntheticGraphs.Graph g = SyntheticGraphs.homophilous(3, 40, 6, h, 1.5, graphSeed);
                    boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.3, graphSeed);
                    SplittableRandom r = new SplittableRandom(graphSeed);
                    GraphTraining.NodeModel m = switch (name) {
                        case "MLP" -> GraphTraining.mlp(3, 16, 3, r);
                        case "GCN" -> GraphTraining.gcn(3, 16, 3, r);
                        case "SAGE" -> GraphTraining.sage(3, 16, 3, r);
                        case "H2GCN" -> GraphTraining.single(new H2GcnLayer(3, 16, 3, 2, r));
                        case "GPR-GNN" -> GraphTraining.single(new GeneralizedPageRank(3, 16, 3, 10, 0.1f, r));
                        case "Sheaf diag" -> sheaf(SheafDiffusionLayer.Maps.DIAGONAL, 3, 3, r);
                        case "Sheaf orth" -> sheaf(SheafDiffusionLayer.Maps.ORTHOGONAL, 3, 3, r);
                        case "Sheaf general" -> sheaf(SheafDiffusionLayer.Maps.GENERAL, 3, 3, r);
                        default -> throw new IllegalArgumentException(name);
                    };
                    return GraphTraining.trainNodes(m, g, g.adjacencyTensor(), train, 3, steps, 0.02f);
                });
                out.put(name + " @ " + h, GraphTraining.round(acc));
            }
        }
        return out;
    }

    /**
     * GRAND against GCN, GAT and GCNII on SBM node classification.
     *
     * @param times integration times {@code T} to sweep (RK4 with 2 steps per unit time)
     * @param steps training steps
     * @param seed the seed
     * @return model to test accuracy, and "GRAND T evals" to function evaluations per forward
     */
    public static Map<String, Double> graphDiffusion(float[] times, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SyntheticGraphs.Graph g = sbm(2.0, seed);
        boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.1, seed);
        Tensor adj = g.adjacencyTensor();
        SplittableRandom r0 = new SplittableRandom(seed);
        out.put("GCN", GraphTraining.round(GraphTraining.trainNodes(GraphTraining.gcn(3, 16, 3, r0), g, adj, train, 3,
                steps, 0.02f)));
        out.put("GAT", GraphTraining.round(GraphTraining.trainNodes(GraphTraining.gat(3, 3, new SplittableRandom(seed)),
                g, adj, train, 3, steps, 0.02f)));
        out.put("GCNII (8)", GraphTraining.round(GraphTraining.trainNodes(
                new DeepGcn("GCNII", 8, 3, 16, 3, new SplittableRandom(seed)), g, adj, train, 3, steps, 0.01f)));
        for (float t : times) {
            SplittableRandom r = new SplittableRandom(seed);
            Var w1 = Var.param(Tensor.normal(Shape.of(3, 16), 0, 0.8f, r));
            Var w2 = Var.param(Tensor.normal(Shape.of(16, 3), 0, 0.25f, r));
            Var b2 = Var.param(Tensor.zeros(Shape.of(1, 3)));
            int solverSteps = Math.max(1, Math.round(2 * t));
            GraphDiffusionOde ode = new GraphDiffusionOde(16, 8, OdeSolver.rk4(solverSteps), t, false, r);
            GraphTraining.NodeModel m = new GraphTraining.NodeModel() {
                @Override
                public Var forward(Var x, Tensor adjacency, boolean training) {
                    return ode.forward(List.of(x.matmul(w1).tanh(), Var.of(adjacency)), training).matmul(w2).add(b2);
                }

                @Override
                public List<Var> parameters() {
                    List<Var> p = new ArrayList<>(List.of(w1, w2, b2));
                    p.addAll(ode.parameters());
                    return p;
                }
            };
            out.put("GRAND T=" + t, GraphTraining.round(GraphTraining.trainNodes(m, g, adj, train, 3, steps, 0.02f)));
            out.put("GRAND T=" + t + " evals", (double) ode.solver().evaluations());
        }
        return out;
    }

    private static double auc(double[] positives, double[] negatives) {
        return GnnBenchmarks.auc(positives, negatives);
    }

    /**
     * DeepWalk and node2vec link prediction and node classification, against the graph
     * autoencoder and common neighbours of {@link GnnBenchmarks#linkPrediction} (same graph,
     * same held-out edges and negatives) and a spectral embedding.
     *
     * @param pq the (p, q) pairs to try; (1, 1) is DeepWalk
     * @param seed the seed
     * @return "model link AUC" and "model node acc"
     */
    public static Map<String, Double> randomWalkEmbeddings(double[][] pq, long seed) {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 30, 0.3, 0.02, 1.0, seed);
        int n = g.nodes();
        SplittableRandom r = new SplittableRandom(seed);
        float[] trainAdj = g.adjacency().clone();
        List<int[]> heldOut = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (g.adjacency()[i * n + j] > 0 && r.nextDouble() < 0.15) {
                    heldOut.add(new int[] {i, j});
                    trainAdj[i * n + j] = 0;
                    trainAdj[j * n + i] = 0;
                }
            }
        }
        List<int[]> nonEdges = new ArrayList<>();
        while (nonEdges.size() < heldOut.size()) {
            int i = r.nextInt(n);
            int j = r.nextInt(n);
            if (i != j && g.adjacency()[i * n + j] == 0) {
                nonEdges.add(new int[] {i, j});
            }
        }
        Tensor adj = Tensor.of(Shape.of(n, n), trainAdj);
        boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.1, seed);
        Map<String, Double> out = new LinkedHashMap<>();
        Map<String, Double> gae = GnnBenchmarks.linkPrediction(150, seed);
        out.put("GAE link AUC", GraphTraining.round(gae.get("GAE")));
        out.put("common neighbours link AUC", GraphTraining.round(gae.get("common neighbours")));
        double[][] aff = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aff[i][j] = trainAdj[i * n + j];
            }
        }
        double[][] spec = SpectralClustering.embed(aff, 8);
        float[][] specF = new float[n][8];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 8; k++) {
                specF[i][k] = (float) spec[i][k];
            }
        }
        java.util.function.ToDoubleBiFunction<Integer, Integer> specScore = (i, j) -> {
            double s = 0;
            for (int k = 0; k < 8; k++) {
                s += spec[i][k] * spec[j][k];
            }
            return s;
        };
        out.put("spectral link AUC", GraphTraining.round(auc(
                heldOut.stream().mapToDouble(e -> specScore.applyAsDouble(e[0], e[1])).toArray(),
                nonEdges.stream().mapToDouble(e -> specScore.applyAsDouble(e[0], e[1])).toArray())));
        out.put("spectral node acc", GraphTraining.round(GraphTraining.probe(specF, g.nodeLabels(), train, 3, seed)));
        for (double[] p : pq) {
            RandomWalkEmbedding e = RandomWalkEmbedding.train(adj, 16, 10, 20, p[0], p[1], 4, 2, seed);
            String name = p[0] == 1 && p[1] == 1 ? "DeepWalk" : "node2vec p=" + p[0] + " q=" + p[1];
            out.put(name + " link AUC", GraphTraining.round(auc(
                    heldOut.stream().mapToDouble(x -> e.score(x[0], x[1])).toArray(),
                    nonEdges.stream().mapToDouble(x -> e.score(x[0], x[1])).toArray())));
            float[][] emb = new float[n][];
            for (int i = 0; i < n; i++) {
                emb[i] = e.vector(i);
            }
            out.put(name + " node acc", GraphTraining.round(GraphTraining.probe(emb, g.nodeLabels(), train, 3, seed)));
        }
        return out;
    }

    /**
     * Label propagation and Correct and Smooth against message passing, over feature noise.
     *
     * @param noises feature noise levels
     * @param alpha the propagation weight of both C&amp;S stages
     * @param steps training steps of the trained models
     * @param seed the seed
     * @return "model @ noise" to test accuracy, and "model ms" to the time at the last noise
     */
    public static Map<String, Double> correctAndSmooth(double[] noises, float alpha, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (double noise : noises) {
            SyntheticGraphs.Graph g = sbm(noise, seed);
            boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.1, seed);
            Tensor adj = g.adjacencyTensor();
            for (String name : new String[] {"MLP", "GCN", "GAT", "APPNP"}) {
                SplittableRandom r = new SplittableRandom(seed);
                GraphTraining.NodeModel m = switch (name) {
                    case "MLP" -> GraphTraining.mlp(3, 16, 3, r);
                    case "GCN" -> GraphTraining.gcn(3, 16, 3, r);
                    case "GAT" -> GraphTraining.gat(3, 3, r);
                    default -> GraphTraining.single(DecoupledPropagation.appnp(3, 16, 3, 10, 0.1f, 0.5f, r));
                };
                long start = System.nanoTime();
                double acc = GraphTraining.trainNodes(m, g, adj, train, 3, steps, 0.02f);
                out.put(name + " @ " + noise, GraphTraining.round(acc));
                out.put(name + " ms", (double) ((System.nanoTime() - start) / 1_000_000));
                if (name.equals("MLP")) {
                    long cs = System.nanoTime();
                    Tensor logits = Var.noGrad(() -> m.forward(Var.of(g.featureTensor()), adj, false)).value();
                    Tensor probs = logits.sub(logits.max(1, true)).exp();
                    probs = probs.div(probs.sum(1, true));
                    Tensor smoothed = CorrectAndSmooth.correctAndSmooth(adj, probs, g.nodeLabels(), train, alpha, alpha,
                            50);
                    out.put("MLP + C&S @ " + noise, GraphTraining.round(
                            GraphTraining.accuracy(smoothed, g.nodeLabels(), train, false)));
                    out.put("MLP + C&S ms", (double) ((System.nanoTime() - start) / 1_000_000));
                    out.put("C&S post-processing ms", (double) ((System.nanoTime() - cs) / 1_000_000));
                }
            }
            Tensor lp = CorrectAndSmooth.labelPropagation(adj, g.nodeLabels(), train, 3, 0.9f, 30);
            out.put("LP @ " + noise, GraphTraining.round(GraphTraining.accuracy(lp, g.nodeLabels(), train, false)));
        }
        return out;
    }
}
