package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.graph.DiffPool;
import io.github.zzuegg.brainiac.experimental.graph.GraphGpsLayer;
import io.github.zzuegg.brainiac.experimental.graph.GraphPositionalEncodings;
import io.github.zzuegg.brainiac.experimental.graph.GraphRewiring;
import io.github.zzuegg.brainiac.experimental.graph.PnaLayer;
import io.github.zzuegg.brainiac.experimental.graph.RandomNodeFeatures;
import io.github.zzuegg.brainiac.experimental.graph.RandomWalkSequenceLayer;
import io.github.zzuegg.brainiac.experimental.graph.SubgraphEnsembleGnn;
import io.github.zzuegg.brainiac.experimental.graph.TopKPooling;
import io.github.zzuegg.brainiac.experimental.nn.GatLayer;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.GraphReadout;
import io.github.zzuegg.brainiac.experimental.nn.SageLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Graph-level structure: what message passing can and cannot see, and how to fix it
 * (wave 7, WP1), against GIN/GCN/GAT/GraphSAGE baselines:
 * <ul>
 *   <li>{@link #positionalEncodings} (7.1.1): GraphGPS with Laplacian and random-walk
 *       encodings on triangle counting, trees vs cycles and SBM node classification;</li>
 *   <li>{@link #overSquashing} (7.1.7, 7.1.17): the tree neighbours-match task, with a
 *       virtual node, FoSR rewiring, GraphGPS and NeuralWalker random-walk sequences;</li>
 *   <li>{@link #expressivity} (7.1.8): circular skip-link graphs that 1-WL cannot
 *       separate, with random node features, RWSE and ESAN subgraph bags;</li>
 *   <li>{@link #principalAggregation} (7.1.9): PNA on trees vs cycles, max-neighbour
 *       regression and graph-diameter regression;</li>
 *   <li>{@link #hierarchicalPooling} (7.1.10): DiffPool, top-k (gPool) and SAGPool on
 *       trees vs cycles and on two- vs three-community graphs.</li>
 * </ul>
 */
public final class GraphStructureShootout {

    private GraphStructureShootout() {
    }

    /** Embed, a stack of graph layers, a readout and a linear head. */
    private static final class Layered implements GraphTraining.GraphModel {
        private final Var embed;
        private final Var embedBias;
        private final List<MultiInputLayer> layers;
        private final boolean mask;
        private final Function<SyntheticGraphs.Batch, Tensor> extra;
        private final BiFunction<Var, SyntheticGraphs.Batch, Var> readout;
        private final Var head;
        private final Var headBias;
        private final java.util.Map<SyntheticGraphs.Batch, Tensor> cache = new java.util.IdentityHashMap<>();
        private boolean cacheExtra = true;

        Layered(int in, int width, int out, List<MultiInputLayer> layers, boolean mask,
                Function<SyntheticGraphs.Batch, Tensor> extra, BiFunction<Var, SyntheticGraphs.Batch, Var> readout,
                SplittableRandom r) {
            this.embed = Var.param(Tensor.normal(Shape.of(in, width), 0, (float) Math.sqrt(1.0 / in), r));
            this.embedBias = Var.param(Tensor.zeros(Shape.of(1, width)));
            this.layers = layers;
            this.mask = mask;
            this.extra = extra;
            this.readout = readout;
            this.head = Var.param(Tensor.normal(Shape.of(width, out), 0, (float) Math.sqrt(1.0 / width), r));
            this.headBias = Var.param(Tensor.zeros(Shape.of(1, out)));
        }

        Var nodes(SyntheticGraphs.Batch b, boolean training) {
            Tensor x = extra == null ? b.features()
                    : Tensor.concat(1, b.features(), cacheExtra ? cache.computeIfAbsent(b, extra) : extra.apply(b));
            Var h = Var.of(x).matmul(embed).add(embedBias);
            Var adj = Var.of(b.adjacency());
            Var m = mask ? Var.of(GraphGpsLayer.sameGraphMask(b.membership())) : null;
            for (MultiInputLayer l : layers) {
                h = m == null ? l.forward(List.of(h, adj), training) : l.forward(List.of(h, adj, m), training);
            }
            return h;
        }

        @Override
        public Var forward(SyntheticGraphs.Batch b, boolean training) {
            return readout.apply(nodes(b, training), b).matmul(head).add(headBias);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, embedBias, head, headBias));
            for (MultiInputLayer l : layers) {
                p.addAll(l.parameters());
            }
            return p;
        }
    }

    private static BiFunction<Var, SyntheticGraphs.Batch, Var> pooled(GraphReadout.Mode mode) {
        GraphReadout r = new GraphReadout(mode);
        return (h, b) -> r.forward(List.of(h, Var.of(b.membership())), true);
    }

    /** Reads the first node of every graph (the root of a neighbours-match tree). */
    private static Var roots(Var h, SyntheticGraphs.Batch b) {
        float[] m = b.membership().toArray();
        int g = b.membership().shape().dim(0);
        int n = b.membership().shape().dim(1);
        float[] idx = new float[g];
        for (int c = 0; c < g; c++) {
            for (int i = 0; i < n; i++) {
                if (m[c * n + i] > 0) {
                    idx[c] = i;
                    break;
                }
            }
        }
        return h.gather(0, Tensor.of(Shape.of(g), idx));
    }

    private static List<MultiInputLayer> layers(String kind, int count, int width, SplittableRandom r) {
        List<MultiInputLayer> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(switch (kind) {
                case "GIN" -> new GinLayer(width, width, width, r);
                case "GCN" -> new GcnLayer(width, width, Activation.RELU, r);
                case "SAGE" -> new SageLayer(width, width, Activation.RELU, false, r);
                case "GAT" -> new GatLayer(width, 2, width / 2, true, Activation.elu(), r);
                case "GPS" -> new GraphGpsLayer(width, 2, true, r);
                case "GIN-PE" -> new GraphGpsLayer(width, 2, false, r);
                case "Walker" -> new RandomWalkSequenceLayer(width, 8, 4, RandomWalkSequenceLayer.Mixer.CONVOLUTION, r);
                default -> throw new IllegalArgumentException(kind);
            });
        }
        return out;
    }

    private static Tensor rwse(SyntheticGraphs.Batch b) {
        return GraphPositionalEncodings.randomWalk(b.adjacency(), 8);
    }

    private static Tensor lap(SyntheticGraphs.Batch b) {
        return GraphPositionalEncodings.laplacian(b.adjacency(), b.membership(), 4);
    }

    private static double trainRegression(GraphTraining.GraphModel m, SyntheticGraphs.Batch train,
            SyntheticGraphs.Batch test, int steps, float lr) {
        float[] y = labels(train);
        Tensor target = Tensor.of(Shape.of(y.length, 1), y);
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            m.forward(train, true).sub(target).square().mean().backward();
            opt.step(m.parameters());
        }
        float[] pred = Var.noGrad(() -> m.forward(test, false)).value().toArray();
        float[] truth = labels(test);
        double mae = 0;
        for (int i = 0; i < truth.length; i++) {
            mae += Math.abs(pred[i] - truth[i]);
        }
        return mae / truth.length;
    }

    private static float[] labels(SyntheticGraphs.Batch b) {
        float[] y = new float[b.labels().length];
        for (int i = 0; i < y.length; i++) {
            y[i] = b.labels()[i];
        }
        return y;
    }

    private static double meanBaselineMae(SyntheticGraphs.Batch train, SyntheticGraphs.Batch test) {
        double mean = GraphTraining.mean(toDouble(labels(train)));
        double mae = 0;
        for (float t : labels(test)) {
            mae += Math.abs(t - mean);
        }
        return mae / test.labels().length;
    }

    private static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }

    /**
     * GraphGPS and positional encodings (7.1.1).
     *
     * @param graphs training graphs per graph task (half as many test graphs)
     * @param steps training steps per model
     * @param seed the seed
     * @return "triangles MAE model", "trees/cycles model" accuracies and "SBM model" node
     *     accuracies
     */
    public static Map<String, Double> positionalEncodings(int graphs, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SyntheticGraphs.Batch tri = SyntheticGraphs.batch(SyntheticGraphs.triangleCounting(graphs, 10, 0.35, seed));
        SyntheticGraphs.Batch triTest = SyntheticGraphs.batch(SyntheticGraphs.triangleCounting(graphs / 2, 10, 0.35,
                seed + 1));
        out.put("triangles MAE predict-mean", GraphTraining.round(meanBaselineMae(tri, triTest)));
        String[][] triModels = {{"GCN", ""}, {"GIN", ""}, {"GIN-PE", "RWSE"}, {"GIN-PE", "LapPE"}, {"GPS", "RWSE"},
            {"GPS", ""}};
        for (String[] spec : triModels) {
            SplittableRandom r = new SplittableRandom(seed);
            Function<SyntheticGraphs.Batch, Tensor> pe = spec[1].equals("RWSE") ? GraphStructureShootout::rwse
                    : spec[1].equals("LapPE") ? GraphStructureShootout::lap : null;
            int in = 1 + (pe == null ? 0 : spec[1].equals("RWSE") ? 8 : 4);
            Layered m = new Layered(in, 16, 1, layers(spec[0], 2, 16, r), spec[0].equals("GPS"), pe,
                    pooled(GraphReadout.Mode.SUM), r);
            out.put("triangles MAE " + name(spec), GraphTraining.round(trainRegression(m, tri, triTest, steps, 0.01f)));
        }
        SyntheticGraphs.Batch tc = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(graphs, seed));
        SyntheticGraphs.Batch tcTest = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(graphs / 2, seed + 1));
        for (String[] spec : new String[][] {{"GIN", ""}, {"GPS", "RWSE"}, {"GPS", "LapPE"}}) {
            SplittableRandom r = new SplittableRandom(seed);
            Function<SyntheticGraphs.Batch, Tensor> pe = spec[1].equals("RWSE") ? GraphStructureShootout::rwse
                    : spec[1].equals("LapPE") ? GraphStructureShootout::lap : null;
            int in = 1 + (pe == null ? 0 : spec[1].equals("RWSE") ? 8 : 4);
            Layered m = new Layered(in, 16, 2, layers(spec[0], 2, 16, r), spec[0].equals("GPS"), pe,
                    pooled(GraphReadout.Mode.SUM), r);
            out.put("trees/cycles " + name(spec),
                    GraphTraining.round(GraphTraining.trainGraphs(m, tc, tcTest, 2, steps, 0.01f)));
        }
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(3, 30, 0.25, 0.02, 2.0, seed);
        boolean[] train = GraphTraining.labelMask(g.nodeLabels(), 3, 0.1, seed);
        Tensor adj = g.adjacencyTensor();
        Tensor lapPe = GraphPositionalEncodings.laplacian(adj, null, 4);
        SplittableRandom r = new SplittableRandom(seed);
        out.put("SBM GCN", GraphTraining.round(GraphTraining.trainNodes(GraphTraining.gcn(3, 16, 3, r), g, adj, train,
                3, steps, 0.02f)));
        for (String kind : new String[] {"GIN", "GPS"}) {
            SplittableRandom rr = new SplittableRandom(seed);
            Var w = Var.param(Tensor.normal(Shape.of(7, 16), 0, 0.4f, rr));
            Var head = Var.param(Tensor.normal(Shape.of(16, 3), 0, 0.25f, rr));
            List<MultiInputLayer> stack = layers(kind.equals("GIN") ? "GIN" : "GPS", 2, 16, rr);
            SplittableRandom flips = new SplittableRandom(seed + 7);
            GraphTraining.NodeModel m = new GraphTraining.NodeModel() {
                @Override
                public Var forward(Var x, Tensor adjacency, boolean training) {
                    Tensor pe = training ? GraphPositionalEncodings.randomSignFlip(lapPe, flips) : lapPe;
                    Var h = Var.of(Tensor.concat(1, x.value(), pe)).matmul(w).relu();
                    for (MultiInputLayer layer : stack) {
                        h = layer.forward(List.of(h, Var.of(adjacency)), training);
                    }
                    return h.matmul(head);
                }

                @Override
                public List<Var> parameters() {
                    List<Var> p = new ArrayList<>(List.of(w, head));
                    for (MultiInputLayer layer : stack) {
                        p.addAll(layer.parameters());
                    }
                    return p;
                }
            };
            out.put("SBM " + kind + " + LapPE", GraphTraining.round(GraphTraining.trainNodes(m, g, adj, train, 3, steps,
                    0.02f)));
        }
        return out;
    }

    private static String name(String[] spec) {
        String base = spec[0].equals("GIN-PE") ? "GIN" : spec[0];
        return spec[1].isEmpty() ? base : base + " + " + spec[1];
    }

    private static List<SyntheticGraphs.Batch> minibatches(List<SyntheticGraphs.Graph> graphs, int size,
            Function<SyntheticGraphs.Batch, SyntheticGraphs.Batch> rewire) {
        List<SyntheticGraphs.Batch> out = new ArrayList<>();
        for (int i = 0; i < graphs.size(); i += size) {
            SyntheticGraphs.Batch b = SyntheticGraphs.batch(graphs.subList(i, Math.min(graphs.size(), i + size)));
            out.add(rewire == null ? b : rewire.apply(b));
        }
        return out;
    }

    private static double trainMinibatches(GraphTraining.GraphModel m, List<SyntheticGraphs.Batch> train,
            SyntheticGraphs.Batch test, int classes, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            SyntheticGraphs.Batch b = train.get(s % train.size());
            GraphTraining.crossEntropy(m.forward(b, true), b.labels(), null, classes).backward();
            opt.step(m.parameters());
        }
        Tensor logits = Var.noGrad(() -> m.forward(test, false)).value();
        return GraphTraining.accuracy(logits, test.labels(), null, true);
    }

    private static SyntheticGraphs.Batch withVirtualNode(SyntheticGraphs.Batch b) {
        GraphRewiring.Rewired v = GraphRewiring.virtualNode(b.adjacency(), b.features(), b.membership());
        int n = v.adjacency().shape().dim(0);
        return new SyntheticGraphs.Batch(v.adjacency(), v.features(), v.membership(), new int[n], b.labels());
    }

    private static SyntheticGraphs.Batch padFeature(SyntheticGraphs.Batch b) {
        int n = b.features().shape().dim(0);
        return new SyntheticGraphs.Batch(b.adjacency(), Tensor.concat(1, b.features(), Tensor.zeros(Shape.of(n, 1))),
                b.membership(), b.nodeLabels(), b.labels());
    }

    /**
     * Over-squashing on tree neighbours-match (7.1.7, 7.1.17).
     *
     * @param models model names: "GCN", "GIN", "GIN + virtual node", "GIN + FoSR", "GPS",
     *     "NeuralWalker"
     * @param depths tree depths
     * @param trees training trees (a quarter as many test trees)
     * @param steps training steps (one minibatch of 16 trees each)
     * @param seed the seed
     * @return "model @ depth" to root accuracy on held-out trees (chance = {@code 2^-depth})
     */
    public static Map<String, Double> overSquashing(List<String> models, int[] depths, int trees, int steps,
            long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (int depth : depths) {
            int leaves = 1 << depth;
            List<SyntheticGraphs.Graph> train = SyntheticGraphs.neighboursMatch(depth, trees, seed);
            List<SyntheticGraphs.Graph> test = SyntheticGraphs.neighboursMatch(depth, trees / 4, seed + 1);
            for (String name : models) {
                SplittableRandom r = new SplittableRandom(seed);
                Function<SyntheticGraphs.Batch, SyntheticGraphs.Batch> rewire = switch (name) {
                    case "GIN + virtual node" -> GraphStructureShootout::withVirtualNode;
                    case "GIN + FoSR" -> b -> padFeature(new SyntheticGraphs.Batch(
                            GraphRewiring.fosr(b.adjacency(), b.membership(), leaves), b.features(), b.membership(),
                            b.nodeLabels(), b.labels()));
                    default -> GraphStructureShootout::padFeature;
                };
                String kind = switch (name) {
                    case "GCN" -> "GCN";
                    case "GPS" -> "GPS";
                    case "NeuralWalker" -> "Walker";
                    default -> "GIN";
                };
                int layerCount = name.equals("NeuralWalker") || name.equals("GPS") ? 2 : depth + 1;
                Layered m = new Layered(2 * leaves + 1, 32, leaves, layers(kind, layerCount, 32, r), kind.equals("GPS"),
                        null, GraphStructureShootout::roots, r);
                double acc = trainMinibatches(m, minibatches(train, 16, rewire),
                        rewire.apply(SyntheticGraphs.batch(test)), leaves, steps, 0.005f);
                out.put(name + " @ " + depth, GraphTraining.round(acc));
            }
        }
        return out;
    }

    /** The ten standard CSL skip lengths for 41 nodes. */
    public static final int[] CSL_SKIPS = {2, 3, 4, 5, 6, 9, 11, 12, 13, 16};

    /** GIN, then a readout, then a head; with ESAN's bags or random node features. */
    private static GraphTraining.GraphModel cslModel(String name, SplittableRandom r) {
        if (name.equals("ESAN")) {
            // six GIN layers: the cycles that tell skip lengths apart are up to 17 edges long, so the
            // damage around a deleted node must travel several hops before the bags differ
            SubgraphEnsembleGnn esan = new SubgraphEnsembleGnn(1, 16, 10, 6, SubgraphEnsembleGnn.Policy.NODE_DELETED, 4,
                    r);
            return new GraphTraining.GraphModel() {
                @Override
                public Var forward(SyntheticGraphs.Batch b, boolean training) {
                    return esan.forward(List.of(Var.of(b.features()), Var.of(b.adjacency()), Var.of(b.membership())),
                            training);
                }

                @Override
                public List<Var> parameters() {
                    return esan.parameters();
                }
            };
        }
        RandomNodeFeatures rnf = new RandomNodeFeatures(4, r.split());
        Function<SyntheticGraphs.Batch, Tensor> extra = switch (name) {
            case "GIN + RWSE" -> GraphStructureShootout::rwse;
            case "GIN + RNF" -> b -> rnf.forward(Var.of(b.features())).value().slice(1, 1, 5);
            default -> null;
        };
        int in = 1 + (extra == null ? 0 : name.endsWith("RWSE") ? 8 : 4);
        Layered m = new Layered(in, 16, 10, layers("GIN", 3, 16, r), false, extra, pooled(GraphReadout.Mode.MEAN), r);
        m.cacheExtra = !name.endsWith("RNF");
        return m;
    }

    /**
     * Expressivity beyond 1-WL on circular skip-link graphs (7.1.8).
     *
     * @param models model names: "GIN", "GIN + RWSE", "GIN + RNF", "ESAN"
     * @param perClass training graphs per class (test: 2 per class, new permutations)
     * @param steps training steps
     * @param seed the seed
     * @return model to test accuracy (chance 0.1)
     */
    public static Map<String, Double> expressivity(List<String> models, int perClass, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SyntheticGraphs.Batch train = SyntheticGraphs.batch(SyntheticGraphs.circularSkipLinks(41, CSL_SKIPS, perClass,
                seed));
        SyntheticGraphs.Batch test = SyntheticGraphs.batch(SyntheticGraphs.circularSkipLinks(41, CSL_SKIPS, 2, seed + 1));
        for (String name : models) {
            GraphTraining.GraphModel m = cslModel(name, new SplittableRandom(seed));
            out.put(name, GraphTraining.round(GraphTraining.trainGraphs(m, train, test, 10, steps, 0.01f)));
        }
        return out;
    }

    /** Random graphs with uniform node features; node target: the max over neighbours. */
    private static SyntheticGraphs.Graph maxNeighbourGraph(int n, SplittableRandom r) {
        float[] adj = new float[n * n];
        for (int i = 0; i < n; i++) {
            double p = r.nextDouble() < 0.5 ? 0.1 : 0.5;
            for (int j = i + 1; j < n; j++) {
                if (r.nextDouble() < p) {
                    adj[i * n + j] = 1;
                    adj[j * n + i] = 1;
                }
            }
        }
        float[] feat = new float[n];
        for (int i = 0; i < n; i++) {
            feat[i] = (float) r.nextDouble();
        }
        return new SyntheticGraphs.Graph(n, adj, feat, 1, new int[n], 0);
    }

    private static float[] neighbourMax(SyntheticGraphs.Graph g) {
        int n = g.nodes();
        float[] t = new float[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (g.adjacency()[i * n + j] > 0) {
                    t[i] = Math.max(t[i], g.features()[j]);
                }
            }
        }
        return t;
    }

    private static int diameter(SyntheticGraphs.Graph g) {
        int n = g.nodes();
        int best = 0;
        for (int s = 0; s < n; s++) {
            int[] d = new int[n];
            java.util.Arrays.fill(d, -1);
            d[s] = 0;
            java.util.ArrayDeque<Integer> q = new java.util.ArrayDeque<>(List.of(s));
            while (!q.isEmpty()) {
                int v = q.poll();
                for (int u = 0; u < n; u++) {
                    if (g.adjacency()[v * n + u] > 0 && d[u] < 0) {
                        d[u] = d[v] + 1;
                        best = Math.max(best, d[u]);
                        q.add(u);
                    }
                }
            }
        }
        return best;
    }

    /**
     * PNA against single-aggregator layers (7.1.9).
     *
     * @param graphs training graphs per graph task (test sets are smaller)
     * @param steps training steps
     * @param seed the seed
     * @return "trees/cycles model" accuracy, "max-neighbour logMSE model" (log10 of the test
     *     MSE) and "diameter MAE model"
     */
    public static Map<String, Double> principalAggregation(int graphs, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        String[] names = {"PNA", "GIN", "GCN", "SAGE"};
        SyntheticGraphs.Batch tc = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(graphs, seed));
        SyntheticGraphs.Batch tcTest = SyntheticGraphs.batch(SyntheticGraphs.treesAndCycles(graphs / 2, seed + 1));
        float delta = PnaLayer.delta(tc.adjacency());
        for (String name : names) {
            SplittableRandom r = new SplittableRandom(seed);
            Layered m = new Layered(1, 8, 2, pnaLayersWidth(name, 8, delta, r), false, null, pooled(
                    name.equals("GIN") ? GraphReadout.Mode.SUM : GraphReadout.Mode.MEAN), r);
            out.put("trees/cycles " + name, GraphTraining.round(GraphTraining.trainGraphs(m, tc, tcTest, 2, steps,
                    0.01f)));
        }
        SplittableRandom data = new SplittableRandom(seed);
        SyntheticGraphs.Graph train = maxNeighbourGraph(80, data);
        SyntheticGraphs.Graph test = maxNeighbourGraph(80, data);
        float mdelta = PnaLayer.delta(train.adjacencyTensor());
        for (String name : names) {
            SplittableRandom r = new SplittableRandom(seed);
            List<MultiInputLayer> ls = pnaLayersWidth(name, 8, mdelta, r).subList(0, 1);
            Var in = Var.param(Tensor.normal(Shape.of(1, 8), 0, 1, r));
            Var inB = Var.param(Tensor.zeros(Shape.of(1, 8)));
            Var head = Var.param(Tensor.normal(Shape.of(8, 1), 0, 0.35f, r));
            Var headB = Var.param(Tensor.zeros(Shape.of(1, 1)));
            Function<SyntheticGraphs.Graph, Var> f = g -> ls.getFirst().forward(List.of(
                    Var.of(g.featureTensor()).matmul(in).add(inB), Var.of(g.adjacencyTensor())), true)
                    .matmul(head).add(headB);
            List<Var> params = new ArrayList<>(List.of(in, inB, head, headB));
            params.addAll(ls.getFirst().parameters());
            Tensor target = Tensor.of(Shape.of(80, 1), neighbourMax(train));
            Optimizer opt = Optimizer.adam(0.01f);
            for (int s = 0; s < steps; s++) {
                f.apply(train).sub(target).square().mean().backward();
                opt.step(params);
            }
            double mse = Var.noGrad(() -> f.apply(test).sub(Tensor.of(Shape.of(80, 1), neighbourMax(test))).square()
                    .mean()).value().item();
            out.put("max-neighbour logMSE " + name, GraphTraining.round(Math.log10(mse)));
        }
        List<SyntheticGraphs.Graph> dTrain = new ArrayList<>();
        List<SyntheticGraphs.Graph> dTest = new ArrayList<>();
        for (int i = 0; i < graphs + graphs / 2; i++) {
            SyntheticGraphs.Graph g = SyntheticGraphs.treesAndCycles(1, seed * 1000 + i).getFirst();
            g = new SyntheticGraphs.Graph(g.nodes(), g.adjacency(), g.features(), 1, g.nodeLabels(), diameter(g));
            (i < graphs ? dTrain : dTest).add(g);
        }
        SyntheticGraphs.Batch db = SyntheticGraphs.batch(dTrain);
        SyntheticGraphs.Batch dbTest = SyntheticGraphs.batch(dTest);
        float ddelta = PnaLayer.delta(db.adjacency());
        for (String name : names) {
            SplittableRandom r = new SplittableRandom(seed);
            Layered m = new Layered(1, 8, 1, pnaLayersWidth(name, 8, ddelta, r), false, null,
                    pooled(GraphReadout.Mode.SUM), r);
            double mae = trainRegression(m, db, dbTest, steps, 0.01f);
            out.put("diameter MAE " + name, GraphTraining.round(mae));
        }
        out.put("diameter MAE predict-mean", GraphTraining.round(meanBaselineMae(db, dbTest)));
        return out;
    }

    private static List<MultiInputLayer> pnaLayersWidth(String name, int width, float delta, SplittableRandom r) {
        return switch (name) {
            case "PNA" -> List.of(new PnaLayer(width, width, delta, r), new PnaLayer(width, width, delta, r));
            case "GIN" -> List.of(new GinLayer(width, width, width, r), new GinLayer(width, width, width, r));
            case "GCN" -> List.of(new GcnLayer(width, width, Activation.RELU, r),
                    new GcnLayer(width, width, Activation.RELU, r));
            default -> List.of(new SageLayer(width, width, Activation.RELU, false, r),
                    new SageLayer(width, width, Activation.RELU, false, r));
        };
    }

    /** GCN, pool, GCN, readout. */
    private static GraphTraining.GraphModel pooling(String name, DiffPool diff, SplittableRandom r) {
        GcnLayer g1 = new GcnLayer(1, 16, Activation.RELU, r);
        GcnLayer g2 = new GcnLayer(16, 16, Activation.RELU, r);
        Var head = Var.param(Tensor.normal(Shape.of(16, 2), 0, 0.25f, r));
        Var headB = Var.param(Tensor.zeros(Shape.of(1, 2)));
        Var coarse = Var.param(Tensor.normal(Shape.of(16, 16), 0, 0.25f, r));
        TopKPooling topk = name.equals("TopK") ? new TopKPooling(16, 0.5, TopKPooling.Score.PROJECTION, r)
                : name.equals("SAGPool") ? new TopKPooling(16, 0.5, TopKPooling.Score.GNN, r) : null;
        return new GraphTraining.GraphModel() {
            @Override
            public Var forward(SyntheticGraphs.Batch b, boolean training) {
                Var adj = Var.of(b.adjacency());
                Var h = g1.forward(List.of(Var.of(b.features()), adj), training);
                if (diff != null) {
                    DiffPool.Pooled p = diff.pool(h, b.adjacency(), b.membership());
                    int g = b.membership().shape().dim(0);
                    Tensor eye = Linalg.identity(4, p.adjacency().value().dtype(), p.adjacency().value().device())
                            .reshape(1, 4, 4);
                    Var a = p.adjacency().add(eye);
                    Var deg = a.sum(2, true);
                    Var z = a.matmul(p.features()).div(deg).matmul(coarse).relu();
                    Var pooled = z.sum(1, false).reshape(g, 16);
                    return pooled.matmul(head).add(headB);
                }
                if (topk != null) {
                    TopKPooling.Pooled p = topk.pool(h, b.adjacency(), b.membership());
                    Var h2 = g2.forward(List.of(p.features(), Var.of(p.adjacency())), training);
                    return new GraphReadout(GraphReadout.Mode.SUM).forward(List.of(h2, Var.of(p.membership())), training)
                            .matmul(head).add(headB);
                }
                Var h2 = g2.forward(List.of(h, adj), training);
                GraphReadout.Mode mode = name.equals("GCN + mean") ? GraphReadout.Mode.MEAN : GraphReadout.Mode.SUM;
                return new GraphReadout(mode).forward(List.of(h2, Var.of(b.membership())), training).matmul(head)
                        .add(headB);
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(g1.parameters());
                p.addAll(g2.parameters());
                p.addAll(List.of(head, headB));
                if (diff != null) {
                    p.addAll(diff.parameters());
                    p.add(coarse);
                }
                if (topk != null) {
                    p.addAll(topk.parameters());
                }
                return p;
            }
        };
    }

    /**
     * Hierarchical pooling (7.1.10).
     *
     * @param graphs training graphs per task (test sets are half as large)
     * @param steps training steps
     * @param seed the seed
     * @return "task model" to test accuracy, and "params model"
     */
    public static Map<String, Double> hierarchicalPooling(int graphs, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        Map<String, List<List<SyntheticGraphs.Graph>>> tasks = new LinkedHashMap<>();
        tasks.put("trees/cycles", List.of(SyntheticGraphs.treesAndCycles(graphs, seed),
                SyntheticGraphs.treesAndCycles(graphs / 2, seed + 1)));
        tasks.put("communities", List.of(SyntheticGraphs.communityCount(graphs, seed),
                SyntheticGraphs.communityCount(graphs / 2, seed + 1)));
        for (Map.Entry<String, List<List<SyntheticGraphs.Graph>>> t : tasks.entrySet()) {
            SyntheticGraphs.Batch train = SyntheticGraphs.batch(t.getValue().get(0));
            SyntheticGraphs.Batch test = SyntheticGraphs.batch(t.getValue().get(1));
            for (String name : new String[] {"GIN + sum", "GCN + mean", "DiffPool", "TopK", "SAGPool"}) {
                SplittableRandom r = new SplittableRandom(seed);
                DiffPool diff = name.equals("DiffPool") ? new DiffPool(16, 4, r) : null;
                GraphTraining.GraphModel model = name.equals("GIN + sum")
                        ? new Layered(1, 16, 2, layers("GIN", 2, 16, r), false, null, pooled(GraphReadout.Mode.SUM), r)
                        : pooling(name, diff, r);
                double acc = diff != null ? trainDiffPool(model, diff, train, test, steps)
                        : GraphTraining.trainGraphs(model, train, test, 2, steps, 0.01f);
                out.put(t.getKey() + " " + name, GraphTraining.round(acc));
                if (t.getKey().equals("trees/cycles")) {
                    out.put("params " + name, (double) model.parameters().stream()
                            .mapToLong(v -> v.value().size()).sum());
                }
            }
        }
        return out;
    }

    /** As {@link GraphTraining#trainGraphs}, plus DiffPool's auxiliary losses. */
    private static double trainDiffPool(GraphTraining.GraphModel m, DiffPool pool, SyntheticGraphs.Batch train,
            SyntheticGraphs.Batch test, int steps) {
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Var logits = m.forward(train, true);
            Var loss = GraphTraining.crossEntropy(logits, train.labels(), null, 2);
            loss.add(pool.auxiliaryLoss().mul(0.1f)).backward();
            opt.step(m.parameters());
        }
        Tensor logits = Var.noGrad(() -> m.forward(test, false)).value();
        return GraphTraining.accuracy(logits, test.labels(), null, true);
    }
}
