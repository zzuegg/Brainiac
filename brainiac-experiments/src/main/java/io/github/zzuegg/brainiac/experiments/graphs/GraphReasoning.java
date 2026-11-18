package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.AlgorithmTraces;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.graph.AlgorithmicProcessor;
import io.github.zzuegg.brainiac.experimental.graph.GraphGpsLayer;
import io.github.zzuegg.brainiac.experimental.graph.NeuroSat;
import io.github.zzuegg.brainiac.experimental.nn.GatLayer;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.GraphReadout;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Reasoning with graph networks (wave 7, WP1):
 * <ul>
 *   <li>{@link #algorithmic} (7.1.14): neural algorithmic reasoning, CLRS-lite. An
 *       {@link AlgorithmicProcessor} imitates BFS, Bellman-Ford or Prim step by step
 *       ({@link AlgorithmTraces}), trained on 8-16 node graphs and tested on 64-node graphs,
 *       with and without hint supervision and with max or sum aggregation, against
 *       recurrent GIN, GAT and GraphGPS processors trained on the output only;</li>
 *   <li>{@link #neuroSat} (7.1.15): NeuroSAT on SR(n) formula pairs against a GIN classifier
 *       on the same literal-clause graph, with WalkSAT as the non-learned reference.</li>
 * </ul>
 */
public final class GraphReasoning {

    private GraphReasoning() {
    }

    /** A recurrent processor: a state per step from node inputs and edge weights. */
    private interface Processor {
        List<Var> run(Var x, Tensor weights, int steps);

        Var nodeLogits(Var h);

        Var pointerLogits(Var h, Tensor weights);

        List<Var> parameters();
    }

    private static Processor ours(AlgorithmicProcessor.Aggregation aggregation, int width, SplittableRandom r) {
        AlgorithmicProcessor p = new AlgorithmicProcessor(1, width, 2, aggregation, r);
        return new Processor() {
            @Override
            public List<Var> run(Var x, Tensor weights, int steps) {
                return p.run(x, weights, steps);
            }

            @Override
            public Var nodeLogits(Var h) {
                return p.nodeLogits(h);
            }

            @Override
            public Var pointerLogits(Var h, Tensor weights) {
                return p.pointerLogits(h, weights);
            }

            @Override
            public List<Var> parameters() {
                return p.parameters();
            }
        };
    }

    /** One shared GIN, GAT or GraphGPS layer applied recurrently, with linear and bilinear heads. */
    private static Processor baseline(String kind, int width, SplittableRandom r) {
        MultiInputLayer layer = switch (kind) {
            case "GIN" -> new GinLayer(width, width, width, r);
            case "GAT" -> new GatLayer(width, 2, width / 2, true, Activation.elu(), r);
            default -> new GraphGpsLayer(width, 2, true, r);
        };
        Var enc = Var.param(Tensor.normal(Shape.of(1, width), 0, 1, r));
        Var node = Var.param(Tensor.normal(Shape.of(width, 2), 0, 0.3f, r));
        Var pa = Var.param(Tensor.normal(Shape.of(width, width), 0, 0.3f, r));
        Var pw = Var.param(Tensor.normal(Shape.of(1, 1), 0, 0.3f, r));
        return new Processor() {
            @Override
            public List<Var> run(Var x, Tensor weights, int steps) {
                int n = x.shape().dim(0);
                Tensor adj = weights.greater(Tensor.scalar(0f));
                Var e = x.matmul(enc);
                Var h = Var.of(Tensor.zeros(Shape.of(n, width)));
                List<Var> states = new ArrayList<>();
                for (int s = 0; s < steps; s++) {
                    h = layer.forward(List.of(h.add(e), Var.of(adj)), true).tanh();
                    states.add(h);
                }
                return states;
            }

            @Override
            public Var nodeLogits(Var h) {
                return h.matmul(node);
            }

            @Override
            public Var pointerLogits(Var h, Tensor weights) {
                int n = h.shape().dim(0);
                float[] w = weights.toArray();
                float[] pen = new float[n * n];
                for (int i = 0; i < n * n; i++) {
                    pen[i] = w[i] != 0 || i / n == i % n ? 0 : -1e9f;
                }
                return h.matmul(pa).matmul(h.transpose()).add(Var.of(weights).mul(pw))
                        .add(Tensor.of(Shape.of(n, n), pen));
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(layer.parameters());
                p.addAll(List.of(enc, node, pa, pw));
                return p;
            }
        };
    }

    private static Var decodeLoss(Processor p, AlgorithmTraces.Algorithm alg, Var h, Tensor w, int[] target) {
        Var logits = alg.pointer() ? p.pointerLogits(h, w) : p.nodeLogits(h);
        return GraphTraining.crossEntropy(logits, target, null, alg.pointer() ? target.length : 2);
    }

    private static Tensor inputs(AlgorithmTraces.Trace t) {
        float[] x = new float[t.nodes()];
        x[t.source()] = 1;
        return Tensor.of(Shape.of(t.nodes(), 1), x);
    }

    /**
     * Trains on small graphs, tests on large ones.
     *
     * @param algorithm the algorithm
     * @param models "max + hints", "max", "sum + hints", "GIN", "GAT", "GPS" (the baselines
     *     train on the output only)
     * @param testNodes the test graph size (training graphs have 8-16 nodes)
     * @param steps training steps (one graph each)
     * @param seed the seed
     * @return "model" to per-node accuracy of the output on test graphs, and "model (train
     *     size)" on held-out graphs of the training size
     */
    public static Map<String, Double> algorithmic(AlgorithmTraces.Algorithm algorithm, List<String> models,
            int testNodes, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        int width = 16;
        for (String name : models) {
            SplittableRandom r = new SplittableRandom(seed);
            boolean hints = name.endsWith("hints");
            Processor p = name.startsWith("max") ? ours(AlgorithmicProcessor.Aggregation.MAX, width, r)
                    : name.startsWith("sum") ? ours(AlgorithmicProcessor.Aggregation.SUM, width, r)
                    : baseline(name, width, r);
            Optimizer opt = Optimizer.adam(0.005f);
            SplittableRandom data = new SplittableRandom(seed + 1);
            for (int s = 0; s < steps; s++) {
                AlgorithmTraces.Trace t = AlgorithmTraces.generate(algorithm, 8 + data.nextInt(9), data.nextLong());
                Tensor w = Tensor.of(Shape.of(t.nodes(), t.nodes()), t.weights());
                List<Var> states = p.run(Var.of(inputs(t)), w, t.steps());
                Var loss;
                if (hints) {
                    loss = null;
                    for (int k = 0; k < t.steps(); k++) {
                        Var l = decodeLoss(p, algorithm, states.get(k), w, t.hints()[k]);
                        loss = loss == null ? l : loss.add(l);
                    }
                    loss = loss.div(t.steps()).add(decodeLoss(p, algorithm, states.getLast(), w, t.output()));
                } else {
                    loss = decodeLoss(p, algorithm, states.getLast(), w, t.output());
                }
                loss.backward();
                opt.step(p.parameters());
            }
            out.put(name + " (train size)", GraphTraining.round(evaluate(p, algorithm, 12, 10, seed + 7)));
            out.put(name, GraphTraining.round(evaluate(p, algorithm, testNodes, 4, seed + 9)));
        }
        return out;
    }

    private static double evaluate(Processor p, AlgorithmTraces.Algorithm algorithm, int nodes, int graphs, long seed) {
        int right = 0;
        int total = 0;
        SplittableRandom data = new SplittableRandom(seed);
        for (int g = 0; g < graphs; g++) {
            AlgorithmTraces.Trace t = AlgorithmTraces.generate(algorithm, nodes, data.nextLong());
            Tensor w = Tensor.of(Shape.of(nodes, nodes), t.weights());
            Tensor logits = Var.noGrad(() -> {
                Var h = p.run(Var.of(inputs(t)), w, t.steps()).getLast();
                return algorithm.pointer() ? p.pointerLogits(h, w) : p.nodeLogits(h);
            }).value();
            float[] pred = logits.argmax(1).toArray();
            for (int i = 0; i < nodes; i++) {
                right += Math.round(pred[i]) == t.output()[i] ? 1 : 0;
                total++;
            }
        }
        return right / (double) total;
    }

    /** WalkSAT: flip a variable of a random unsatisfied clause, greedily or at random. */
    static boolean[] walkSat(RandomSatFormulas.Formula f, int flips, SplittableRandom r) {
        int n = f.variables();
        boolean[] a = new boolean[n + 1];
        for (int v = 1; v <= n; v++) {
            a[v] = r.nextBoolean();
        }
        for (int s = 0; s < flips; s++) {
            List<int[]> unsat = new ArrayList<>();
            for (int[] c : f.clauses()) {
                boolean ok = false;
                for (int lit : c) {
                    ok |= a[Math.abs(lit)] == lit > 0;
                }
                if (!ok) {
                    unsat.add(c);
                }
            }
            if (unsat.isEmpty()) {
                return a;
            }
            int[] c = unsat.get(r.nextInt(unsat.size()));
            int pick;
            if (r.nextDouble() < 0.5) {
                pick = Math.abs(c[r.nextInt(c.length)]);
            } else {
                pick = Math.abs(c[0]);
                int best = Integer.MAX_VALUE;
                for (int lit : c) {
                    int v = Math.abs(lit);
                    a[v] = !a[v];
                    int broken = 0;
                    for (int[] d : f.clauses()) {
                        boolean ok = false;
                        for (int l : d) {
                            ok |= a[Math.abs(l)] == l > 0;
                        }
                        broken += ok ? 0 : 1;
                    }
                    a[v] = !a[v];
                    if (broken < best) {
                        best = broken;
                        pick = v;
                    }
                }
            }
            a[pick] = !a[pick];
        }
        return RandomSatFormulas.satisfies(f.clauses(), a) ? a : null;
    }

    private static List<RandomSatFormulas.Formula> pairs(int count, int n, SplittableRandom r) {
        List<RandomSatFormulas.Formula> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.addAll(List.of(RandomSatFormulas.srPair(n, r)));
        }
        return out;
    }

    private static int[] labels(List<RandomSatFormulas.Formula> fs) {
        return fs.stream().mapToInt(f -> f.satisfiable() ? 1 : 0).toArray();
    }

    /** The literal-clause graph for GIN: literal nodes, clause nodes and negation edges. */
    private static Tensor[] bipartite(List<RandomSatFormulas.Formula> fs) {
        NeuroSat.Batch b = NeuroSat.batch(fs);
        int nl = b.incidence().shape().dim(0);
        int nc = b.incidence().shape().dim(1);
        int n = nl + nc;
        float[] adj = new float[n * n];
        float[] m = b.incidence().toArray();
        float[] flip = b.flip().toArray();
        for (int l = 0; l < nl; l++) {
            for (int c = 0; c < nc; c++) {
                if (m[l * nc + c] > 0) {
                    adj[l * n + nl + c] = 1;
                    adj[(nl + c) * n + l] = 1;
                }
            }
            adj[l * n + (int) flip[l]] = 1;
        }
        float[] feat = new float[n * 2];
        float[] member = new float[fs.size() * n];
        float[] lm = b.membership().toArray();
        int co = 0;
        for (int g = 0; g < fs.size(); g++) {
            for (int l = 0; l < nl; l++) {
                member[g * n + l] = lm[g * nl + l];
            }
            for (int c = 0; c < fs.get(g).clauses().size(); c++) {
                member[g * n + nl + co + c] = 1;
            }
            co += fs.get(g).clauses().size();
        }
        for (int i = 0; i < n; i++) {
            feat[i * 2 + (i < nl ? 0 : 1)] = 1;
        }
        return new Tensor[] {Tensor.of(Shape.of(n, n), adj), Tensor.of(Shape.of(n, 2), feat),
            Tensor.of(Shape.of(fs.size(), n), member)};
    }

    private static Var binaryLoss(Var logits, int[] labels) {
        float[] y = new float[labels.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = labels[i];
        }
        return logits.softplus().sub(logits.mul(Tensor.of(logits.shape(), y))).mean();
    }

    private static double binaryAccuracy(Tensor logits, int[] labels) {
        float[] l = logits.toArray();
        int right = 0;
        for (int i = 0; i < labels.length; i++) {
            right += (l[i] > 0 ? 1 : 0) == labels[i] ? 1 : 0;
        }
        return right / (double) labels.length;
    }

    /**
     * NeuroSAT against a GIN classifier and WalkSAT.
     *
     * @param trainPairs SR(n) pairs for training
     * @param n training variables
     * @param testN test variable counts
     * @param rounds message-passing rounds in training
     * @param testRounds rounds tried at test time
     * @param steps training steps (batches of 8 pairs)
     * @param seed the seed
     * @return "NeuroSAT n=.. T=.." accuracy, "NeuroSAT decoded n=.. T=.." (fraction of
     *     satisfiable formulas solved by the decoded assignment), "GIN n=.." accuracy,
     *     "WalkSAT solved n=.." and "two random assignments solved n=.." (the chance level of
     *     decoding, which also tries two readings)
     */
    public static Map<String, Double> neuroSat(int trainPairs, int n, int[] testN, int rounds, int[] testRounds,
            int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        List<RandomSatFormulas.Formula> train = pairs(trainPairs, n, r);
        NeuroSat model = new NeuroSat(16, new SplittableRandom(seed));
        SplittableRandom ginRandom = new SplittableRandom(seed);
        List<GinLayer> gin = List.of(new GinLayer(2, 16, 16, ginRandom), new GinLayer(16, 16, 16, ginRandom),
                new GinLayer(16, 16, 16, ginRandom), new GinLayer(16, 16, 16, ginRandom));
        Var ginHead = Var.param(Tensor.normal(Shape.of(16, 1), 0, 0.1f, ginRandom));
        List<Var> ginParams = new ArrayList<>(List.of(ginHead));
        for (GinLayer l : gin) {
            ginParams.addAll(l.parameters());
        }
        GraphReadout readout = new GraphReadout(GraphReadout.Mode.MEAN);
        java.util.function.Function<Tensor[], Var> ginForward = g -> {
            Var h = Var.of(g[1]);
            for (GinLayer l : gin) {
                h = l.forward(List.of(h, Var.of(g[0])), true).tanh();
            }
            return readout.forward(List.of(h, Var.of(g[2])), true).matmul(ginHead).reshape(g[2].shape().dim(0));
        };
        Optimizer opt = Optimizer.adam(0.005f);
        Optimizer ginOpt = Optimizer.adam(0.002f);
        int batchPairs = 8;
        for (int s = 0; s < steps; s++) {
            int start = (s * 2 * batchPairs) % train.size();
            List<RandomSatFormulas.Formula> batch = train.subList(start, Math.min(train.size(), start + 2 * batchPairs));
            binaryLoss(model.logits(NeuroSat.batch(batch), rounds), labels(batch)).backward();
            opt.step(model.parameters());
            binaryLoss(ginForward.apply(bipartite(batch)), labels(batch)).backward();
            ginOpt.step(ginParams);
        }
        for (int tn : testN) {
            List<RandomSatFormulas.Formula> test = pairs(20, tn, r);
            NeuroSat.Batch tb = NeuroSat.batch(test);
            for (int t : testRounds) {
                Tensor logits = Var.noGrad(() -> model.logits(tb, t)).value();
                out.put("NeuroSAT n=" + tn + " T=" + t, GraphTraining.round(binaryAccuracy(logits, labels(test))));
                List<boolean[]> decoded = model.decode(tb, t);
                int solved = 0;
                int sat = 0;
                for (int i = 0; i < test.size(); i++) {
                    if (test.get(i).satisfiable()) {
                        sat++;
                        solved += decoded.get(i) != null ? 1 : 0;
                    }
                }
                out.put("NeuroSAT decoded n=" + tn + " T=" + t, GraphTraining.round(solved / (double) sat));
            }
            Tensor gl = Var.noGrad(() -> ginForward.apply(bipartite(test))).value();
            out.put("GIN n=" + tn, GraphTraining.round(binaryAccuracy(gl, labels(test))));
            int solved = 0;
            int sat = 0;
            SplittableRandom wr = new SplittableRandom(seed + 3);
            for (RandomSatFormulas.Formula f : test) {
                if (f.satisfiable()) {
                    sat++;
                    solved += walkSat(f, 10 * tn, wr) != null ? 1 : 0;
                }
            }
            out.put("WalkSAT solved n=" + tn, GraphTraining.round(solved / (double) sat));
            int lucky = 0;
            for (RandomSatFormulas.Formula f : test) {
                if (f.satisfiable()) {
                    for (int k = 0; k < 2; k++) {
                        boolean[] guess = new boolean[tn + 1];
                        for (int v = 1; v <= tn; v++) {
                            guess[v] = wr.nextBoolean();
                        }
                        if (RandomSatFormulas.satisfies(f.clauses(), guess)) {
                            lucky++;
                            break;
                        }
                    }
                }
            }
            out.put("two random assignments solved n=" + tn, GraphTraining.round(lucky / (double) sat));
        }
        return out;
    }
}
