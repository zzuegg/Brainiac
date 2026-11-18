package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FamilyTreeGraph;
import io.github.zzuegg.brainiac.experimental.graph.BellmanFordNetwork;
import io.github.zzuegg.brainiac.experimental.graph.HyperbolicEmbedding;
import io.github.zzuegg.brainiac.experimental.graph.KnowledgeGraphEmbedding;
import io.github.zzuegg.brainiac.experimental.graph.RandomWalkEmbedding;
import io.github.zzuegg.brainiac.experimental.graph.RelationalGcn;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Relational and hierarchical graph models (wave 10, items 10.5.13 to 10.5.15):
 * <ul>
 *   <li>{@link #linkPrediction}: R-GCN and CompGCN encoders with a DistMult decoder
 *       ({@link RelationalGcn}) and neural Bellman-Ford networks ({@link BellmanFordNetwork})
 *       against shallow {@link KnowledgeGraphEmbedding}s and common-neighbour / Adamic-Adar
 *       heuristics on {@link FamilyTreeGraph}, transductively and on an unseen family graph
 *       (inductive), with NBFNet's top edge checked against the planted
 *       {@code parent . parent = grandparent} rule;</li>
 *   <li>{@link #nodeClassification}: each person's generation from featureless inputs, where
 *       relation types (parent versus child) carry the signal and an untyped GCN cannot tell the
 *       directions apart;</li>
 *   <li>{@link #hierarchy}: Poincaré, Lorentz and Euclidean embeddings ({@link HyperbolicEmbedding})
 *       of the transitive closure of a random tree, and node2vec, by reconstruction mean rank and
 *       MAP, and the correlation of the embedding norm with depth.</li>
 * </ul>
 */
public final class RelationalGraphs {

    private RelationalGraphs() {
    }

    /** Scores every entity as the tail of a query. */
    @FunctionalInterface
    private interface Scorer {
        float[] tails(int head, int rel);
    }

    /**
     * @return {@code {filtered MRR, Hits@10}} over the test triples
     */
    private static double[] rank(Scorer scorer, List<int[]> test, Set<Long> known) {
        double rr = 0;
        double hits = 0;
        for (int[] t : test) {
            float[] s = scorer.tails(t[0], t[1]);
            int rank = 1;
            for (int e = 0; e < s.length; e++) {
                if (e != t[2] && s[e] > s[t[2]] && !known.contains(KnowledgeGraphEmbedding.key(t[0], t[1], e))) {
                    rank++;
                }
            }
            rr += 1.0 / rank;
            hits += rank <= 10 ? 1 : 0;
        }
        return new double[] {rr / test.size(), hits / test.size()};
    }

    private static Set<Long> known(FamilyTreeGraph.Data d) {
        Set<Long> k = new HashSet<>();
        for (int[] t : d.all()) {
            k.add(KnowledgeGraphEmbedding.key(t[0], t[1], t[2]));
        }
        return k;
    }

    /** Common neighbours (or Adamic-Adar) over the untyped context graph, ignoring the relation. */
    private static Scorer heuristic(List<int[]> context, int n, boolean adamicAdar) {
        List<Set<Integer>> nb = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            nb.add(new HashSet<>());
        }
        for (int[] t : context) {
            nb.get(t[0]).add(t[2]);
            nb.get(t[2]).add(t[0]);
        }
        return (h, r) -> {
            float[] s = new float[n];
            for (int v = 0; v < n; v++) {
                for (int w : nb.get(h)) {
                    if (nb.get(v).contains(w)) {
                        s[v] += adamicAdar ? (float) (1 / Math.log(1 + nb.get(w).size())) : 1;
                    }
                }
            }
            return s;
        };
    }

    /**
     * Link prediction, transductive and inductive (10.5.13, 10.5.14).
     *
     * @param families families in the training graph (and in the unseen inductive graph)
     * @param steps training steps per model
     * @param seed the seed
     * @return model to {@code {transductive MRR, Hits@10, inductive MRR (NaN if not inductive),
     *     inductive Hits@10}}; {@code "NBFNet rule recovery"} holds the share of grandparent test
     *     queries whose top edge lies on a planted parent-parent path
     */
    public static Map<String, double[]> linkPrediction(int families, int steps, long seed) {
        FamilyTreeGraph.Data data = FamilyTreeGraph.generate(families, 0.25, seed);
        FamilyTreeGraph.Data unseen = FamilyTreeGraph.generate(families, 0.25, seed + 100);
        Set<Long> known = known(data);
        Set<Long> knownUnseen = known(unseen);
        int n = data.entities();
        int m = unseen.entities();
        int rels = FamilyTreeGraph.RELATIONS.size();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (KnowledgeGraphEmbedding.Model model : new KnowledgeGraphEmbedding.Model[] {
            KnowledgeGraphEmbedding.Model.TRANSE, KnowledgeGraphEmbedding.Model.DISTMULT,
            KnowledgeGraphEmbedding.Model.COMPLEX, KnowledgeGraphEmbedding.Model.ROTATE}) {
            SplittableRandom r = new SplittableRandom(seed);
            KnowledgeGraphEmbedding kge = new KnowledgeGraphEmbedding(model, n, rels, 16, 6f, r);
            Optimizer opt = Optimizer.adam(0.02f);
            for (int s = 0; s < steps; s++) {
                List<int[]> batch = new ArrayList<>();
                for (int i = 0; i < 64; i++) {
                    batch.add(data.train().get(r.nextInt(data.train().size())));
                }
                kge.loss(batch, 16, 1f, r).backward();
                opt.step(kge.parameters());
            }
            double[] t = rank(kge::scoreTails, data.test(), known);
            out.put(model.name() + " (shallow)", new double[] {t[0], t[1], Double.NaN, Double.NaN});
        }
        Tensor adj = RelationalGcn.adjacency(data.train(), n, rels);
        Tensor adjUnseen = RelationalGcn.adjacency(unseen.train(), m, rels);
        for (RelationalGcn.Kind kind : new RelationalGcn.Kind[] {RelationalGcn.Kind.RGCN_BASIS,
            RelationalGcn.Kind.RGCN_BLOCK, RelationalGcn.Kind.COMPGCN_SUB, RelationalGcn.Kind.COMPGCN_MULT,
            RelationalGcn.Kind.COMPGCN_CORR}) {
            RelationalGcn enc = new RelationalGcn(kind, n, rels, 16, 2, 4, seed);
            enc.train(data.train(), n, steps, 32, 0.01f, seed);
            Var[] encoded = Var.noGrad(() -> enc.encode(adj, n));
            double[] t = rank((h, r) -> Var.noGrad(() -> enc.scores(encoded, new int[] {h}, new int[] {r}).value()
                    .toArray()), data.test(), known);
            RelationalGcn ind = new RelationalGcn(kind, 0, rels, 16, 2, 4, seed);
            ind.train(data.train(), n, steps, 32, 0.01f, seed);
            Var[] encodedUnseen = Var.noGrad(() -> ind.encode(adjUnseen, m));
            double[] u = rank((h, r) -> Var.noGrad(() -> ind.scores(encodedUnseen, new int[] {h}, new int[] {r})
                    .value().toArray()), unseen.test(), knownUnseen);
            out.put(kind.name(), new double[] {t[0], t[1], u[0], u[1]});
        }
        BellmanFordNetwork nbf = new BellmanFordNetwork(rels, 16, 3, seed);
        nbf.train(data.train(), n, steps, 16, 0.01f, seed);
        float[] a = BellmanFordNetwork.adjacency(data.train(), n, rels);
        float[] au = BellmanFordNetwork.adjacency(unseen.train(), m, rels);
        double[] t = rank((h, r) -> nbf.scoreTails(a, n, h, r), data.test(), known);
        double[] u = rank((h, r) -> nbf.scoreTails(au, m, h, r), unseen.test(), knownUnseen);
        out.put("NBFNet", new double[] {t[0], t[1], u[0], u[1]});
        for (boolean aa : new boolean[] {false, true}) {
            double[] ht = rank(heuristic(data.train(), n, aa), data.test(), known);
            double[] hu = rank(heuristic(unseen.train(), m, aa), unseen.test(), knownUnseen);
            out.put(aa ? "Adamic-Adar" : "common neighbours", new double[] {ht[0], ht[1], hu[0], hu[1]});
        }
        out.put("NBFNet rule recovery", new double[] {ruleRecovery(nbf, data, a)});
        return out;
    }

    /** The share of grandparent test queries whose top edge lies on a planted 2-hop parent path. */
    private static double ruleRecovery(BellmanFordNetwork nbf, FamilyTreeGraph.Data data, float[] a) {
        int n = data.entities();
        int rels = FamilyTreeGraph.RELATIONS.size();
        Set<Long> parent = new HashSet<>();
        for (int[] t : data.train()) {
            if (t[1] == FamilyTreeGraph.PARENT) {
                parent.add((long) t[0] * n + t[2]);
            }
            if (t[1] == FamilyTreeGraph.CHILD) {
                parent.add((long) t[2] * n + t[0]);
            }
        }
        int hit = 0;
        int total = 0;
        for (int[] t : data.test()) {
            if (t[1] != FamilyTreeGraph.GRANDPARENT) {
                continue;
            }
            int[] e = nbf.topEdge(a, n, t[0], t[1], t[2]);
            int rel = e[0] % rels;
            // the edge as written in the triple: inverse relations swap source and target
            int src = e[0] < rels ? e[1] : e[2];
            int dst = e[0] < rels ? e[2] : e[1];
            long pc = rel == FamilyTreeGraph.PARENT ? (long) src * n + dst
                    : rel == FamilyTreeGraph.CHILD ? (long) dst * n + src : -1;
            boolean onPath = false;
            if (pc >= 0 && parent.contains(pc)) {
                int p = (int) (pc / n);
                int c = (int) (pc % n);
                // the edge h -> mid or mid -> t on a path h -> mid -> t
                onPath = (p == t[0] && parent.contains((long) c * n + t[2]))
                        || (c == t[2] && parent.contains((long) t[0] * n + p));
            }
            hit += onPath ? 1 : 0;
            total++;
        }
        return total == 0 ? Double.NaN : hit / (double) total;
    }

    /**
     * Node classification by relation pattern (10.5.13): predict each person's generation (root,
     * child, grandchild) from featureless inputs, with half of the people labelled.
     *
     * @param families families
     * @param steps training steps
     * @param seed the seed
     * @return model to test accuracy
     */
    public static Map<String, Double> nodeClassification(int families, int steps, long seed) {
        FamilyTreeGraph.Data data = FamilyTreeGraph.generate(families, 0, seed);
        int n = data.entities();
        int rels = FamilyTreeGraph.RELATIONS.size();
        int[] gen = new int[n];
        boolean[] hasParent = new boolean[n];
        boolean[] hasGrand = new boolean[n];
        for (int[] t : data.all()) {
            if (t[1] == FamilyTreeGraph.PARENT) {
                hasParent[t[2]] = true;
            }
            if (t[1] == FamilyTreeGraph.GRANDPARENT) {
                hasGrand[t[2]] = true;
            }
        }
        for (int i = 0; i < n; i++) {
            gen[i] = hasGrand[i] ? 2 : hasParent[i] ? 1 : 0;
        }
        // only parent/child/sibling edges: the generation is not given away by grandparent edges
        List<int[]> graph = data.all().stream().filter(t -> t[1] != FamilyTreeGraph.GRANDPARENT).toList();
        int[] perm = new SplittableRandom(seed).ints(0, n).distinct().limit(n).toArray();
        boolean[] labelled = new boolean[n];
        for (int i = 0; i < n / 2; i++) {
            labelled[perm[i]] = true;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        Tensor adj = RelationalGcn.adjacency(graph, n, rels);
        for (RelationalGcn.Kind kind : new RelationalGcn.Kind[] {RelationalGcn.Kind.RGCN_BASIS,
            RelationalGcn.Kind.COMPGCN_MULT}) {
            RelationalGcn enc = new RelationalGcn(kind, 0, rels, 16, 2, 4, seed);
            Var head = Var.param(Tensor.uniform(Shape.of(16, 3), -0.5f, 0.5f, new SplittableRandom(seed)));
            List<Var> params = new ArrayList<>(enc.parameters());
            params.add(head);
            out.put(kind.name(), classify(() -> enc.encode(adj, n)[0].matmul(head), params, gen, labelled, steps));
        }
        float[] und = new float[n * n];
        for (int[] t : graph) {
            und[t[0] * n + t[2]] = 1;
            und[t[2] * n + t[0]] = 1;
        }
        Var adjU = Var.of(Tensor.of(Shape.of(n, n), und));
        SplittableRandom r = new SplittableRandom(seed);
        GcnLayer g1 = new GcnLayer(16, 16, Activation.RELU, r);
        GcnLayer g2 = new GcnLayer(16, 3, Activation.IDENTITY, r);
        List<Var> gp = new ArrayList<>(g1.parameters());
        gp.addAll(g2.parameters());
        Var ones = Var.of(Tensor.full(Shape.of(n, 16), 1f));
        out.put("GCN (untyped)", classify(() -> g2.forward(List.of(g1.forward(List.of(ones, adjU), true), adjU),
                true), gp, gen, labelled, steps));
        return out;
    }

    private static double classify(java.util.function.Supplier<Var> logits, List<Var> params, int[] labels,
            boolean[] labelled, int steps) {
        int n = labels.length;
        float[] mask = new float[n];
        float[] y = new float[n * 3];
        int count = 0;
        for (int i = 0; i < n; i++) {
            mask[i] = labelled[i] ? 1 : 0;
            y[i * 3 + labels[i]] = 1;
            count += labelled[i] ? 1 : 0;
        }
        Tensor m = Tensor.of(Shape.of(n, 1), mask);
        Tensor onehot = Tensor.of(Shape.of(n, 3), y);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int s = 0; s < steps; s++) {
            Var l = logits.get();
            Var logp = l.sub(l.logsumexp(-1, true));
            logp.mul(onehot).mul(m).sum().neg().div(count).backward();
            opt.step(params);
        }
        float[] p = Var.noGrad(() -> logits.get()).value().toArray();
        int right = 0;
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (labelled[i]) {
                continue;
            }
            int best = 0;
            for (int k = 1; k < 3; k++) {
                if (p[i * 3 + k] > p[i * 3 + best]) {
                    best = k;
                }
            }
            right += best == labels[i] ? 1 : 0;
            total++;
        }
        return right / (double) total;
    }

    /** A random tree's transitive closure: edges from every node to each of its ancestors. */
    record Closure(int nodes, int[] depth, List<int[]> edges) {
    }

    static Closure closure(int depth, int maxChildren, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Integer> parent = new ArrayList<>(List.of(-1));
        List<Integer> level = new ArrayList<>(List.of(0));
        List<Integer> frontier = new ArrayList<>(List.of(0));
        for (int d = 1; d <= depth; d++) {
            List<Integer> next = new ArrayList<>();
            for (int p : frontier) {
                int k = 1 + r.nextInt(maxChildren);
                for (int c = 0; c < k; c++) {
                    parent.add(p);
                    level.add(d);
                    next.add(parent.size() - 1);
                }
            }
            frontier = next;
        }
        int n = parent.size();
        List<int[]> edges = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            for (int a = parent.get(v); a >= 0; a = parent.get(a)) {
                edges.add(new int[] {v, a});
            }
        }
        return new Closure(n, level.stream().mapToInt(Integer::intValue).toArray(), edges);
    }

    /**
     * Hierarchy reconstruction (10.5.15).
     *
     * @param depth tree depth
     * @param dims embedding dimensions to try
     * @param epochs training epochs
     * @param seed the seed
     * @return {@code "<geometry> <dim>d"} to {@code {mean rank, MAP, Pearson(norm, depth), millis}}
     */
    public static Map<String, double[]> hierarchy(int depth, int[] dims, int epochs, long seed) {
        Closure c = closure(depth, 3, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("tree", new double[] {c.nodes(), c.edges().size()});
        for (int dim : dims) {
            for (HyperbolicEmbedding.Geometry g : HyperbolicEmbedding.Geometry.values()) {
                HyperbolicEmbedding e = new HyperbolicEmbedding(g, c.nodes(), dim, seed);
                long t0 = System.currentTimeMillis();
                e.train(c.edges(), epochs, 10, g == HyperbolicEmbedding.Geometry.EUCLIDEAN ? 0.1 : 0.3, 5, seed);
                long ms = System.currentTimeMillis() - t0;
                double[] rec = reconstruction(c, (u, v) -> -e.distance(u, v));
                double[] norm = new double[c.nodes()];
                double[] dep = new double[c.nodes()];
                for (int i = 0; i < c.nodes(); i++) {
                    norm[i] = e.norm(i);
                    dep[i] = c.depth()[i];
                }
                out.put(g.name().toLowerCase(java.util.Locale.ROOT) + " " + dim + "d",
                        new double[] {rec[0], rec[1], pearson(norm, dep), ms});
            }
        }
        float[] adj = new float[c.nodes() * c.nodes()];
        for (int[] e : c.edges()) {
            adj[e[0] * c.nodes() + e[1]] = 1;
            adj[e[1] * c.nodes() + e[0]] = 1;
        }
        int dim = dims[dims.length - 1];
        long t0 = System.currentTimeMillis();
        RandomWalkEmbedding n2v = RandomWalkEmbedding.train(Tensor.of(Shape.of(c.nodes(), c.nodes()), adj), dim, 10,
                20, 1, 1, 5, 2, seed);
        long ms = System.currentTimeMillis() - t0;
        double[] rec = reconstruction(c, n2v::score);
        out.put("node2vec " + dim + "d", new double[] {rec[0], rec[1], Double.NaN, ms});
        return out;
    }

    /** Similarity of two nodes, higher is closer. */
    @FunctionalInterface
    private interface Similarity {
        double of(int u, int v);
    }

    /** @return {@code {mean rank, MAP}} of every node's closure neighbours among all other nodes */
    private static double[] reconstruction(Closure c, Similarity sim) {
        int n = c.nodes();
        List<Set<Integer>> nb = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            nb.add(new HashSet<>());
        }
        for (int[] e : c.edges()) {
            nb.get(e[0]).add(e[1]);
            nb.get(e[1]).add(e[0]);
        }
        double rankSum = 0;
        int pairs = 0;
        double apSum = 0;
        int nodes = 0;
        for (int u = 0; u < n; u++) {
            if (nb.get(u).isEmpty()) {
                continue;
            }
            double[] s = new double[n];
            for (int v = 0; v < n; v++) {
                s[v] = v == u ? Double.NEGATIVE_INFINITY : sim.of(u, v);
            }
            double[] negatives = new double[n];
            int k = 0;
            for (int v = 0; v < n; v++) {
                if (v != u && !nb.get(u).contains(v)) {
                    negatives[k++] = s[v];
                }
            }
            double[] neg = Arrays.copyOf(negatives, k);
            Arrays.sort(neg);
            double[] pos = nb.get(u).stream().mapToDouble(v -> s[v]).sorted().toArray();
            double ap = 0;
            for (int i = 0; i < pos.length; i++) {
                // negatives scoring above this positive
                int above = k - upperBound(neg, pos[i]);
                rankSum += 1 + above;
                pairs++;
                int posAbove = pos.length - 1 - i;   // positives ranked above (sorted ascending)
                ap += (posAbove + 1.0) / (posAbove + 1 + above);
            }
            apSum += ap / pos.length;
            nodes++;
        }
        return new double[] {rankSum / pairs, apSum / nodes};
    }

    private static int upperBound(double[] sorted, double v) {
        int lo = 0;
        int hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private static double pearson(double[] a, double[] b) {
        double ma = Arrays.stream(a).average().orElse(0);
        double mb = Arrays.stream(b).average().orElse(0);
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += (a[i] - ma) * (b[i] - mb);
            aa += (a[i] - ma) * (a[i] - ma);
            bb += (b[i] - mb) * (b[i] - mb);
        }
        return ab / Math.sqrt(aa * bb + 1e-12);
    }

    /**
     * Prints the larger runs.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        linkPrediction(15, 600, 1).forEach((k, v) -> System.out.printf("link %s %s%n", k, Arrays.toString(v)));
        nodeClassification(15, 300, 1).forEach((k, v) -> System.out.printf("nodes %s %.3f%n", k, v));
        hierarchy(7, new int[] {2, 5, 10}, 100, 1).forEach((k, v) -> System.out.printf("hierarchy %s %s%n", k,
                Arrays.toString(v)));
    }
}
