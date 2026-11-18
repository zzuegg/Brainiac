package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.experimental.kernels.TensorExpressionGraph.Op;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * An e-graph for equality saturation (Willsey et al., "egg", POPL 2021,
 * <a href="https://arxiv.org/abs/2004.03082">arXiv:2004.03082</a>) over
 * {@link TensorExpressionGraph}. It has the usual parts: e-classes in a union-find, a
 * hash-consed node table, and a rebuild that restores congruence after unions. Rewrites
 * ({@link Rule}) only add nodes and unions, so applying them in any order loses nothing, and
 * {@link #saturate} runs them until nothing changes or a node limit is reached.
 * {@link #extract} picks the cheapest equivalent graph by
 * {@link TensorExpressionGraph#cost}.
 *
 * <p>The built-in rules ({@link #rules}) are TASO-style substitutions:
 * <ul>
 * <li>constant folding;</li>
 * <li>moving a scalar scale through a matmul into the weight;</li>
 * <li>fusing the matmuls that share an input into one matmul with concatenated weights, read
 *     back by slices;</li>
 * <li>re-association of additions, which lets bias chains fold;</li>
 * <li>folding an inference batch norm into the preceding convolution.</li>
 * </ul>
 *
 * <p>Extraction is sharing-aware: a fused matmul feeds several slices. The first pass
 * therefore charges each slice its width's share of its producer's cost. Local single-node
 * switches then lower the true DAG cost. This is a heuristic, not egg's ILP extractor.
 */
public final class EGraph {

    /**
     * An e-node: an operation over e-classes.
     *
     * @param op the operation
     * @param children the child classes
     * @param payload as in {@link TensorExpressionGraph.Node}
     */
    public record ENode(Op op, int[] children, Object payload) {
        @Override
        public boolean equals(Object o) {
            return o instanceof ENode n && n.op == op && Arrays.equals(n.children, children)
                    && (op == Op.CONST ? n.payload == payload : Objects.deepEquals(n.payload, payload));
        }

        @Override
        public int hashCode() {
            return Objects.hash(op, Arrays.hashCode(children),
                    op == Op.CONST ? System.identityHashCode(payload)
                            : payload instanceof int[] a ? Arrays.hashCode(a) : Objects.hashCode(payload));
        }
    }

    /** A rewrite: adds nodes and unions, returns how many unions changed something. */
    public interface Rule {
        /** @return the rule's name */
        String name();

        /**
         * @param g the e-graph
         * @return effective unions
         */
        int apply(EGraph g);
    }

    private final List<Integer> parent = new ArrayList<>();
    private final List<Set<ENode>> classNodes = new ArrayList<>();
    private final List<Shape> shapes = new ArrayList<>();
    private final Map<ENode, Integer> memo = new HashMap<>();
    private final List<Integer> roots = new ArrayList<>();
    private final Map<String, Integer> applied = new TreeMap<>();
    private final Map<Tensor, Map<Tensor, int[]>> derived = new java.util.IdentityHashMap<>();

    /** An empty e-graph. */
    public EGraph() {
        // filled by add and of
    }

    /**
     * Loads a graph; its outputs become the roots.
     *
     * @param g the graph
     * @return the e-graph
     */
    public static EGraph of(TensorExpressionGraph g) {
        EGraph e = new EGraph();
        List<TensorExpressionGraph.Node> nodes = g.nodes();
        int[] cls = new int[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            TensorExpressionGraph.Node n = nodes.get(i);
            int[] ch = new int[n.inputs().length];
            for (int j = 0; j < ch.length; j++) {
                ch[j] = cls[n.inputs()[j]];
            }
            cls[i] = e.add(new ENode(n.op(), ch, n.payload()), n.shape());
        }
        for (int o : g.outputs()) {
            e.roots.add(cls[o]);
        }
        return e;
    }

    /**
     * @param c a class id
     * @return its canonical id
     */
    public int find(int c) {
        int r = c;
        while (parent.get(r) != r) {
            r = parent.get(r);
        }
        while (parent.get(c) != r) {
            int next = parent.get(c);
            parent.set(c, r);
            c = next;
        }
        return r;
    }

    private ENode canonical(ENode n) {
        int[] ch = new int[n.children().length];
        for (int i = 0; i < ch.length; i++) {
            ch[i] = find(n.children()[i]);
        }
        return new ENode(n.op(), ch, n.payload());
    }

    /**
     * Adds a node (or finds it).
     *
     * @param n the node
     * @param shape its shape
     * @return its class
     */
    public int add(ENode n, Shape shape) {
        ENode c = canonical(n);
        Integer found = memo.get(c);
        if (found != null) {
            return find(found);
        }
        int id = parent.size();
        parent.add(id);
        Set<ENode> set = new HashSet<>();
        set.add(c);
        classNodes.add(set);
        shapes.add(shape);
        memo.put(c, id);
        return id;
    }

    /**
     * Adds a node, inferring its shape.
     *
     * @param op the operation
     * @param payload the payload
     * @param children the child classes
     * @return its class
     */
    public int add(Op op, Object payload, int... children) {
        Shape[] in = new Shape[children.length];
        for (int i = 0; i < in.length; i++) {
            in[i] = shape(children[i]);
        }
        Shape s = op == Op.CONST ? ((Tensor) payload).shape() : TensorExpressionGraph.shapeOf(op, payload, in);
        return add(new ENode(op, children, payload), s);
    }

    /**
     * @param a a class
     * @param b another
     * @return whether they were different
     */
    public boolean union(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return false;
        }
        if (ra > rb) {
            int t = ra;
            ra = rb;
            rb = t;
        }
        parent.set(rb, ra);
        classNodes.get(ra).addAll(classNodes.get(rb));
        classNodes.get(rb).clear();
        return true;
    }

    /** Restores the hash-cons and congruence closure after unions. */
    public void rebuild() {
        boolean changed = true;
        while (changed) {
            changed = false;
            memo.clear();
            for (int c = 0; c < parent.size(); c++) {
                if (find(c) != c) {
                    continue;
                }
                Set<ENode> canon = new HashSet<>();
                for (ENode n : classNodes.get(c)) {
                    canon.add(canonical(n));
                }
                classNodes.set(c, canon);
            }
            for (int c = 0; c < parent.size(); c++) {
                if (find(c) != c) {
                    continue;
                }
                for (ENode n : List.copyOf(classNodes.get(find(c)))) {
                    Integer other = memo.putIfAbsent(n, find(c));
                    if (other != null && find(other) != find(c)) {
                        union(other, c);
                        changed = true;
                    }
                }
            }
        }
    }

    /** @return the canonical classes */
    public List<Integer> classes() {
        List<Integer> out = new ArrayList<>();
        for (int c = 0; c < parent.size(); c++) {
            if (find(c) == c) {
                out.add(c);
            }
        }
        return out;
    }

    /**
     * @param c a class
     * @return its nodes (canonical after {@link #rebuild})
     */
    public Set<ENode> nodes(int c) {
        return Set.copyOf(classNodes.get(find(c)));
    }

    /**
     * @param c a class
     * @return its shape
     */
    public Shape shape(int c) {
        return shapes.get(find(c));
    }

    /**
     * @param c a class
     * @return its constant value, or null
     */
    public Tensor constant(int c) {
        for (ENode n : classNodes.get(find(c))) {
            if (n.op() == Op.CONST) {
                return (Tensor) n.payload();
            }
        }
        return null;
    }

    /** @return the number of e-nodes */
    public int size() {
        int s = 0;
        for (int c : classes()) {
            s += classNodes.get(c).size();
        }
        return s;
    }

    /** @return effective unions per rule name so far */
    public Map<String, Integer> applied() {
        return Map.copyOf(applied);
    }

    /**
     * Runs rules until no union changes anything.
     *
     * @param rules the rules
     * @param maxIterations an iteration cap
     * @param maxNodes a node cap
     * @return the iterations run
     */
    public int saturate(List<Rule> rules, int maxIterations, int maxNodes) {
        for (int it = 1; it <= maxIterations; it++) {
            int changed = 0;
            for (Rule r : rules) {
                int k = r.apply(this);
                rebuild();
                applied.merge(r.name(), k, Integer::sum);
                changed += k;
                if (size() > maxNodes) {
                    return it;
                }
            }
            if (changed == 0) {
                return it;
            }
        }
        return maxIterations;
    }

    // ---------------------------------------------------------------- rules

    /** @return constant folding, scale motion, matmul fusion, add re-association, BN folding */
    public static List<Rule> rules() {
        return List.of(constantFolding(), scaleIntoWeight(), fuseSharedMatmuls(), reassociateAdd(), foldBatchNorm());
    }

    private static Rule rule(String name, java.util.function.ToIntFunction<EGraph> body) {
        return new Rule() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public int apply(EGraph g) {
                return body.applyAsInt(g);
            }
        };
    }

    /** @return folds nodes over constant classes into constants */
    public static Rule constantFolding() {
        return rule("constant-fold", g -> {
            int k = 0;
            for (int c : g.classes()) {
                if (g.constant(c) != null) {
                    continue;
                }
                for (ENode n : g.nodes(c)) {
                    if (n.op() == Op.INPUT || n.op() == Op.CONST) {
                        continue;
                    }
                    Tensor[] in = new Tensor[n.children().length];
                    boolean all = true;
                    for (int i = 0; i < in.length && all; i++) {
                        in[i] = g.constant(n.children()[i]);
                        all = in[i] != null;
                    }
                    if (all) {
                        Tensor v = TensorExpressionGraph.apply(new TensorExpressionGraph.Node(n.op(), new int[0],
                                n.payload(), g.shape(c)), in);
                        k += g.union(c, g.add(Op.CONST, v)) ? 1 : 0;
                        break;
                    }
                }
            }
            return k;
        });
    }

    /** @return {@code (s x) W = x (s W)} and {@code s (x W) = x (s W)} */
    public static Rule scaleIntoWeight() {
        return rule("scale-into-weight", g -> {
            int k = 0;
            for (int c : g.classes()) {
                for (ENode n : g.nodes(c)) {
                    if (n.op() == Op.MATMUL) {
                        for (ENode a : g.nodes(n.children()[0])) {
                            if (a.op() == Op.SCALE) {
                                int sw = g.add(Op.SCALE, a.payload(), n.children()[1]);
                                k += g.union(c, g.add(Op.MATMUL, null, a.children()[0], sw)) ? 1 : 0;
                            }
                        }
                    } else if (n.op() == Op.SCALE) {
                        for (ENode a : g.nodes(n.children()[0])) {
                            if (a.op() == Op.MATMUL) {
                                int sw = g.add(Op.SCALE, n.payload(), a.children()[1]);
                                k += g.union(c, g.add(Op.MATMUL, null, a.children()[0], sw)) ? 1 : 0;
                            }
                        }
                    }
                }
            }
            return k;
        });
    }

    /** @return {@code x W_i = slice_i(x [W_1 ... W_k])} for constant 2-D weights sharing {@code x} */
    public static Rule fuseSharedMatmuls() {
        return rule("fuse-matmuls", g -> {
            Map<Integer, Map<Integer, Integer>> byInput = new TreeMap<>();
            for (int c : g.classes()) {
                for (ENode n : g.nodes(c)) {
                    if (n.op() != Op.MATMUL) {
                        continue;
                    }
                    int w = g.find(n.children()[1]);
                    if (g.constant(w) == null || g.shape(w).rank() != 2 || g.hasOp(w, Op.CONCAT)) {
                        continue;
                    }
                    byInput.computeIfAbsent(g.find(n.children()[0]), x -> new TreeMap<>()).put(w, c);
                }
            }
            int k = 0;
            for (Map.Entry<Integer, Map<Integer, Integer>> e : byInput.entrySet()) {
                Map<Integer, Integer> ws = e.getValue();
                if (ws.size() < 2) {
                    continue;
                }
                int[] weights = ws.keySet().stream().mapToInt(Integer::intValue).toArray();
                int cat = g.add(Op.CONCAT, null, weights);
                int fused = g.add(Op.MATMUL, null, e.getKey(), cat);
                int off = 0;
                for (int w : weights) {
                    int width = g.shape(w).dim(1);
                    int slice = g.add(Op.SLICE, new int[] {off, off + width}, fused);
                    k += g.union(ws.get(w), slice) ? 1 : 0;
                    off += width;
                }
            }
            return k;
        });
    }

    private boolean hasOp(int c, Op op) {
        for (ENode n : classNodes.get(find(c))) {
            if (n.op() == op) {
                return true;
            }
        }
        return false;
    }

    /** @return {@code (a + b) + c = a + (b + c)} */
    public static Rule reassociateAdd() {
        return rule("reassociate-add", g -> {
            int k = 0;
            for (int c : g.classes()) {
                for (ENode n : g.nodes(c)) {
                    if (n.op() != Op.ADD) {
                        continue;
                    }
                    for (ENode a : g.nodes(n.children()[0])) {
                        if (a.op() == Op.ADD) {
                            int bc = g.add(Op.ADD, null, a.children()[1], n.children()[1]);
                            k += g.union(c, g.add(Op.ADD, null, a.children()[0], bc)) ? 1 : 0;
                        }
                    }
                }
            }
            return k;
        });
    }

    /** @return {@code affine(conv(x, W), s, t) = conv(x, s W) + t} for constant {@code W, s, t} */
    public static Rule foldBatchNorm() {
        return rule("fold-batch-norm", g -> {
            int k = 0;
            for (int c : g.classes()) {
                for (ENode n : g.nodes(c)) {
                    if (n.op() != Op.AFFINE) {
                        continue;
                    }
                    Tensor s = g.constant(n.children()[1]);
                    Tensor t = g.constant(n.children()[2]);
                    if (s == null || t == null) {
                        continue;
                    }
                    for (ENode conv : g.nodes(n.children()[0])) {
                        Tensor w = conv.op() == Op.CONV ? g.constant(conv.children()[1]) : null;
                        if (w == null) {
                            continue;
                        }
                        int f = w.shape().dim(0);
                        int[] consts = g.derived.computeIfAbsent(w, key -> new java.util.IdentityHashMap<>())
                                .computeIfAbsent(s, key -> new int[] {
                                    g.add(Op.CONST, w.mul(s.reshape(f, 1, 1, 1))),
                                    g.add(Op.CONST, t.reshape(1, f, 1, 1))});
                        int nc = g.add(Op.CONV, conv.payload(), conv.children()[0], consts[0]);
                        int shift = consts[1];
                        k += g.union(c, g.add(Op.ADD, null, nc, shift)) ? 1 : 0;
                    }
                }
            }
            return k;
        });
    }

    // ---------------------------------------------------------------- extraction

    private double nodeCost(int cls, ENode n) {
        if (n.op() == Op.CONST || n.op() == Op.INPUT) {
            return 0;
        }
        Shape[] in = new Shape[n.children().length];
        for (int i = 0; i < in.length; i++) {
            in[i] = shape(n.children()[i]);
        }
        return TensorExpressionGraph.flops(n.op(), shape(cls), in)
                + (n.op() == Op.SLICE ? 0 : TensorExpressionGraph.KERNEL_OVERHEAD);
    }

    /**
     * Extracts the cheapest equivalent graph.
     *
     * @return the graph, its outputs in root order
     */
    public TensorExpressionGraph extract() {
        rebuild();
        Map<Integer, ENode> amortized = treeChoice(true);
        Map<Integer, ENode> plain = treeChoice(false);
        Map<Integer, ENode> best = dagCost(amortized) <= dagCost(plain) ? amortized : plain;
        boolean improved = true;
        for (int pass = 0; pass < 5 && improved; pass++) {
            improved = false;
            double current = dagCost(best);
            for (int c : classes()) {
                for (ENode alt : classNodes.get(c)) {
                    if (alt.equals(best.get(c))) {
                        continue;
                    }
                    Map<Integer, ENode> trial = new HashMap<>(best);
                    trial.put(c, alt);
                    double cost = dagCost(trial);
                    if (cost < current - 1e-9) {
                        best = trial;
                        current = cost;
                        improved = true;
                    }
                }
            }
        }
        TensorExpressionGraph out = new TensorExpressionGraph();
        Map<Integer, Integer> built = new HashMap<>();
        for (int r : roots) {
            out.output(build(find(r), best, out, built));
        }
        return out;
    }

    private Map<Integer, ENode> treeChoice(boolean amortize) {
        Map<Integer, Double> cost = new HashMap<>();
        Map<Integer, ENode> choice = new HashMap<>();
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int c : classes()) {
                for (ENode n : classNodes.get(c)) {
                    double total = nodeCost(c, n);
                    for (int ch : n.children()) {
                        Double cc = cost.get(find(ch));
                        if (cc == null) {
                            total = Double.POSITIVE_INFINITY;
                            break;
                        }
                        total += n.op() == Op.SLICE && amortize ? cc * shape(c).dim(-1) / shape(ch).dim(-1) : cc;
                    }
                    if (constant(c) != null && n.op() == Op.CONST) {
                        total = 0;
                    }
                    if (total < cost.getOrDefault(c, Double.POSITIVE_INFINITY) - 1e-9) {
                        cost.put(c, total);
                        choice.put(c, n);
                        changed = true;
                    }
                }
            }
        }
        return choice;
    }

    private double dagCost(Map<Integer, ENode> choice) {
        Set<Integer> seen = new HashSet<>();
        Set<Integer> onPath = new HashSet<>();
        double[] total = {0};
        for (int r : roots) {
            if (!dag(find(r), choice, seen, onPath, total)) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return total[0];
    }

    private boolean dag(int c, Map<Integer, ENode> choice, Set<Integer> seen, Set<Integer> onPath, double[] total) {
        if (seen.contains(c)) {
            return !onPath.contains(c);
        }
        ENode n = choice.get(c);
        if (n == null) {
            return false;
        }
        seen.add(c);
        onPath.add(c);
        total[0] += constant(c) != null ? 0 : nodeCost(c, n);
        if (constant(c) == null) {
            for (int ch : n.children()) {
                if (!dag(find(ch), choice, seen, onPath, total)) {
                    return false;
                }
            }
        }
        onPath.remove(c);
        return true;
    }

    private int build(int c, Map<Integer, ENode> choice, TensorExpressionGraph out, Map<Integer, Integer> built) {
        Integer done = built.get(c);
        if (done != null) {
            return done;
        }
        Tensor value = constant(c);
        int id;
        if (value != null) {
            id = out.constant(value);
        } else {
            ENode n = choice.get(c);
            int[] in = new int[n.children().length];
            for (int i = 0; i < in.length; i++) {
                in[i] = build(find(n.children()[i]), choice, out, built);
            }
            id = out.add(new TensorExpressionGraph.Node(n.op(), in, n.payload(), shape(c)));
        }
        built.put(c, id);
        return id;
    }

    @Override
    public String toString() {
        return "EGraph(" + classes().size() + " classes, " + size() + " nodes)";
    }
}
