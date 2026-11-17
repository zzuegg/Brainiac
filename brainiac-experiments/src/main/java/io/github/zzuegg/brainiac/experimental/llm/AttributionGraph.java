package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.CrossLayerTranscoder;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * An attribution graph: the linear computational graph of one prompt through a
 * <em>local replacement model</em>, in which the MLPs are replaced by a
 * {@link CrossLayerTranscoder} (plus one error node per layer, so the graph reproduces the
 * real model exactly) and every nonlinearity is frozen at its value on this prompt. Nodes
 * are input embeddings, active transcoder features, error nodes and the target logit; an
 * edge's weight is the direct, linear contribution of the source to the target node's
 * pre-activation. The graph is then pruned to the nodes that carry most of the
 * <em>indirect</em> influence on the target, which leaves a small readable circuit.
 *
 * <p>Influence follows the method paper: edge weights are made absolute and normalised per
 * target node (incoming weights sum to one), and the indirect influence matrix is
 * {@code B = A + A^2 + ... = (I - A)^-1 - I}; nodes are kept, strongest first, until they
 * account for the requested share of the target's total influence.
 *
 * <p>Ameisen et al. 2025, "Circuit Tracing: Revealing Computational Graphs in Language
 * Models", <a href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html">Transformer
 * Circuits</a>. {@link #ofResidualMlp} builds the graph for a residual stack of MLPs with a
 * linear readout; for a transformer, attention patterns would be frozen the same way and
 * add edges between positions.
 */
public final class AttributionGraph {

    /** What a node stands for. */
    public enum Kind {
        /** an input embedding */
        INPUT,
        /** an active transcoder feature */
        FEATURE,
        /** the part of an MLP the transcoder misses */
        ERROR,
        /** the output being explained */
        TARGET
    }

    /**
     * A node.
     *
     * @param id its index
     * @param kind what it is
     * @param layer its layer (-1 for inputs, the layer count for the target)
     * @param index the feature or input index (-1 when not applicable)
     * @param activation its value on the prompt
     * @param label a readable name
     */
    public record Node(int id, Kind kind, int layer, int index, double activation, String label) {
    }

    private final List<Node> nodes;
    private final double[][] weights;
    private final int target;

    private AttributionGraph(List<Node> nodes, double[][] weights, int target) {
        this.nodes = List.copyOf(nodes);
        this.weights = weights;
        this.target = target;
    }

    /**
     * A graph from explicit nodes and edges.
     *
     * @param nodes the nodes, with ids {@code 0..n-1} in order
     * @param weights {@code weights[from][to]}, the direct contribution
     * @param target the id of the target node
     * @return the graph
     */
    public static AttributionGraph of(List<Node> nodes, double[][] weights, int target) {
        double[][] w = new double[nodes.size()][];
        for (int i = 0; i < w.length; i++) {
            w[i] = weights[i].clone();
        }
        return new AttributionGraph(nodes, w, target);
    }

    /**
     * The attribution graph of one sample through a residual stack of MLPs,
     * {@code h_0 = sum inputs}, {@code h_{l+1} = h_l + MLP_l(h_l)}, {@code logit = h_L . u},
     * with the MLPs replaced by a cross-layer transcoder.
     *
     * @param clt the transcoder trained on this stack's MLPs
     * @param inputContributions what each input node adds to the residual stream
     * @param inputLabels names of the input nodes
     * @param mlpInputs the real residual stream entering each MLP, {@code [layer][dim]}
     * @param mlpOutputs the real output of each MLP, {@code [layer][dim]}
     * @param readout the target logit's readout direction
     * @param targetLabel the target's name
     * @return the graph
     */
    public static AttributionGraph ofResidualMlp(CrossLayerTranscoder clt, float[][] inputContributions,
            String[] inputLabels, float[][] mlpInputs, float[][] mlpOutputs, float[] readout, String targetLabel) {
        int layers = clt.layers();
        int dim = readout.length;
        List<Node> nodes = new ArrayList<>();
        List<float[]> writes = new ArrayList<>();      // for each node: what it adds to each MLP output, flattened
        for (int i = 0; i < inputContributions.length; i++) {
            nodes.add(new Node(nodes.size(), Kind.INPUT, -1, i, norm(inputContributions[i]), inputLabels[i]));
            writes.add(null);
        }
        float[][][] codes = new float[layers][][];
        List<Var> codeVars = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            Var z = clt.encode(l, Var.of(Tensor.of(Shape.of(1, dim), mlpInputs[l])));
            codeVars.add(z);
            codes[l] = new float[][] {z.value().to(DType.FLOAT32).toArray()};
        }
        int[] firstFeature = new int[layers];
        for (int l = 0; l < layers; l++) {
            firstFeature[l] = nodes.size();
            for (int f = 0; f < clt.features(); f++) {
                if (codes[l][0][f] > 0) {
                    nodes.add(new Node(nodes.size(), Kind.FEATURE, l, f, codes[l][0][f], "L" + l + "/F" + f));
                }
            }
        }
        int[] errorNode = new int[layers];
        float[][] errors = new float[layers][];
        for (int l = 0; l < layers; l++) {
            // the error node also carries the decoder bias, so the graph adds up exactly
            float[] recon = clt.decode(l, codeVars).value().to(DType.FLOAT32).toArray();
            float[] bias = clt.outputBias(l);
            errors[l] = new float[dim];
            for (int d = 0; d < dim; d++) {
                errors[l][d] = mlpOutputs[l][d] - (recon[d] - bias[d]);
            }
            errorNode[l] = nodes.size();
            nodes.add(new Node(nodes.size(), Kind.ERROR, l, -1, norm(errors[l]), "err" + l));
        }
        int t = nodes.size();
        double logit = 0;
        for (int d = 0; d < dim; d++) {
            double h = mlpInputs[layers - 1][d] + mlpOutputs[layers - 1][d];
            logit += h * readout[d];
        }
        nodes.add(new Node(t, Kind.TARGET, layers, -1, logit, targetLabel));
        double[][] w = new double[nodes.size()][nodes.size()];
        // the residual vector each source adds, read by each later receiver
        for (Node src : nodes) {
            if (src.kind() == Kind.TARGET) {
                continue;
            }
            for (Node dst : nodes) {
                if (dst.kind() != Kind.FEATURE && dst.kind() != Kind.TARGET) {
                    continue;
                }
                float[] reader = dst.kind() == Kind.TARGET ? readout : clt.encoderDirection(dst.layer(), dst.index());
                int before = dst.layer();         // the residual entering layer `before`
                double v = 0;
                switch (src.kind()) {
                    case INPUT -> v = dot(inputContributions[src.index()], reader);
                    case ERROR -> v = src.layer() < before ? dot(errors[src.layer()], reader) : 0;
                    case FEATURE -> {
                        for (int m = src.layer(); m < before && m < layers; m++) {
                            v += src.activation() * dot(clt.decoderDirection(src.layer(), m, src.index()), reader);
                        }
                    }
                    default -> {
                    }
                }
                w[src.id()][dst.id()] = v;
            }
        }
        return new AttributionGraph(nodes, w, t);
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double norm(float[] a) {
        return Math.sqrt(dot(a, a));
    }

    /** @return the nodes */
    public List<Node> nodes() {
        return nodes;
    }

    /** @return the target node */
    public Node target() {
        return nodes.get(target);
    }

    /**
     * @param from source id
     * @param to receiver id
     * @return the direct contribution
     */
    public double weight(int from, int to) {
        return weights[from][to];
    }

    /**
     * The indirect influence of every node on the target: the target's column of
     * {@code A + A^2 + ...}, with {@code A} the absolute edge weights normalised per
     * receiver.
     *
     * @return one value per node
     */
    public double[] influence() {
        int n = nodes.size();
        double[][] a = new double[n][n];
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += Math.abs(weights[i][j]);
            }
            for (int i = 0; i < n; i++) {
                a[i][j] = s == 0 ? 0 : Math.abs(weights[i][j]) / s;
            }
        }
        // influence on target: x = A e_t + A (A e_t) + ...  (the graph is acyclic)
        double[] direct = new double[n];
        for (int i = 0; i < n; i++) {
            direct[i] = a[i][target];
        }
        double[] total = direct.clone();
        double[] power = direct;
        for (int step = 0; step < n; step++) {
            double[] next = new double[n];
            double mass = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    next[i] += a[i][j] * power[j];
                }
                mass += next[i];
            }
            if (mass < 1e-12) {
                break;
            }
            for (int i = 0; i < n; i++) {
                total[i] += next[i];
            }
            power = next;
        }
        return total;
    }

    /**
     * Keeps the target and the most influential nodes until they hold {@code share} of the
     * total influence; edges between kept nodes stay.
     *
     * @param share for example 0.8
     * @return the pruned graph (node ids renumbered, original ids kept in the labels' order)
     */
    public AttributionGraph prune(double share) {
        double[] inf = influence();
        double total = Arrays.stream(inf).sum();
        Integer[] order = new Integer[nodes.size()];
        Arrays.setAll(order, i -> i);
        Arrays.sort(order, Comparator.comparingDouble(i -> -inf[i]));
        boolean[] keep = new boolean[nodes.size()];
        keep[target] = true;
        double acc = 0;
        for (int i : order) {
            if (i == target) {
                continue;
            }
            if (acc >= share * total) {
                break;
            }
            keep[i] = true;
            acc += inf[i];
        }
        List<Node> kept = new ArrayList<>();
        int[] map = new int[nodes.size()];
        for (Node node : nodes) {
            if (keep[node.id()]) {
                map[node.id()] = kept.size();
                kept.add(new Node(kept.size(), node.kind(), node.layer(), node.index(), node.activation(), node.label()));
            }
        }
        double[][] w = new double[kept.size()][kept.size()];
        for (Node a : nodes) {
            for (Node b : nodes) {
                if (keep[a.id()] && keep[b.id()]) {
                    w[map[a.id()]][map[b.id()]] = weights[a.id()][b.id()];
                }
            }
        }
        return new AttributionGraph(kept, w, map[target]);
    }

    /**
     * The kept node of a kind with the most influence on the target.
     *
     * @param kind the kind
     * @param layer the layer, or -2 for any
     * @return the node, or null if there is none
     */
    public Node strongest(Kind kind, int layer) {
        double[] inf = influence();
        Node best = null;
        for (Node node : nodes) {
            if (node.kind() == kind && (layer == -2 || node.layer() == layer)
                    && (best == null || inf[node.id()] > inf[best.id()])) {
                best = node;
            }
        }
        return best;
    }

    /**
     * The target rebuilt from its direct incoming edges. In an unpruned graph from
     * {@link #ofResidualMlp} it equals the target's activation (error nodes make the graph
     * exact), which is a completeness check.
     *
     * @return the sum of incoming weights to the target
     */
    public double targetFromEdges() {
        double s = 0;
        for (Node n : nodes) {
            s += weights[n.id()][target];
        }
        return s;
    }

    @Override
    public String toString() {
        long features = nodes.stream().filter(n -> n.kind() == Kind.FEATURE).count();
        return "AttributionGraph(" + nodes.size() + " nodes, " + features + " features, target "
                + Objects.toString(target().label()) + ")";
    }
}
