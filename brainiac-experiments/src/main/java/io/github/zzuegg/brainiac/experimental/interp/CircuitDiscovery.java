package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Automated circuit discovery over the <em>edges</em> of a network's computational graph:
 * which connections between components (heads, MLPs, the embedding and the output, each
 * reading the sum of upstream outputs from the residual stream) implement a behaviour.
 *
 * <p>Every method works on an {@link EdgeModel}, whose forward pass takes one gate per edge:
 * edge {@code (u, v)} delivers {@code out_u + g (out_u' - out_u)} to {@code v}, where
 * {@code out_u'} is {@code u}'s output on a corrupted input. Gate 0 is the clean model,
 * gate 1 patches the edge.
 *
 * <ul>
 *   <li>{@link #acdc} (Conmy et al., NeurIPS 2023) visits edges from the output backwards,
 *       patches each, and prunes it for good if the KL divergence from the clean output rises
 *       by less than a threshold; one forward pass per edge.</li>
 *   <li>{@link #eap} (edge attribution patching, Syed et al., 2023) scores all edges at once
 *       with the first-order estimate {@code (corrupt - clean) . grad}, the gradient of the
 *       metric with respect to the gates at 0: two forward passes and one backward.</li>
 *   <li>{@link #eapIg} (Hanna et al., COLM 2024) averages that gradient along the path from
 *       clean (gate 0) to corrupt (gate 1), an integrated gradient, which also sees edges
 *       whose gradient saturates at the clean input.</li>
 *   <li>{@link #nodePatching} and {@link #nodeAttribution} are the node-level baselines
 *       (activation patching and AtP of whole components); an edge then scores as the
 *       lower of its two end nodes.</li>
 * </ul>
 *
 * <p>{@link #faithfulness} and {@link #completeness} score a found circuit, and
 * {@link #rocAuc} scores edge rankings against a known circuit. {@link PlantedCircuit} is a
 * residual network whose circuit is known by construction.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2304.14997">arXiv:2304.14997</a>,
 * <a href="https://arxiv.org/abs/2310.10348">arXiv:2310.10348</a>,
 * <a href="https://arxiv.org/abs/2403.17806">arXiv:2403.17806</a>.
 */
public final class CircuitDiscovery {

    private CircuitDiscovery() {
    }

    /** A network whose edges can be patched one by one. */
    public interface EdgeModel {

        /** @return the edges as {@code {from, to}} node pairs, in topological order of {@code to} */
        int[][] edges();

        /** @return the number of nodes; the last is the output */
        int nodes();

        /**
         * Runs the network with gated edges.
         *
         * @param clean the clean inputs {@code [batch, ...]}
         * @param corrupt the corrupted inputs, same shape
         * @param gates one gate per edge, {@code [edges]}
         * @return the logits {@code [batch, classes]}
         */
        Var forward(Tensor clean, Tensor corrupt, Var gates);
    }

    /**
     * The mean logit difference between each row's clean answer and the mean of the other
     * classes, the metric whose gradient EAP uses.
     *
     * @param logits {@code [batch, classes]}
     * @param answers the clean model's answers
     * @return the scalar metric
     */
    public static Var logitDifference(Var logits, int[] answers) {
        int b = answers.length;
        int c = logits.shape().dim(1);
        float[] w = new float[b * c];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < c; k++) {
                w[i * c + k] = (k == answers[i] ? 1f : -1f / (c - 1)) / b;
            }
        }
        return logits.mul(Tensor.of(Shape.of(b, c), w)).sum();
    }

    private static double kl(float[] p, float[] q, int classes) {
        int b = p.length / classes;
        double total = 0;
        for (int i = 0; i < b; i++) {
            double[] lp = logSoftmax(p, i * classes, classes);
            double[] lq = logSoftmax(q, i * classes, classes);
            for (int k = 0; k < classes; k++) {
                total += Math.exp(lp[k]) * (lp[k] - lq[k]);
            }
        }
        return total / b;
    }

    private static double[] logSoftmax(float[] v, int off, int n) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, v[off + i]);
        }
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += Math.exp(v[off + i] - max);
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = v[off + i] - max - Math.log(s);
        }
        return out;
    }

    private static float[] run(EdgeModel m, Tensor clean, Tensor corrupt, float[] gates) {
        return Var.noGrad(() -> m.forward(clean, corrupt, Var.of(Tensor.of(Shape.of(gates.length), gates))))
                .value().toArray();
    }

    /**
     * The clean model's answers.
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @return the argmax class per row
     */
    public static int[] answers(EdgeModel m, Tensor clean, Tensor corrupt) {
        float[] l = run(m, clean, corrupt, new float[m.edges().length]);
        int b = clean.shape().dim(0);
        int c = l.length / b;
        int[] a = new int[b];
        for (int i = 0; i < b; i++) {
            for (int k = 1; k < c; k++) {
                a[i] = l[i * c + k] > l[i * c + a[i]] ? k : a[i];
            }
        }
        return a;
    }

    /**
     * The result of ACDC.
     *
     * @param scores per edge the KL increase measured when it was tried
     * @param kept per edge whether it stayed in the circuit
     * @param forwardPasses the forward passes used
     */
    public record AcdcResult(double[] scores, boolean[] kept, int forwardPasses) {
    }

    /**
     * ACDC with the KL divergence from the clean output as the metric.
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @param threshold prune an edge if patching it raises the KL by less than this
     * @return the scores and the circuit
     */
    public static AcdcResult acdc(EdgeModel m, Tensor clean, Tensor corrupt, double threshold) {
        int[][] edges = m.edges();
        int e = edges.length;
        float[] gates = new float[e];
        float[] reference = run(m, clean, corrupt, gates);
        int classes = reference.length / clean.shape().dim(0);
        double current = 0;
        Integer[] order = new Integer[e];
        for (int i = 0; i < e; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.<Integer>comparingInt(i -> -edges[i][1]).thenComparingInt(i -> -edges[i][0]));
        double[] scores = new double[e];
        boolean[] kept = new boolean[e];
        int passes = 1;
        for (int i : order) {
            gates[i] = 1;
            double now = kl(reference, run(m, clean, corrupt, gates), classes);
            passes++;
            scores[i] = now - current;
            if (now - current < threshold) {
                current = now;
            } else {
                gates[i] = 0;
                kept[i] = true;
            }
        }
        return new AcdcResult(scores, kept, passes);
    }

    /**
     * Edge attribution patching: {@code |d metric / d gate|} at the clean model.
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @return per edge its score
     */
    public static double[] eap(EdgeModel m, Tensor clean, Tensor corrupt) {
        return gateGradient(m, clean, corrupt, new float[] {0f});
    }

    /**
     * EAP with integrated gradients: the gate gradient averaged at {@code steps} points
     * between clean and corrupt (midpoint rule).
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @param steps integration points
     * @return per edge its score
     */
    public static double[] eapIg(EdgeModel m, Tensor clean, Tensor corrupt, int steps) {
        float[] alphas = new float[steps];
        for (int i = 0; i < steps; i++) {
            alphas[i] = (i + 0.5f) / steps;
        }
        return gateGradient(m, clean, corrupt, alphas);
    }

    private static double[] gateGradient(EdgeModel m, Tensor clean, Tensor corrupt, float[] alphas) {
        int e = m.edges().length;
        int[] answers = answers(m, clean, corrupt);
        double[] sum = new double[e];
        for (float a : alphas) {
            float[] g = new float[e];
            Arrays.fill(g, a);
            Var gates = Var.param(Tensor.of(Shape.of(e), g));
            logitDifference(m.forward(clean, corrupt, gates), answers).backward();
            float[] grad = gates.grad().toArray();
            for (int i = 0; i < e; i++) {
                sum[i] += grad[i] / alphas.length;
            }
        }
        for (int i = 0; i < e; i++) {
            sum[i] = Math.abs(sum[i]);
        }
        return sum;
    }

    /**
     * Node-level activation patching: patch every outgoing edge of a node at once and
     * measure the KL increase; an edge scores as the lower of its endpoints (the output node
     * counts as infinitely important).
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @return per edge its score
     */
    public static double[] nodePatching(EdgeModel m, Tensor clean, Tensor corrupt) {
        int[][] edges = m.edges();
        float[] reference = run(m, clean, corrupt, new float[edges.length]);
        int classes = reference.length / clean.shape().dim(0);
        double[] node = new double[m.nodes()];
        node[m.nodes() - 1] = Double.POSITIVE_INFINITY;
        for (int u = 0; u < m.nodes() - 1; u++) {
            float[] g = new float[edges.length];
            for (int i = 0; i < edges.length; i++) {
                g[i] = edges[i][0] == u ? 1 : 0;
            }
            node[u] = kl(reference, run(m, clean, corrupt, g), classes);
        }
        return edgeFromNodes(edges, node);
    }

    /**
     * Node-level attribution patching (AtP): a node's score is the absolute sum of its
     * outgoing gate gradients, the first-order effect of patching the whole node.
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @return per edge its score
     */
    public static double[] nodeAttribution(EdgeModel m, Tensor clean, Tensor corrupt) {
        int[][] edges = m.edges();
        int e = edges.length;
        int[] answers = answers(m, clean, corrupt);
        Var gates = Var.param(Tensor.zeros(Shape.of(e)));
        logitDifference(m.forward(clean, corrupt, gates), answers).backward();
        float[] grad = gates.grad().toArray();
        double[] node = new double[m.nodes()];
        for (int i = 0; i < e; i++) {
            node[edges[i][0]] += grad[i];
        }
        for (int u = 0; u < node.length; u++) {
            node[u] = Math.abs(node[u]);
        }
        node[m.nodes() - 1] = Double.POSITIVE_INFINITY;
        return edgeFromNodes(edges, node);
    }

    private static double[] edgeFromNodes(int[][] edges, double[] node) {
        double[] s = new double[edges.length];
        for (int i = 0; i < edges.length; i++) {
            s[i] = Math.min(node[edges[i][0]], node[edges[i][1]]);
        }
        return s;
    }

    /**
     * ROC-AUC of edge scores against the true circuit.
     *
     * @param scores per edge a score, higher meaning "in the circuit"
     * @param truth per edge membership of the true circuit
     * @return the AUC
     */
    public static double rocAuc(double[] scores, boolean[] truth) {
        return InterpMath.rocAuc(scores, truth);
    }

    /**
     * Faithfulness of a circuit: with every other edge patched, the share of the clean
     * logit difference (over the fully patched model's) that the circuit recovers.
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @param circuit per edge membership
     * @return {@code (m(circuit) - m(none)) / (m(full) - m(none))}
     */
    public static double faithfulness(EdgeModel m, Tensor clean, Tensor corrupt, boolean[] circuit) {
        int e = circuit.length;
        float[] only = new float[e];
        for (int i = 0; i < e; i++) {
            only[i] = circuit[i] ? 0 : 1;
        }
        float[] all = new float[e];
        Arrays.fill(all, 1f);
        int[] answers = answers(m, clean, corrupt);
        double full = metric(m, clean, corrupt, new float[e], answers);
        double none = metric(m, clean, corrupt, all, answers);
        double kept = metric(m, clean, corrupt, only, answers);
        return (kept - none) / (full - none);
    }

    /**
     * Completeness of a circuit: with the circuit's edges patched and the rest left clean,
     * the share of the logit difference that remains (low is complete).
     *
     * @param m the model
     * @param clean clean inputs
     * @param corrupt corrupted inputs
     * @param circuit per edge membership
     * @return {@code (m(rest) - m(none)) / (m(full) - m(none))}
     */
    public static double completeness(EdgeModel m, Tensor clean, Tensor corrupt, boolean[] circuit) {
        int e = circuit.length;
        float[] rest = new float[e];
        for (int i = 0; i < e; i++) {
            rest[i] = circuit[i] ? 1 : 0;
        }
        float[] all = new float[e];
        Arrays.fill(all, 1f);
        int[] answers = answers(m, clean, corrupt);
        double full = metric(m, clean, corrupt, new float[e], answers);
        double none = metric(m, clean, corrupt, all, answers);
        return (metric(m, clean, corrupt, rest, answers) - none) / (full - none);
    }

    private static double metric(EdgeModel m, Tensor clean, Tensor corrupt, float[] gates, int[] answers) {
        return Var.noGrad(() -> logitDifference(m.forward(clean, corrupt,
                Var.of(Tensor.of(Shape.of(gates.length), gates))), answers)).value().item();
    }

    /**
     * The top-scoring edges.
     *
     * @param scores per edge a score
     * @param k how many to keep
     * @return per edge whether it is among the {@code k} best
     */
    public static boolean[] top(double[] scores, int k) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        boolean[] keep = new boolean[scores.length];
        for (int i = 0; i < Math.min(k, order.length); i++) {
            keep[order[i]] = true;
        }
        return keep;
    }

    /**
     * A residual network with a planted circuit. Node 0 embeds the input; then
     * {@code layers} layers of {@code width} nodes, each reading the outputs of all earlier
     * nodes through its own matrix and applying {@code tanh}; the last node reads all nodes
     * into the logits. Each node writes into its own subspace of the residual stream, so an
     * edge carries exactly its source node's output. Planted edges have unit-scale weights,
     * all others weights of scale {@code leak}, so the circuit is known by construction. One
     * planted edge has a large gain: its target saturates on clean and corrupted inputs
     * alike, so its gradient vanishes at the clean input although patching it flips the
     * node's output, the case where plain EAP fails and EAP-IG does not.
     */
    public static final class PlantedCircuit implements EdgeModel {

        private final int inputs;
        private final int width;
        private final int dim;
        private final int classes;
        private final int nodes;
        private final int[][] edges;
        private final boolean[] planted;
        private final Tensor embed;
        private final Tensor[] read;
        private final int[] layerOf;

        /**
         * Builds the network.
         *
         * @param inputs input features
         * @param dim each node's output width
         * @param layers hidden layers, at least 3
         * @param width nodes per layer, at least 4
         * @param classes output classes
         * @param leak the scale of non-circuit weights, such as 0.03
         * @param seed the seed
         */
        public PlantedCircuit(int inputs, int dim, int layers, int width, int classes, double leak, long seed) {
            if (layers < 3 || width < 4) {
                throw new IllegalArgumentException("need at least 3 layers of 4 nodes");
            }
            this.inputs = inputs;
            this.width = width;
            this.dim = dim;
            this.classes = classes;
            this.nodes = 2 + layers * width;
            this.layerOf = new int[nodes];
            for (int v = 1; v < nodes - 1; v++) {
                layerOf[v] = 1 + (v - 1) / width;
            }
            layerOf[nodes - 1] = layers + 1;
            SplittableRandom r = new SplittableRandom(seed);
            this.embed = Tensor.normal(Shape.of(inputs, dim), 0, (float) (1 / Math.sqrt(inputs)), r);
            List<int[]> list = new ArrayList<>();
            for (int v = 1; v < nodes; v++) {
                for (int u = 0; u < v; u++) {
                    if (layerOf[u] < layerOf[v]) {
                        list.add(new int[] {u, v});
                    }
                }
            }
            this.edges = list.toArray(new int[0][]);
            this.planted = new boolean[edges.length];
            // the circuit: emb -> a1 -> b2 -> c3 -> out, emb -> b2, a1 -> c3, emb -> d1 -> c3,
            // and the saturating edge emb -> s1 -> out
            int a1 = node(1, 0);
            int d1 = node(1, 1);
            int s1 = node(1, width - 1);
            int b2 = node(2, 0);
            int c3 = node(3, 0);
            int out = nodes - 1;
            int[][] circuit = {{0, a1}, {a1, b2}, {b2, c3}, {c3, out}, {0, b2}, {a1, c3}, {0, d1}, {d1, c3}, {0, s1},
                {s1, out}};
            int saturating = -1;
            for (int i = 0; i < edges.length; i++) {
                for (int[] c : circuit) {
                    if (edges[i][0] == c[0] && edges[i][1] == c[1]) {
                        planted[i] = true;
                        if (c[0] == 0 && c[1] == s1) {
                            saturating = i;
                        }
                    }
                }
            }
            this.read = new Tensor[edges.length];
            for (int i = 0; i < edges.length; i++) {
                int outW = edges[i][1] == out ? classes : dim;
                float scale = planted[i] ? (float) (1.5 / Math.sqrt(dim)) : (float) (leak / Math.sqrt(dim));
                if (i == saturating) {
                    scale *= 12;
                }
                read[i] = Tensor.normal(Shape.of(dim, outW), 0, scale, r);
            }
        }

        private int node(int layer, int index) {
            return 1 + (layer - 1) * width + index;
        }

        @Override
        public int[][] edges() {
            return edges.clone();
        }

        @Override
        public int nodes() {
            return nodes;
        }

        /** @return per edge whether it belongs to the planted circuit */
        public boolean[] circuit() {
            return planted.clone();
        }

        /** @return the number of input features */
        public int inputs() {
            return inputs;
        }

        private Var[] outputs(Var x, Var[] corrupt, Var gates) {
            Var[] out = new Var[nodes];
            out[0] = x.matmul(embed);
            int e = 0;
            for (int v = 1; v < nodes; v++) {
                Var z = null;
                for (; e < edges.length && edges[e][1] == v; e++) {
                    Var src = out[edges[e][0]];
                    if (corrupt != null) {
                        Var g = gates.slice(0, e, e + 1);
                        src = src.add(corrupt[edges[e][0]].sub(src).mul(g));
                    }
                    Var term = src.matmul(read[e]);
                    z = z == null ? term : z.add(term);
                }
                out[v] = v == nodes - 1 ? z : Activation.TANH.forward(z);
            }
            return out;
        }

        @Override
        public Var forward(Tensor clean, Tensor corrupt, Var gates) {
            Var[] c = Var.noGrad(() -> outputs(Var.of(corrupt), null, null));
            Var[] detached = new Var[c.length];
            for (int i = 0; i < c.length; i++) {
                detached[i] = Var.of(c[i].value());
            }
            return outputs(Var.of(clean), detached, gates)[nodes - 1];
        }

        /**
         * Random clean inputs in {@code {-1, +1}} and their corruption (every sign flipped).
         *
         * @param n rows
         * @param seed the seed
         * @return {@code {clean, corrupt}}
         */
        public Tensor[] inputs(int n, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            float[] x = new float[n * inputs];
            float[] c = new float[n * inputs];
            for (int i = 0; i < x.length; i++) {
                x[i] = r.nextBoolean() ? 1 : -1;
                c[i] = -x[i];
            }
            return new Tensor[] {Tensor.of(Shape.of(n, inputs), x), Tensor.of(Shape.of(n, inputs), c)};
        }
    }
}
