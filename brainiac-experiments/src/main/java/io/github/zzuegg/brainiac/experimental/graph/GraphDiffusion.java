package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * DiGress-style discrete denoising diffusion for graphs (Vignac et al., 2023,
 * <a href="https://arxiv.org/abs/2209.14734">arXiv:2209.14734</a>), for graphs with a fixed
 * node count and two edge states (edge / no edge).
 *
 * <p><b>Noise.</b> Each edge state is a categorical variable noised by the
 * marginal-preserving transition {@code Q_t = alpha_t I + (1 - alpha_t) 1 m^T}, where
 * {@code m} is the edge-state marginal of the training graphs; so
 * {@code q(e_t | e_0) = abar_t e_0 + (1 - abar_t) m} (cosine schedule). Unlike uniform
 * noise, a noisy graph keeps the data's edge density, so the denoiser always sees
 * realistic sparse graphs.
 *
 * <p><b>Denoiser.</b> A graph transformer ({@link GraphGpsLayer}s: local message passing on
 * the noisy graph plus attention over all nodes) reads node features computed from the
 * noisy graph, the structural features DiGress adds because message passing alone
 * cannot see cycles: degree, random-walk return probabilities (cycle counts up to length 4,
 * {@link GraphPositionalEncodings#randomWalk}) and the noise level. An edge head predicts
 * {@code p(e_0 = 1 | G_t)} from {@code [h_i * h_j || h_i + h_j || e_t]}; training is
 * cross-entropy on the clean adjacency.
 *
 * <p><b>Sampling.</b> Start from {@code m}, then for {@code t = T..1} draw every edge from
 * the posterior {@code sum_{e_0} p(e_0 | G_t) q(e_{t-1} | e_t, e_0)}, symmetrically.
 */
public final class GraphDiffusion {

    private static final int FEATURES = 6;

    private final int nodes;
    private final int width;
    private final int steps;
    private final Var embed;
    private final Var embedBias;
    private final List<GraphGpsLayer> layers = new ArrayList<>();
    private final Var head1;
    private final Var headBias;
    private final Var head2;
    private final Var head2Bias;
    private float density = 0.2f;

    /**
     * Creates the model.
     *
     * @param nodes the node count of every graph
     * @param width the transformer width
     * @param depth the number of GraphGPS layers
     * @param steps the diffusion steps {@code T}
     * @param random the initialization source
     */
    public GraphDiffusion(int nodes, int width, int depth, int steps, SplittableRandom random) {
        this.nodes = nodes;
        this.width = width;
        this.steps = steps;
        this.embed = GraphMath.weight(FEATURES, width, random);
        this.embedBias = GraphMath.bias(width);
        for (int l = 0; l < depth; l++) {
            layers.add(new GraphGpsLayer(width, 2, true, random));
        }
        this.head1 = GraphMath.weight(2 * width + 1, width, random);
        this.headBias = GraphMath.bias(width);
        this.head2 = GraphMath.weight(width, 1, random);
        this.head2Bias = GraphMath.bias(1);
    }

    /**
     * Sets the edge marginal {@code m} from training graphs.
     *
     * @param graphs adjacencies, {@code nodes x nodes} each
     */
    public void fitMarginal(List<float[]> graphs) {
        double e = 0;
        for (float[] a : graphs) {
            for (int i = 0; i < nodes; i++) {
                for (int j = i + 1; j < nodes; j++) {
                    e += a[i * nodes + j];
                }
            }
        }
        density = (float) (e / (graphs.size() * nodes * (nodes - 1) / 2.0));
    }

    /** @return the edge density {@code m[1]} */
    public float density() {
        return density;
    }

    /**
     * @param t a step in {@code 0..T}
     * @return {@code abar_t}, the probability that an edge state is still clean
     */
    public double alphaBar(int t) {
        double s = 0.008;
        double f = Math.cos((t / (double) steps + s) / (1 + s) * Math.PI / 2);
        double f0 = Math.cos(s / (1 + s) * Math.PI / 2);
        return Math.max(0, Math.min(1, f * f / (f0 * f0)));
    }

    private float[] noise(float[] a, int t, RandomGenerator random) {
        double ab = alphaBar(t);
        float[] out = new float[nodes * nodes];
        for (int i = 0; i < nodes; i++) {
            for (int j = i + 1; j < nodes; j++) {
                double p1 = ab * a[i * nodes + j] + (1 - ab) * density;
                float e = random.nextDouble() < p1 ? 1 : 0;
                out[i * nodes + j] = e;
                out[j * nodes + i] = e;
            }
        }
        return out;
    }

    /**
     * Edge logits for a batch of noisy graphs.
     *
     * @param noisy noisy adjacencies
     * @param times their steps
     * @return {@code [graphs * nodes, nodes]}: row {@code g * nodes + i} scores the edges of
     *     node {@code i} in graph {@code g}
     */
    public Var denoise(List<float[]> noisy, int[] times) {
        int g = noisy.size();
        int n = g * nodes;
        float[] adj = new float[n * n];
        float[] mem = new float[g * n];
        for (int k = 0; k < g; k++) {
            for (int i = 0; i < nodes; i++) {
                System.arraycopy(noisy.get(k), i * nodes, adj, (k * nodes + i) * n + k * nodes, nodes);
                mem[k * n + k * nodes + i] = 1;
            }
        }
        Tensor a = Tensor.of(Shape.of(n, n), adj);
        Tensor rw = GraphPositionalEncodings.randomWalk(a, 4);
        float[] rwv = rw.toArray();
        float[] feat = new float[n * FEATURES];
        for (int i = 0; i < n; i++) {
            float d = 0;
            for (int j = 0; j < n; j++) {
                d += adj[i * n + j];
            }
            feat[i * FEATURES] = d / nodes;
            for (int s = 0; s < 4; s++) {
                feat[i * FEATURES + 1 + s] = rwv[i * 4 + s];
            }
            feat[i * FEATURES + 5] = times[i / nodes] / (float) steps;
        }
        Var mask = Var.of(GraphGpsLayer.sameGraphMask(Tensor.of(Shape.of(g, n), mem)));
        Var h = GraphMath.linear(Var.of(Tensor.of(Shape.of(n, FEATURES), feat)), embed, embedBias);
        Var av = Var.of(a);
        for (GraphGpsLayer layer : layers) {
            h = layer.forward(List.of(h, av, mask), true);
        }
        Var hg = h.reshape(g, nodes, width);
        Var hi = hg.reshape(g, nodes, 1, width);
        Var hj = hg.reshape(g, 1, nodes, width);
        float[] local = new float[g * nodes * nodes];
        for (int k = 0; k < g; k++) {
            System.arraycopy(noisy.get(k), 0, local, k * nodes * nodes, nodes * nodes);
        }
        Var e = Var.of(Tensor.of(Shape.of(g, nodes, nodes, 1), local));
        Var pair = Var.concat(3, hi.mul(hj), hi.add(hj).expand(Shape.of(g, nodes, nodes, width)), e);
        Var logits = GraphMath.linear(pair.reshape(g * nodes * nodes, 2 * width + 1), head1, headBias).relu()
                .matmul(head2).add(head2Bias).reshape(g, nodes, nodes);
        return logits.add(logits.permute(0, 2, 1)).mul(0.5f).reshape(g * nodes, nodes);
    }

    /**
     * The denoising loss of a batch of clean graphs at random noise levels.
     *
     * @param clean adjacencies, {@code nodes x nodes} each
     * @param random the noise source
     * @return the cross-entropy over node pairs
     */
    public Var loss(List<float[]> clean, RandomGenerator random) {
        List<float[]> noisy = new ArrayList<>();
        int[] times = new int[clean.size()];
        float[] target = new float[clean.size() * nodes * nodes];
        float[] off = new float[target.length];
        for (int k = 0; k < clean.size(); k++) {
            times[k] = 1 + random.nextInt(steps);
            noisy.add(noise(clean.get(k), times[k], random));
            System.arraycopy(clean.get(k), 0, target, k * nodes * nodes, nodes * nodes);
            for (int i = 0; i < nodes; i++) {
                for (int j = 0; j < nodes; j++) {
                    off[k * nodes * nodes + i * nodes + j] = i == j ? 0 : 1;
                }
            }
        }
        Var logits = denoise(noisy, times);
        Shape s = Shape.of(clean.size() * nodes, nodes);
        Tensor y = Tensor.of(s, target);
        return logits.softplus().sub(logits.mul(y)).mul(Tensor.of(s, off)).sum()
                .div((float) clean.size() * nodes * (nodes - 1));
    }

    /**
     * Samples graphs by running the reverse process.
     *
     * @param count how many
     * @param random the sampling source
     * @return adjacencies, {@code nodes x nodes} each
     */
    public List<float[]> sample(int count, RandomGenerator random) {
        List<float[]> graphs = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            float[] a = new float[nodes * nodes];
            for (int i = 0; i < nodes; i++) {
                for (int j = i + 1; j < nodes; j++) {
                    float e = random.nextDouble() < density ? 1 : 0;
                    a[i * nodes + j] = e;
                    a[j * nodes + i] = e;
                }
            }
            graphs.add(a);
        }
        double[] m = {1 - density, density};
        for (int t = steps; t >= 1; t--) {
            int[] times = new int[count];
            java.util.Arrays.fill(times, t);
            float[] logits = Var.noGrad(() -> denoise(graphs, times)).value().toArray();
            double alpha = alphaBar(t) / Math.max(alphaBar(t - 1), 1e-9);
            double prev = alphaBar(t - 1);
            for (int k = 0; k < count; k++) {
                float[] a = graphs.get(k);
                float[] next = new float[nodes * nodes];
                for (int i = 0; i < nodes; i++) {
                    for (int j = i + 1; j < nodes; j++) {
                        double p0 = 1 / (1 + Math.exp(-logits[(k * nodes + i) * nodes + j]));
                        int et = a[i * nodes + j] > 0 ? 1 : 0;
                        double p1 = 0;
                        for (int e0 = 0; e0 < 2; e0++) {
                            double w = e0 == 1 ? p0 : 1 - p0;
                            double[] post = new double[2];
                            for (int kk = 0; kk < 2; kk++) {
                                double step = alpha * (et == kk ? 1 : 0) + (1 - alpha) * m[et];
                                double marg = prev * (kk == e0 ? 1 : 0) + (1 - prev) * m[kk];
                                post[kk] = step * marg;
                            }
                            p1 += w * post[1] / Math.max(post[0] + post[1], 1e-12);
                        }
                        float e = random.nextDouble() < p1 ? 1 : 0;
                        next[i * nodes + j] = e;
                        next[j * nodes + i] = e;
                    }
                }
                graphs.set(k, next);
            }
        }
        return graphs;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias));
        for (GraphGpsLayer l : layers) {
            p.addAll(l.parameters());
        }
        p.addAll(List.of(head1, headBias, head2, head2Bias));
        return p;
    }

    @Override
    public String toString() {
        return "GraphDiffusion(n=" + nodes + ", T=" + steps + ")";
    }
}
