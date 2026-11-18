package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.MambaLayer;
import io.github.zzuegg.brainiac.experimental.nn.S4dLayer;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A NeuralWalker layer (Chen et al., 2024, <a href="https://arxiv.org/abs/2406.03386">arXiv:2406.03386</a>):
 * random walks as sequences. Every forward samples one walk of length {@code L} from each
 * node, embeds each walk position by its node's features plus a walk positional encoding
 * (for {@code k = 1..K}: is the node the same as {@code k} steps ago, and is it adjacent to
 * it; together these expose cycles and local structure), and runs a sequence model over
 * the walk in both directions: a depthwise convolution, an {@link S4dLayer} or a
 * {@link MambaLayer} ({@link Mixer}). The walk outputs are scattered back to the nodes they
 * visited (averaged), added to the node states, and a local message-passing update
 * ({@link GinLayer}) follows:
 * <pre>
 * h' = LN(h + scatter_mean(SeqModel(walk embeddings)))
 * out = LN(h' + GIN(h', A))
 * </pre>
 * A walk of length {@code L} carries information {@code L} hops in one layer at a cost
 * linear in {@code L}, without squeezing it through the bottleneck edges that cause
 * over-squashing in message passing.
 *
 * <p>Inputs: node features {@code [n, width]} and the adjacency {@code [n, n]}.
 */
public final class RandomWalkSequenceLayer implements MultiInputLayer {

    /** The sequence model run along walks. */
    public enum Mixer {
        /** a width-3 depthwise convolution (non-causal) */
        CONVOLUTION,
        /** a diagonal state-space layer, both directions */
        S4D,
        /** a selective state-space layer, both directions */
        MAMBA
    }

    private final int width;
    private final int length;
    private final int window;
    private final Mixer mixer;
    private final RandomGenerator walker;
    private final Var posWeight;
    private final Var conv;
    private final Layer forwardModel;
    private final Layer backwardModel;
    private final GinLayer local;

    /**
     * Creates the layer.
     *
     * @param width node features in and out
     * @param length the walk length {@code L}
     * @param window the positional-encoding window {@code K}
     * @param mixer the sequence model
     * @param random the initialization source (and the walk sampler's seed)
     */
    public RandomWalkSequenceLayer(int width, int length, int window, Mixer mixer, SplittableRandom random) {
        this.width = width;
        this.length = length;
        this.window = window;
        this.mixer = Objects.requireNonNull(mixer);
        this.posWeight = GraphMath.weight(2 * window, width, random);
        this.conv = Var.param(Tensor.normal(Shape.of(3, width), 0, 0.3f, random));
        this.forwardModel = switch (mixer) {
            case CONVOLUTION -> null;
            case S4D -> new S4dLayer(width, 4, random);
            case MAMBA -> new MambaLayer(width, width, 4, random);
        };
        this.backwardModel = switch (mixer) {
            case CONVOLUTION -> null;
            case S4D -> new S4dLayer(width, 4, random);
            case MAMBA -> new MambaLayer(width, width, 4, random);
        };
        this.local = new GinLayer(width, width, width, random);
        this.walker = random.split();
    }

    /**
     * Samples one uniform random walk from every node.
     *
     * @param adjacency {@code [n, n]}
     * @param length nodes per walk
     * @param random the walk source
     * @return {@code [n][length]} (a walk at an isolated node stays put)
     */
    public static int[][] sampleWalks(Tensor adjacency, int length, RandomGenerator random) {
        int n = adjacency.shape().dim(0);
        float[] a = GraphMath.binary(adjacency);
        int[][] nbr = new int[n][];
        for (int i = 0; i < n; i++) {
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (a[i * n + j] != 0) {
                    l.add(j);
                }
            }
            nbr[i] = l.stream().mapToInt(Integer::intValue).toArray();
        }
        int[][] walks = new int[n][length];
        for (int s = 0; s < n; s++) {
            walks[s][0] = s;
            for (int t = 1; t < length; t++) {
                int[] c = nbr[walks[s][t - 1]];
                walks[s][t] = c.length == 0 ? walks[s][t - 1] : c[random.nextInt(c.length)];
            }
        }
        return walks;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var h = inputs.get(0);
        Tensor adjacency = inputs.get(1).value();
        int n = h.shape().dim(0);
        float[] a = adjacency.toArray();
        int[][] walks = sampleWalks(adjacency, length, walker);
        float[] index = new float[n * length];
        float[] pos = new float[n * length * 2 * window];
        float[] visits = new float[n];
        for (int w = 0; w < n; w++) {
            for (int t = 0; t < length; t++) {
                int v = walks[w][t];
                index[w * length + t] = v;
                visits[v]++;
                for (int k = 1; k <= window && t - k >= 0; k++) {
                    int u = walks[w][t - k];
                    int base = (w * length + t) * 2 * window;
                    pos[base + k - 1] = u == v ? 1 : 0;
                    pos[base + window + k - 1] = a[v * n + u] != 0 ? 1 : 0;
                }
            }
        }
        Tensor idx = Tensor.of(Shape.of(n * length), index);
        Var tokens = h.gather(0, idx).add(Var.of(Tensor.of(Shape.of(n * length, 2 * window), pos)).matmul(posWeight))
                .reshape(n, length, width);
        Var mixed = switch (mixer) {
            case CONVOLUTION -> convolve(tokens);
            case S4D, MAMBA -> forwardModel.forward(tokens, training)
                    .add(backwardModel.forward(tokens.flip(1), training).flip(1));
        };
        for (int i = 0; i < n; i++) {
            visits[i] = 1 / Math.max(1, visits[i]);
        }
        Var back = mixed.reshape(n * length, width).scatterAdd(0, idx, n).mul(Tensor.of(Shape.of(n, 1), visits));
        Var h1 = GraphMath.layerNorm(h.add(back));
        return GraphMath.layerNorm(h1.add(local.forward(List.of(h1, inputs.get(1)), training)));
    }

    private Var convolve(Var tokens) {
        int b = tokens.shape().dim(0);
        Var zero = Var.of(Tensor.zeros(Shape.of(b, 1, width)));
        Var padded = Var.concat(1, zero, tokens, zero);
        Var out = null;
        for (int k = 0; k < 3; k++) {
            Var term = padded.slice(1, k, k + length).mul(conv.slice(0, k, k + 1).reshape(1, 1, width));
            out = out == null ? term : out.add(term);
        }
        return out.relu();
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(posWeight));
        if (mixer == Mixer.CONVOLUTION) {
            p.add(conv);
        } else {
            p.addAll(forwardModel.parameters());
            p.addAll(backwardModel.parameters());
        }
        p.addAll(local.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    @Override
    public String toString() {
        return "RandomWalkSequenceLayer(L=" + length + ", " + mixer + ")";
    }
}
