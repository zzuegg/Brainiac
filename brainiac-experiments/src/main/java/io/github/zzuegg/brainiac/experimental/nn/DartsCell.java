package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A DARTS search cell for feature vectors: a small DAG whose every edge is a <em>mixed
 * operation</em>, the softmax-weighted sum of all candidate operations
 * ({@code sum_k softmax(alpha)_k op_k(x)}). The architecture weights {@code alpha} are
 * continuous, so the architecture is searched by gradient descent alongside the weights;
 * afterwards each edge keeps its strongest non-zero operation ({@link #derive()}).
 *
 * <p>Node {@code i} (from 1) sums the mixed operations applied to the input (node 0) and to
 * every earlier node; the cell outputs the last node. {@link #parameters()} returns only the
 * operations' weights, so a plain {@code Trainer} trains the weights;
 * {@link #architecture()} returns the {@code alpha}s for the outer, architecture step (see
 * {@code DartsSearch}).
 *
 * <p>Liu, Simonyan and Yang 2019, "DARTS: Differentiable Architecture Search",
 * <a href="https://arxiv.org/abs/1806.09055">arXiv:1806.09055</a>.
 */
public final class DartsCell implements Layer {

    /** Candidate operations of an edge; all keep the feature size. */
    public enum Candidate {
        /** outputs zero: lets the search drop an edge */
        ZERO,
        /** skip connection */
        IDENTITY,
        /** {@code x W + b} */
        LINEAR,
        /** {@code relu(x W + b)} */
        RELU,
        /** {@code tanh(x W + b)} */
        TANH,
        /** {@code (x W + b)^2}, a multiplicative interaction */
        SQUARE,
        /** {@code gelu(x W + b)} */
        GELU
    }

    private final int features;
    private final int nodes;
    private final List<Candidate> candidates;
    /** per edge, in the order (1<-0), (2<-0), (2<-1), ... */
    private final List<Var> alphas = new ArrayList<>();
    private final List<List<Var[]>> weights = new ArrayList<>();

    private DartsCell(int features, int nodes, List<Candidate> candidates, SplittableRandom random) {
        this.features = features;
        this.nodes = nodes;
        this.candidates = List.copyOf(candidates);
        float scale = (float) Math.sqrt(1.0 / features);
        for (int i = 1; i <= nodes; i++) {
            for (int j = 0; j < i; j++) {
                alphas.add(Var.param(Tensor.full(Shape.of(candidates.size()), 0)));
                List<Var[]> ops = new ArrayList<>();
                for (Candidate c : candidates) {
                    if (c == Candidate.ZERO || c == Candidate.IDENTITY) {
                        ops.add(null);
                        continue;
                    }
                    float[] w = new float[features * features];
                    for (int k = 0; k < w.length; k++) {
                        w[k] = (float) (random.nextGaussian() * scale);
                    }
                    ops.add(new Var[] {Var.param(Tensor.of(Shape.of(features, features), w)),
                        Var.param(Tensor.zeros(Shape.of(features)))});
                }
                weights.add(ops);
            }
        }
    }

    /**
     * A spec for {@code Model.sequential().add(...)}: the cell keeps the feature size.
     *
     * @param nodes the number of intermediate nodes
     * @param candidates the operations every edge chooses from
     * @return the spec
     */
    public static LayerSpec spec(int nodes, Candidate... candidates) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return of(input.dim(-1), nodes, random, candidates);
            }
        };
    }

    /**
     * Creates a cell.
     *
     * @param features the feature size
     * @param nodes the number of intermediate nodes
     * @param random the random source for the weights
     * @param candidates the operations every edge chooses from (all of them when empty)
     * @return the cell
     */
    public static DartsCell of(int features, int nodes, SplittableRandom random, Candidate... candidates) {
        if (nodes < 1) {
            throw new IllegalArgumentException("need at least one node");
        }
        List<Candidate> list = candidates.length == 0 ? List.of(Candidate.values()) : List.of(candidates);
        return new DartsCell(features, nodes, list, random);
    }

    @Override
    public Var forward(Var input) {
        List<Var> states = new ArrayList<>();
        states.add(input);
        int edge = 0;
        for (int i = 1; i <= nodes; i++) {
            Var sum = null;
            for (int j = 0; j < i; j++, edge++) {
                Var mixed = mixed(edge, states.get(j));
                if (mixed != null) {
                    sum = sum == null ? mixed : sum.add(mixed);
                }
            }
            states.add(sum == null ? input.mul(0f) : sum);
        }
        return states.getLast();
    }

    private Var mixed(int edge, Var x) {
        Var w = Activation.SOFTMAX.forward(alphas.get(edge).reshape(1, candidates.size()));
        Var out = null;
        for (int k = 0; k < candidates.size(); k++) {
            Var term = op(edge, k, x);
            if (term == null) {
                continue;
            }
            Var weighted = term.mul(w.slice(1, k, k + 1));
            out = out == null ? weighted : out.add(weighted);
        }
        return out;
    }

    private Var op(int edge, int k, Var x) {
        Candidate c = candidates.get(k);
        if (c == Candidate.ZERO) {
            return null;
        }
        if (c == Candidate.IDENTITY) {
            return x;
        }
        Var[] p = weights.get(edge).get(k);
        Var lin = x.matmul(p[0]).add(p[1]);
        return switch (c) {
            case LINEAR -> lin;
            case RELU -> lin.relu();
            case TANH -> lin.tanh();
            case SQUARE -> lin.square();
            case GELU -> Activation.GELU.forward(lin);
            default -> throw new IllegalStateException();
        };
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (List<Var[]> ops : weights) {
            for (Var[] p : ops) {
                if (p != null) {
                    out.add(p[0]);
                    out.add(p[1]);
                }
            }
        }
        return out;
    }

    /** @return the architecture weights, one vector of candidate logits per edge */
    public List<Var> architecture() {
        return List.copyOf(alphas);
    }

    /**
     * The softmax weights of an edge's candidates.
     *
     * @param edge the edge, in the order (1&lt;-0), (2&lt;-0), (2&lt;-1), ...
     * @return one weight per candidate
     */
    public float[] mixture(int edge) {
        float[] a = alphas.get(edge).value().to(DType.FLOAT32).toArray();
        float max = Float.NEGATIVE_INFINITY;
        for (float v : a) {
            max = Math.max(max, v);
        }
        float sum = 0;
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (float) Math.exp(a[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < a.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    /** @return the number of edges */
    public int edges() {
        return alphas.size();
    }

    /**
     * The discrete architecture: each edge's strongest operation other than {@code ZERO}.
     *
     * @return one operation per edge
     */
    public List<Candidate> derive() {
        List<Candidate> out = new ArrayList<>();
        for (int e = 0; e < alphas.size(); e++) {
            float[] m = mixture(e);
            int best = -1;
            for (int k = 0; k < m.length; k++) {
                if (candidates.get(k) != Candidate.ZERO && (best < 0 || m[k] > m[best])) {
                    best = k;
                }
            }
            out.add(candidates.get(best));
        }
        return out;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "DartsCell(" + features + " features, " + nodes + " nodes, " + candidates + ")";
    }
}
