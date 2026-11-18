package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * One step of neural sheaf diffusion (Bodnar et al., 2022, <a href="https://arxiv.org/abs/2202.04579">arXiv:2202.04579</a>).
 * A cellular sheaf gives every node a {@code d}-dimensional stalk and every edge
 * {@code e = (v, u)} two restriction maps {@code F_{v<e}}, {@code F_{u<e}} ({@code d x d}).
 * The sheaf Laplacian compares neighbours <em>after</em> transporting them through these
 * maps, {@code (L_F x)_v = sum_u F_{v<e}^T (F_{v<e} x_v - F_{u<e} x_u)}, and a layer is one
 * explicit diffusion step
 * <pre>
 * X &lt;- X - elu( L_F (I (x) W1) X W2 ),   node features X: [n, d * channels]
 * </pre>
 * with symmetric degree normalisation. Ordinary graph diffusion ({@code F = I}) pulls
 * neighbours together, which merges classes on heterophilous graphs; with learned maps
 * the diffusion can instead drive neighbours to <em>different</em> (e.g. rotated or
 * sign-flipped) states. The maps are predicted from the two endpoint features,
 * {@code F_{v<e} = Phi([x_v || x_u])}, in one of three families ({@link Maps}).
 *
 * <p>Inputs: node features {@code [n, d * channels]} and the adjacency {@code [n, n]}.
 */
public final class SheafDiffusionLayer implements MultiInputLayer {

    /** The family of restriction maps. */
    public enum Maps {
        /** diagonal maps: {@code d} scalings per edge end */
        DIAGONAL,
        /** orthogonal maps: a product of {@code d} Householder reflections */
        ORTHOGONAL,
        /** general {@code d x d} maps */
        GENERAL
    }

    private final int stalk;
    private final int channels;
    private final Maps maps;
    private final Var mapWeight;
    private final Var mapBias;
    private final Var w1;
    private final Var w2;

    /**
     * Creates the layer.
     *
     * @param stalk the stalk dimension {@code d}
     * @param channels the channels per stalk dimension
     * @param maps the restriction-map family
     * @param random the initialization source
     */
    public SheafDiffusionLayer(int stalk, int channels, Maps maps, SplittableRandom random) {
        this.stalk = stalk;
        this.channels = channels;
        this.maps = Objects.requireNonNull(maps);
        int width = stalk * channels;
        int outputs = maps == Maps.DIAGONAL ? stalk : stalk * stalk;
        // small map weights: every edge starts near the bias (identity-like maps), so early
        // training diffuses instead of mixing neighbours through random signs
        this.mapWeight = Var.param(Tensor.normal(Shape.of(2 * width, outputs), 0, 0.01f, random));
        float[] b = new float[outputs];
        if (maps == Maps.GENERAL) {
            for (int i = 0; i < stalk; i++) {
                b[i * stalk + i] = 1;
            }
        } else if (maps == Maps.DIAGONAL) {
            java.util.Arrays.fill(b, 1);
        } else {
            for (int k = 0; k < stalk; k++) {
                b[k * stalk + k] = 1;
            }
        }
        this.mapBias = Var.param(Tensor.of(Shape.of(1, outputs), b));
        float[] eye = new float[stalk * stalk];
        for (int i = 0; i < stalk; i++) {
            eye[i * stalk + i] = 1;
        }
        this.w1 = Var.param(Tensor.of(Shape.of(stalk, stalk), eye));
        // identity-initialised channel mixing: the first steps are plain sheaf diffusion
        float[] eyeC = new float[channels * channels];
        for (int i = 0; i < channels; i++) {
            eyeC[i * channels + i] = 1;
        }
        this.w2 = Var.param(Tensor.of(Shape.of(channels, channels), eyeC));
    }

    /**
     * @param stalk the stalk dimension
     * @param maps the restriction-map family
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int stalk, Maps maps) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.get(0);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new SheafDiffusionLayer(stalk, inputs.get(0).dim(-1) / stalk, maps, random);
            }
        };
    }

    private Var eye(int e) {
        float[] eye = new float[stalk * stalk];
        for (int i = 0; i < stalk; i++) {
            eye[i * stalk + i] = 1;
        }
        return Var.of(Tensor.of(Shape.of(1, stalk, stalk), eye)).expand(Shape.of(e, stalk, stalk));
    }

    /**
     * The restriction maps of a batch of edge ends.
     *
     * @param pair {@code [E, 2 * d * channels]}: the end's own features, then the other end's
     * @return {@code [E, d, d]}
     */
    public Var restrictionMaps(Var pair) {
        int e = pair.shape().dim(0);
        Var raw = GraphMath.linear(pair, mapWeight, mapBias);
        return switch (maps) {
            case DIAGONAL -> raw.tanh().reshape(e, stalk, 1).mul(eye(e));
            case GENERAL -> raw.tanh().reshape(e, stalk, stalk);
            case ORTHOGONAL -> {
                Var q = null;
                for (int k = 0; k < stalk; k++) {
                    Var v = raw.slice(1, k * stalk, (k + 1) * stalk).reshape(e, stalk, 1);
                    Var outer = v.matmul(v.transpose()).div(v.square().sum(1, true).add(1e-6f));
                    Var h = eye(e).sub(outer.mul(2f));
                    q = q == null ? h : q.matmul(h);
                }
                yield q;
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        int n = x.shape().dim(0);
        float[] a = GraphMath.binary(inputs.get(1).value());
        int count = 0;
        float[] deg = new float[n];
        for (int i = 0; i < n * n; i++) {
            if (a[i] != 0) {
                count++;
                deg[i / n]++;
            }
        }
        if (count == 0) {
            return x;
        }
        float[] dst = new float[count];
        float[] src = new float[count];
        int c = 0;
        for (int i = 0; i < n * n; i++) {
            if (a[i] != 0) {
                dst[c] = i / n;
                src[c++] = i % n;
            }
        }
        float[] scale = new float[n];
        for (int i = 0; i < n; i++) {
            scale[i] = (float) (1 / Math.sqrt(deg[i] + 1));
        }
        Tensor dstT = Tensor.of(Shape.of(count), dst);
        Tensor srcT = Tensor.of(Shape.of(count), src);
        Tensor norm = Tensor.of(Shape.of(n, 1, 1), scale);
        Var xv = x.gather(0, dstT);
        Var xu = x.gather(0, srcT);
        Var fv = restrictionMaps(Var.concat(1, xv, xu));
        Var fu = restrictionMaps(Var.concat(1, xu, xv));
        Var y = x.reshape(n, stalk, channels).permute(0, 2, 1).matmul(w1.transpose()).permute(0, 2, 1)
                .matmul(w2).mul(norm);
        Var diff = fv.matmul(y.gather(0, dstT)).sub(fu.matmul(y.gather(0, srcT)));
        Var lap = fv.transpose().matmul(diff).scatterAdd(0, dstT, n).mul(norm);
        Var update = lap.activation(ActivationOp.ELU, 1f);
        return x.sub(update.reshape(n, stalk * channels));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        return List.of(mapWeight, mapBias, w1, w2);
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize;
    }

    @Override
    public String toString() {
        return "SheafDiffusionLayer(d=" + stalk + ", channels=" + channels + ", " + maps + ")";
    }
}
