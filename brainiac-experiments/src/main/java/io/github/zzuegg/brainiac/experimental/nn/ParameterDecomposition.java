package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Stochastic parameter decomposition (SPD) of one weight matrix: {@code W ~ sum_c V_c^T U_c},
 * a sum of rank-one <em>subcomponents</em>, each with a learned <em>causal importance</em>
 * {@code g_c(x)} in {@code [0, 1]} predicted from its inner activation {@code x . V_c} by a
 * tiny per-component MLP. Training asks for three things:
 *
 * <ul>
 *   <li><b>faithfulness</b>: the components add up to the original weights
 *       ({@link #faithfulness});</li>
 *   <li><b>stochastic reconstruction</b>: the model still computes the same output when each
 *       component is kept with a random mask {@code m_c = g_c + (1 - g_c) r},
 *       {@code r ~ U(0, 1)} ({@link #stochasticMask}), so components the gate calls
 *       unimportant must really be removable;</li>
 *   <li><b>importance minimality</b>: as few components as possible matter on each input
 *       ({@link #minimality}, an {@code L_p} penalty on the gates).</li>
 * </ul>
 *
 * <p>The result decomposes the <em>mechanisms</em> in the weights rather than the
 * activations: each component is one piece of computation that is used on a sparse set of
 * inputs. As a {@link Layer} it applies the summed weights, so it drops in where the
 * original linear map was; {@link #apply} and {@link #applyTransposed} take a mask for
 * ablation studies.
 *
 * <p>Bushnaq et al. 2025, "Stochastic Parameter Decomposition",
 * <a href="https://arxiv.org/abs/2506.20790">arXiv:2506.20790</a>; Braun et al. 2025,
 * "Interpretability in Parameter Space" (APD), <a href="https://arxiv.org/abs/2501.14926">arXiv:2501.14926</a>.
 */
public final class ParameterDecomposition implements Layer {

    private final int inputs;
    private final int outputs;
    private final int components;
    private final int gateHidden;
    private final Var v;
    private final Var u;
    private final Var gateIn;
    private final Var gateInBias;
    private final Var gateOut;
    private final Var gateOutBias;

    private ParameterDecomposition(int inputs, int outputs, int components, int gateHidden, SplittableRandom r) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.components = components;
        this.gateHidden = gateHidden;
        this.v = Var.param(Tensor.normal(Shape.of(components, inputs), 0, (float) (1 / Math.sqrt(inputs)), r));
        this.u = Var.param(Tensor.normal(Shape.of(components, outputs), 0, (float) (1 / Math.sqrt(components)), r));
        this.gateIn = Var.param(Tensor.normal(Shape.of(components, gateHidden), 0, 1, r));
        this.gateInBias = Var.param(Tensor.zeros(Shape.of(components, gateHidden)));
        this.gateOut = Var.param(Tensor.normal(Shape.of(components, gateHidden), 0, (float) (1 / Math.sqrt(gateHidden)), r));
        this.gateOutBias = Var.param(Tensor.full(Shape.of(components), 0.5f));
    }

    /**
     * A decomposition of an {@code inputs x outputs} matrix (used as {@code x W}).
     *
     * @param inputs the rows of W
     * @param outputs the columns of W
     * @param components the number of rank-one subcomponents
     * @param gateHidden the hidden width of each component's importance MLP
     * @param seed the seed
     * @return the decomposition
     */
    public static ParameterDecomposition of(int inputs, int outputs, int components, int gateHidden, long seed) {
        return new ParameterDecomposition(inputs, outputs, components, gateHidden, new SplittableRandom(seed));
    }

    /** @return the summed weights {@code V^T U}, {@code [inputs, outputs]} */
    public Var weight() {
        return v.transpose().matmul(u);
    }

    /**
     * Every component's inner activation {@code x . V_c}.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, components]}
     */
    public Var innerActs(Var x) {
        return x.matmul(v.transpose());
    }

    /**
     * The raw gate outputs: a per-component one-hidden-layer MLP on each inner activation.
     * Pass them through {@link #lowerLeaky} for masks and {@link #upperLeaky} for the
     * minimality penalty, as in the paper, so a gate stuck at 0 can still turn on and one
     * stuck at 1 can still turn off.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, components]}
     */
    public Var gateLogits(Var x) {
        Var a = innerActs(x);
        int batch = a.shape().dim(0);
        Var h = a.reshape(batch, components, 1).mul(gateIn).add(gateInBias).relu();
        return h.mul(gateOut).sum(2, false).add(gateOutBias);
    }

    /**
     * Causal importances in {@code [0, 1]}: the hard sigmoid of {@link #gateLogits}.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, components]}
     */
    public Var importance(Var x) {
        return gateLogits(x).clamp(0, 1);
    }

    /**
     * The lower-leaky hard sigmoid, {@code clamp(t, 0, 1) + 0.01 min(t, 0)}, for masks.
     *
     * @param t gate logits
     * @return the gates
     */
    public static Var lowerLeaky(Var t) {
        return t.clamp(0, 1).add(t.minimum(0f).mul(0.01f));
    }

    /**
     * The upper-leaky hard sigmoid, {@code clamp(t, 0, 1) + 0.01 max(t - 1, 0)}, for the
     * minimality penalty.
     *
     * @param t gate logits
     * @return the gates
     */
    public static Var upperLeaky(Var t) {
        return t.clamp(0, 1).add(t.sub(1f).maximum(0f).mul(0.01f));
    }

    /**
     * A stochastic mask {@code g + (1 - g) r} with {@code r ~ U(0, 1)}.
     *
     * @param importance gates {@code [batch, components]}, for example {@link #lowerLeaky}
     * @param random the noise source
     * @return the mask
     */
    public static Var stochasticMask(Var importance, SplittableRandom random) {
        Tensor r = Tensor.uniform(importance.shape(), 0, 1, random);
        return importance.add(importance.neg().add(1f).mul(r));
    }

    /**
     * {@code x W} with each component scaled by a mask.
     *
     * @param x {@code [batch, inputs]}
     * @param mask {@code [batch, components]}, or null for all ones
     * @return {@code [batch, outputs]}
     */
    public Var apply(Var x, Var mask) {
        Var a = innerActs(x);
        return (mask == null ? a : a.mul(mask)).matmul(u);
    }

    /**
     * {@code h W^T} with each component scaled by a mask, for tied weights used twice.
     *
     * @param h {@code [batch, outputs]}
     * @param mask {@code [batch, components]}, or null for all ones
     * @return {@code [batch, inputs]}
     */
    public Var applyTransposed(Var h, Var mask) {
        Var a = h.matmul(u.transpose());
        return (mask == null ? a : a.mul(mask)).matmul(v);
    }

    /**
     * The squared distance between the summed components and the target weights.
     *
     * @param target the original {@code [inputs, outputs]} weights
     * @return the scalar loss
     */
    public Var faithfulness(Tensor target) {
        return weight().sub(target).square().sum();
    }

    /**
     * The importance-minimality penalty, {@code mean_batch sum_c |g_c|^p}.
     *
     * @param importance gates {@code [batch, components]}
     * @param p the exponent, at most 1 for sparsity
     * @return the scalar penalty
     */
    public static Var minimality(Var importance, float p) {
        Var g = importance.abs().add(1e-6f);
        return (p == 1 ? g : g.pow(p)).sum(1, false).mean();
    }

    @Override
    public Var forward(Var x) {
        return x.matmul(weight());
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public List<Var> parameters() {
        return List.of(v, u, gateIn, gateInBias, gateOut, gateOutBias);
    }

    /**
     * One component as a matrix.
     *
     * @param c the component
     * @return {@code V_c^T U_c}, {@code [inputs][outputs]}
     */
    public float[][] component(int c) {
        float[] vv = v.value().to(DType.FLOAT32).toArray();
        float[] uu = u.value().to(DType.FLOAT32).toArray();
        float[][] out = new float[inputs][outputs];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < outputs; j++) {
                out[i][j] = vv[c * inputs + i] * uu[c * outputs + j];
            }
        }
        return out;
    }

    /** @return the number of components */
    public int components() {
        return components;
    }

    @Override
    public String toString() {
        return "ParameterDecomposition(" + inputs + "x" + outputs + " -> " + components + " rank-1, gate "
                + gateHidden + ")";
    }
}
