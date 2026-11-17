package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * S4D: a diagonal structured state-space layer. Each channel is an independent continuous
 * linear system {@code x' = A x + B u, y = 2 Re(C x) + D u} with a diagonal complex {@code A}
 * ({@code N/2} conjugate pairs), discretized with step {@code delta} by zero-order hold:
 * {@code A_bar = exp(delta A)}, {@code B_bar = (A_bar - 1) / A * B}. The same layer runs two ways:
 *
 * <ul>
 *   <li><b>convolution</b> ({@link #forward(Var)}, training): the impulse response
 *       {@code K[l] = 2 Re(sum_n C_n B_bar_n A_bar_n^l)} is computed with a Vandermonde
 *       product and applied as a causal long convolution through the FFT;</li>
 *   <li><b>recurrence</b> ({@link #recurrent(Var)}, inference): {@code x_t = A_bar x_{t-1} + B_bar u_t},
 *       constant memory per step.</li>
 * </ul>
 *
 * <p>Initialisation is S4D-Lin: {@code A_n = -1/2 + i pi n}, {@code B = 1}, {@code log delta}
 * uniform in {@code [log 0.001, log 0.1]}. The SSM output passes GELU and a linear mix of the
 * channels. The root of the SSM family (S4, Mamba): one layer, two computations.
 *
 * <p>Papers: Gu et al., "On the Parameterization and Initialization of Diagonal State Space
 * Models" (S4D), <a href="https://arxiv.org/abs/2206.11893">arXiv:2206.11893</a>; Gu et al.,
 * "Efficiently Modeling Long Sequences with Structured State Spaces" (S4),
 * <a href="https://arxiv.org/abs/2111.00396">arXiv:2111.00396</a>.
 */
public final class S4dLayer implements Layer {

    private final int features;
    private final int modes;
    private final Var logDelta;
    private final Var aRealLog;
    private final Var aImag;
    private final Var cRe;
    private final Var cIm;
    private final Var skip;
    private final Var mix;
    private final Var mixBias;

    /**
     * Creates the layer.
     *
     * @param features the number of channels
     * @param state the state size {@code N} per channel (even; {@code N/2} complex modes)
     * @param random the random source
     */
    public S4dLayer(int features, int state, RandomGenerator random) {
        if (state % 2 != 0) {
            throw new IllegalArgumentException("state must be even, got " + state);
        }
        this.features = features;
        this.modes = state / 2;
        float[] dt = new float[features];
        for (int h = 0; h < features; h++) {
            dt[h] = (float) (Math.log(0.001) + random.nextDouble() * (Math.log(0.1) - Math.log(0.001)));
        }
        this.logDelta = Var.param(Tensor.of(Shape.of(features, 1), dt));
        this.aRealLog = Var.param(Tensor.full(Shape.of(features, modes), (float) Math.log(0.5)));
        float[] im = new float[features * modes];
        for (int h = 0; h < features; h++) {
            for (int n = 0; n < modes; n++) {
                im[h * modes + n] = (float) (Math.PI * n);
            }
        }
        this.aImag = Var.param(Tensor.of(Shape.of(features, modes), im));
        this.cRe = Var.param(Tensor.normal(Shape.of(features, modes), 0, (float) Math.sqrt(0.5), random));
        this.cIm = Var.param(Tensor.normal(Shape.of(features, modes), 0, (float) Math.sqrt(0.5), random));
        this.skip = MixerOps.filled(1f, features);
        this.mix = MixerOps.weight(random, features, features);
        this.mixBias = MixerOps.filled(0f, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param state the state size per channel
     * @return the spec
     */
    public static LayerSpec of(int state) {
        return MixerOps.sameShape((in, r) -> new S4dLayer(in.dim(-1), state, r));
    }

    /** Discretized parameters: {A_bar re, A_bar im, C B_bar re, C B_bar im}, each [H, modes]. */
    private Var[] discretize() {
        Var delta = logDelta.exp();
        Var are = aRealLog.exp().neg();
        Var aim = aImag;
        Var mag = are.mul(delta).exp();
        Var ang = aim.mul(delta);
        Var abRe = mag.mul(MixerOps.cos(ang));
        Var abIm = mag.mul(MixerOps.sin(ang));
        // B_bar = (A_bar - 1) / A  (B = 1): multiply by conj(A) / |A|^2
        Var norm = are.square().add(aim.square());
        Var nr = abRe.sub(1f);
        Var bRe = nr.mul(are).add(abIm.mul(aim)).div(norm);
        Var bIm = abIm.mul(are).sub(nr.mul(aim)).div(norm);
        Var cbRe = cRe.mul(bRe).sub(cIm.mul(bIm));
        Var cbIm = cRe.mul(bIm).add(cIm.mul(bRe));
        return new Var[] {abRe, abIm, cbRe, cbIm, delta, are, aim};
    }

    /**
     * Returns the convolution kernel.
     *
     * @param length the kernel length
     * @return {@code [features, length]}
     */
    public Var kernel(int length) {
        Var[] p = discretize();
        float[] steps = new float[length];
        for (int l = 0; l < length; l++) {
            steps[l] = l;
        }
        Tensor ls = Tensor.of(Shape.of(1, 1, length), steps);
        Var dt = p[4].reshape(features, 1, 1);
        Var re = p[5].reshape(features, modes, 1).mul(dt).mul(ls);     // l delta Re(A)
        Var im = p[6].reshape(features, modes, 1).mul(dt).mul(ls);
        Var mag = re.exp();
        Var powRe = mag.mul(MixerOps.cos(im));
        Var powIm = mag.mul(MixerOps.sin(im));
        Var cbRe = p[2].reshape(features, modes, 1);
        Var cbIm = p[3].reshape(features, modes, 1);
        return cbRe.mul(powRe).sub(cbIm.mul(powIm)).sum(1, false).mul(2f);
    }

    private Var finish(Var u, Var y) {
        return Activation.GELU.forward(y.add(u.mul(skip))).matmul(mix).add(mixBias);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "S4dLayer");
        int t = input.shape().dim(1);
        Var y = MixerOps.causalConvolution(input.permute(0, 2, 1), kernel(t)).permute(0, 2, 1);
        return finish(input, y);
    }

    /**
     * Computes the output with the recurrence.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = discretize();
        // x = A_bar x + B_bar u, tracked as C x (C commutes with the diagonal A_bar)
        Var xr = Var.of(Tensor.zeros(Shape.of(b, features, modes)));
        Var xi = xr;
        List<Var> ys = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var u = input.slice(1, s, s + 1).reshape(b, features, 1);
            Var nr = xr.mul(p[0]).sub(xi.mul(p[1])).add(u.mul(p[2]));
            Var ni = xr.mul(p[1]).add(xi.mul(p[0])).add(u.mul(p[3]));
            xr = nr;
            xi = ni;
            ys.add(xr.sum(-1, false).mul(2f).reshape(b, 1, features));
        }
        return finish(input, Var.concat(1, ys.toArray(Var[]::new)));
    }

    @Override
    public List<Var> parameters() {
        return List.of(logDelta, aRealLog, aImag, cRe, cIm, skip, mix, mixBias);
    }

    @Override
    public String toString() {
        return "S4dLayer(" + features + ", modes=" + modes + ")";
    }
}
