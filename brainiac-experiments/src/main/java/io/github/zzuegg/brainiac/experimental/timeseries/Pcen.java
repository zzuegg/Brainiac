package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Per-channel energy normalisation (Wang et al., 2017, <a href="https://arxiv.org/abs/1607.05666">arXiv:1607.05666</a>),
 * the adaptive gain control that replaces the log of a filterbank front end:
 * {@code M_t = (1 - s) M_t-1 + s E_t} (a first-order smoother, {@code M_0 = E_0}), then
 * {@code PCEN = (E / (eps + M)^alpha + delta)^r - delta^r}. Dividing by the smoothed energy
 * removes slowly varying (stationary) loudness and noise; the root compresses the rest. The
 * per-channel {@code alpha} (in (0, 1)), {@code delta} (positive), {@code r} (in (0, 1)) and
 * {@code s} (in (0, 1)) are all trainable, as in LEAF; untrained, the defaults give classic
 * PCEN as a drop-in after any mel or filterbank energies. Input and output {@code [batch,
 * frames, channels]}, inputs non-negative energies.
 */
public final class Pcen implements Layer {

    private static final float EPS = 1e-6f;

    private final int channels;
    private final int frames;
    private final Var alpha;
    private final Var delta;
    private final Var root;
    private final Var smooth;
    private final Tensor lags;
    private final Tensor lower;
    private final Tensor first;

    /**
     * Creates the layer with classic defaults ({@code alpha = 0.98}, {@code delta = 2},
     * {@code r = 0.5}).
     *
     * @param channels channels
     * @param frames frames per input
     * @param s the initial smoothing coefficient per frame, for example 0.04-0.2
     */
    public Pcen(int channels, int frames, double s) {
        this.channels = channels;
        this.frames = frames;
        alpha = Var.param(Tensor.full(Shape.of(channels), (float) logit(0.98)));
        delta = Var.param(Tensor.full(Shape.of(channels), (float) Math.log(2.0)));
        root = Var.param(Tensor.full(Shape.of(channels), (float) logit(0.5)));
        smooth = Var.param(Tensor.full(Shape.of(channels), (float) logit(s)));
        float[] d = new float[frames * frames];
        float[] l = new float[frames * frames];
        float[] f = new float[frames * frames];
        for (int t = 0; t < frames; t++) {
            for (int j = 0; j <= t; j++) {
                d[t * frames + j] = t - j;
                l[t * frames + j] = j == 0 ? 0 : 1;
                f[t * frames + j] = j == 0 ? 1 : 0;
            }
        }
        lags = Tensor.of(Shape.of(1, frames, frames), d);
        lower = Tensor.of(Shape.of(1, frames, frames), l);
        first = Tensor.of(Shape.of(1, frames, frames), f);
    }

    private static double logit(double p) {
        return Math.log(p / (1 - p));
    }

    /**
     * The smoothed energy {@code M}.
     *
     * @param energies {@code [batch, frames, channels]}
     * @return {@code [batch, frames, channels]}
     */
    public Var smoothed(Var energies) {
        int b = energies.shape().dim(0);
        Var s = smooth.sigmoid().reshape(channels, 1, 1);
        Var logKeep = s.neg().add(1f).log();
        // A[c, t, j] = (1 - s)^(t - j) (s for j >= 1, 1 for j = 0), j <= t
        Var a = logKeep.mul(lags).exp().mul(s.mul(lower).add(first));
        Var e = energies.permute(0, 2, 1).reshape(b, channels, 1, frames);
        return e.mul(a).sum(-1, false).permute(0, 2, 1);
    }

    @Override
    public Var forward(Var energies, boolean training) {
        Var m = smoothed(energies);
        Var a = alpha.sigmoid();
        Var d = delta.exp();
        Var r = root.sigmoid();
        Var gain = m.add(EPS).log().mul(a).neg().exp();
        return energies.mul(gain).add(d).log().mul(r).exp().sub(d.log().mul(r).exp());
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public List<Var> parameters() {
        return List.of(alpha, delta, root, smooth);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "Pcen(" + channels + " channels, " + frames + " frames)";
    }
}
