package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TimesNet's block (Wu et al., ICLR 2023, <a href="https://arxiv.org/abs/2210.02186">arXiv:2210.02186</a>):
 * periods become 2-D images. The block finds the {@code k} frequencies with the largest
 * batch-mean FFT amplitude, and for each period {@code p = ceil(L / f)} folds the series
 * (zero-padded to a multiple of {@code p}) into a {@code [width, L / p, p]} image whose rows
 * are successive periods and whose columns are the phase within a period. A shared 2-D
 * inception block (parallel 1x1, 3x3 and 5x5 convolutions by default, averaged, GELU, then the same back
 * to {@code width} channels) captures intra-period variation along columns and inter-period
 * variation along rows at once; the images are unfolded, cut to {@code L}, and summed with
 * weights from a softmax over each sample's amplitudes at the chosen frequencies, plus a
 * residual. Input and output {@code [batch, time, width]}.
 */
public final class TimesBlock implements Layer {

    private final int width;
    private final int topK;
    private final Var[] up;
    private final Var[] upBias;
    private final Var[] down;
    private final Var[] downBias;
    private final int[] kernels;

    /**
     * Creates the block with inception kernels 1, 3 and 5.
     *
     * @param width the channels
     * @param hidden the inception block's inner channels
     * @param topK the number of periods
     * @param random the initialisation source
     */
    public TimesBlock(int width, int hidden, int topK, SplittableRandom random) {
        this(width, hidden, topK, new int[] {1, 3, 5}, random);
    }

    /**
     * Creates the block.
     *
     * @param width the channels
     * @param hidden the inception block's inner channels
     * @param topK the number of periods
     * @param kernels the odd square kernel sizes of the inception branches
     * @param random the initialisation source
     */
    public TimesBlock(int width, int hidden, int topK, int[] kernels, SplittableRandom random) {
        this.width = width;
        this.topK = topK;
        this.kernels = kernels.clone();
        up = new Var[kernels.length];
        upBias = new Var[kernels.length];
        down = new Var[kernels.length];
        downBias = new Var[kernels.length];
        for (int i = 0; i < kernels.length; i++) {
            int k = kernels[i];
            up[i] = Var.param(Tensor.normal(Shape.of(hidden, width, k, k), 0, (float) Math.sqrt(2.0 / (width * k * k)),
                    random));
            upBias[i] = Var.param(Tensor.zeros(Shape.of(hidden)));
            down[i] = Var.param(Tensor.normal(Shape.of(width, hidden, k, k), 0,
                    (float) Math.sqrt(1.0 / (hidden * k * k)), random));
            downBias[i] = Var.param(Tensor.zeros(Shape.of(width)));
        }
    }

    /**
     * @param hidden the inception block's inner channels
     * @param topK the number of periods
     * @return a spec for model builders
     */
    public static LayerSpec spec(int hidden, int topK) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new TimesBlock(input.dim(-1), hidden, topK, random);
            }
        };
    }

    private Var inception(Var x, Var[] w, Var[] bias) {
        Var sum = null;
        for (int i = 0; i < w.length; i++) {
            int p = kernels[i] / 2;
            Var y = Functional.conv(x, w[i], bias[i], new int[] {1, 1}, new int[] {p, p}, new int[] {p, p},
                    new int[] {1, 1}, 1);
            sum = sum == null ? y : sum.add(y);
        }
        return sum.mul(1f / w.length);
    }

    /**
     * The dominant frequencies of a batch: the {@code k} largest batch- and channel-mean FFT
     * amplitudes, excluding the mean.
     *
     * @param x {@code [batch, time, width]}
     * @param k how many
     * @return the frequency indices
     */
    public static int[] dominantFrequencies(Tensor x, int k) {
        Var amp = Complex.abs(Var.of(x).permute(0, 2, 1).rfft());                  // [b, width, F]
        float[] mean = amp.value().mean(0, false).mean(0, false).toArray();
        int[] out = new int[Math.min(k, mean.length - 1)];
        boolean[] used = new boolean[mean.length];
        used[0] = true;
        for (int i = 0; i < out.length; i++) {
            int best = -1;
            for (int f = 1; f < mean.length; f++) {
                if (!used[f] && (best < 0 || mean[f] > mean[best])) {
                    best = f;
                }
            }
            used[best] = true;
            out[i] = best;
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int[] freqs = dominantFrequencies(input.value(), topK);
        Var amp = Complex.abs(input.permute(0, 2, 1).rfft()).mean(1, false);      // [b, F]
        float[] idx = new float[freqs.length];
        for (int i = 0; i < freqs.length; i++) {
            idx[i] = freqs[i];
        }
        Var weights = amp.gather(1, Tensor.of(Shape.of(freqs.length), idx));
        weights = weights.sub(weights.logsumexp(1, true)).exp();                  // [b, k]
        Var out = input;
        for (int i = 0; i < freqs.length; i++) {
            int period = (int) Math.ceil(t / (double) freqs[i]);
            int rows = (int) Math.ceil(t / (double) period);
            int padded = rows * period;
            Var x = padded == t ? input
                    : Var.concat(1, input, Var.of(Tensor.zeros(Shape.of(b, padded - t, width))));
            Var image = x.reshape(b, rows, period, width).permute(0, 3, 1, 2);   // [b, width, rows, period]
            Var h = inception(Activation.GELU.forward(inception(image, up, upBias)), down, downBias);
            Var back = h.permute(0, 2, 3, 1).reshape(b, padded, width).slice(1, 0, t);
            out = out.add(back.mul(weights.slice(1, i, i + 1).reshape(b, 1, 1)));
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < kernels.length; i++) {
            p.addAll(List.of(up[i], upBias[i], down[i], downBias[i]));
        }
        return p;
    }

    @Override
    public String toString() {
        return "TimesBlock(" + width + ", k=" + topK + ")";
    }
}
