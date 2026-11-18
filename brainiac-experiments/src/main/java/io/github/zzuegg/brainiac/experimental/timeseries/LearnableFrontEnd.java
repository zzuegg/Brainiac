package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * LEAF (Zeghidour et al., ICLR 2021, <a href="https://arxiv.org/abs/2101.08596">arXiv:2101.08596</a>):
 * a fully learnable audio front end in three stages. (1) Complex Gabor filters
 * {@code phi(n) = exp(-n^2 / (2 sigma^2)) / (sqrt(2 pi) sigma) exp(i 2 pi eta n)} with a
 * trainable centre frequency {@code eta} and width {@code sigma} per channel, initialised to
 * match a mel filterbank, convolved with the waveform; the squared modulus gives a
 * band-limited energy. (2) Gaussian lowpass pooling: a depthwise convolution with a
 * per-channel Gaussian window of trainable width, strided to the frame rate. (3)
 * {@link Pcen} with trainable parameters instead of a log. Input {@code [batch, samples]}
 * (a multiple of {@code hop}), output {@code [batch, samples / hop, channels]}.
 */
public final class LearnableFrontEnd implements Layer {

    /** Centre frequencies are stored in units of 100 Hz (see {@link SincFilterbank}). */
    private static final double UNIT = 100;

    private final int channels;
    private final int kernel;
    private final double rate;
    private final int stride;
    private final int hop;
    private final Var eta;
    private final Var sigma;
    private final Var poolWidth;
    private final Pcen pcen;
    private final Tensor n;
    private final Tensor poolN;

    /**
     * Creates the front end.
     *
     * @param channels filters
     * @param kernel the Gabor filter length in samples (odd)
     * @param stride the Gabor convolution stride (dividing {@code hop})
     * @param hop samples per output frame
     * @param frames output frames (for PCEN's smoother)
     * @param rate the sample rate in Hz
     */
    public LearnableFrontEnd(int channels, int kernel, int stride, int hop, int frames, double rate) {
        if (kernel % 2 == 0 || hop % stride != 0) {
            throw new IllegalArgumentException("kernel must be odd and stride must divide hop");
        }
        this.channels = channels;
        this.kernel = kernel;
        this.stride = stride;
        this.hop = hop;
        this.rate = rate;
        float[] e = new float[channels];
        float[] s = new float[channels];
        double melMax = SincFilterbank.mel(rate / 2 - 100);
        double melMin = SincFilterbank.mel(30);
        for (int c = 0; c < channels; c++) {
            double a = SincFilterbank.hz(melMin + (melMax - melMin) * c / (channels + 1));
            double centreHz = SincFilterbank.hz(melMin + (melMax - melMin) * (c + 1) / (channels + 1));
            double b = SincFilterbank.hz(melMin + (melMax - melMin) * (c + 2) / (channels + 1));
            e[c] = (float) (centreHz / UNIT);
            // a Gaussian of standard deviation sigma samples has a -3 dB bandwidth of about 0.375 / sigma cycles
            double bw = Math.max(b - a, 40) / rate;
            s[c] = (float) Math.log(Math.min(kernel / 3.0, 0.375 / bw));
        }
        eta = Var.param(Tensor.of(Shape.of(channels, 1), e));
        sigma = Var.param(Tensor.of(Shape.of(channels, 1), s));
        poolWidth = Var.param(Tensor.full(Shape.of(channels, 1), (float) Math.log(0.4)));
        pcen = new Pcen(channels, frames, 0.15);
        float[] t = new float[kernel];
        for (int i = 0; i < kernel; i++) {
            t[i] = i - kernel / 2;
        }
        n = Tensor.of(Shape.of(1, kernel), t);
        int per = hop / stride;
        float[] p = new float[2 * per];
        for (int i = 0; i < p.length; i++) {
            p[i] = (float) ((i - (p.length - 1) / 2.0) / (p.length / 2.0));
        }
        poolN = Tensor.of(Shape.of(1, 2 * per), p);
    }

    /**
     * The Gabor filters, real parts then imaginary parts.
     *
     * @return {@code [2 channels, 1, kernel]}
     */
    public Var filters() {
        Var sd = sigma.exp();
        Var env = Var.of(n.square()).div(sd.square().mul(-2f)).exp().div(sd.mul((float) Math.sqrt(2 * Math.PI)));
        Var phase = eta.mul(n).mul((float) (2 * Math.PI * UNIT / rate));
        return Var.concat(0, env.mul(phase.cos()), env.mul(phase.sin())).reshape(2 * channels, 1, kernel);
    }

    /**
     * The pooled energies before PCEN.
     *
     * @param waves {@code [batch, samples]}
     * @return {@code [batch, samples / hop, channels]}
     */
    public Var energies(Var waves) {
        int b = waves.shape().dim(0);
        int samples = waves.shape().dim(1);
        int half = kernel / 2;
        Var y = Functional.conv(waves.reshape(b, 1, samples), filters(), null, new int[] {stride},
                new int[] {half}, new int[] {half}, new int[] {1}, 1);
        Var power = y.slice(1, 0, channels).square().add(y.slice(1, channels, 2 * channels).square());
        int per = hop / stride;
        Var w = Var.of(poolN.square()).div(poolWidth.exp().square().mul(-2f)).exp();
        w = w.div(w.sum(1, true)).reshape(channels, 1, 2 * per);
        return Functional.conv(power, w, null, new int[] {per}, new int[] {per / 2}, new int[] {per / 2},
                new int[] {1}, channels).permute(0, 2, 1);
    }

    @Override
    public Var forward(Var waves, boolean training) {
        return pcen.forward(energies(waves).mul(1e3f), training);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    /** @return the PCEN stage */
    public Pcen pcen() {
        return pcen;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(eta, sigma, poolWidth));
        p.addAll(pcen.parameters());
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(-1) / hop, channels);
    }

    @Override
    public String toString() {
        return "LearnableFrontEnd(" + channels + " Gabor x " + kernel + ", stride " + stride + ", hop " + hop + ")";
    }
}
