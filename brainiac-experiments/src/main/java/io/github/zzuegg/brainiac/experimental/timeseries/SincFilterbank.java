package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * SincNet's first layer (Ravanelli and Bengio, 2018, <a href="https://arxiv.org/abs/1808.00158">arXiv:1808.00158</a>):
 * a bank of band-pass filters {@code h[n] = 2 f2 sinc(2 pi f2 n) - 2 f1 sinc(2 pi f1 n)}
 * (Hamming-windowed) whose only trainable values are each filter's low cut-off {@code f1}
 * and bandwidth {@code f2 - f1}, initialised on the mel scale. The filters run over the raw
 * waveform (core {@link Functional#conv}); the squared outputs are averaged over each frame
 * and log-compressed, giving {@code [batch, frames, channels]} features in the shape of a
 * log-mel front end ({@link #energies} gives the linear energies, for example for
 * {@link Pcen}).
 */
public final class SincFilterbank implements Layer {

    /**
     * Cut-offs are stored in units of 100 Hz, so an optimiser step of size {@code lr} moves a
     * cut-off by about {@code 100 lr} Hz (SincNet stores Hz and trains them with a small rate).
     */
    private static final double UNIT = 100;

    private final int channels;
    private final int kernel;
    private final int stride;
    private final int hop;
    private final double rate;
    private final Var low;
    private final Var band;
    private final Tensor n;
    private final Tensor centre;
    private final Tensor window;

    /**
     * Creates the filterbank.
     *
     * @param channels filters
     * @param kernel the filter length in samples (odd)
     * @param stride the convolution stride (dividing {@code hop})
     * @param hop samples per output frame
     * @param rate the sample rate in Hz
     */
    public SincFilterbank(int channels, int kernel, int stride, int hop, double rate) {
        if (kernel % 2 == 0 || hop % stride != 0) {
            throw new IllegalArgumentException("kernel must be odd and stride must divide hop");
        }
        this.channels = channels;
        this.kernel = kernel;
        this.stride = stride;
        this.hop = hop;
        this.rate = rate;
        float[] lo = new float[channels];
        float[] bw = new float[channels];
        double melMax = mel(rate / 2 - 100);
        double melMin = mel(30);
        for (int c = 0; c < channels; c++) {
            double a = hz(melMin + (melMax - melMin) * c / (channels + 1));
            double b = hz(melMin + (melMax - melMin) * (c + 2) / (channels + 1));
            lo[c] = (float) (a / UNIT);
            bw[c] = (float) ((b - a) / UNIT);
        }
        low = Var.param(Tensor.of(Shape.of(channels, 1), lo));
        band = Var.param(Tensor.of(Shape.of(channels, 1), bw));
        float[] t = new float[kernel];
        float[] mask = new float[kernel];
        float[] w = new float[kernel];
        int half = kernel / 2;
        for (int i = 0; i < kernel; i++) {
            int k = i - half;
            t[i] = k == 0 ? 1 : k;
            mask[i] = k == 0 ? 1 : 0;
            w[i] = (float) (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (kernel - 1)));
        }
        n = Tensor.of(Shape.of(1, kernel), t);
        centre = Tensor.of(Shape.of(1, kernel), mask);
        window = Tensor.of(Shape.of(1, kernel), w);
    }

    static double mel(double hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    static double hz(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }

    /**
     * The current filters.
     *
     * @return {@code [channels, 1, kernel]}
     */
    public Var filters() {
        double minHz = 30 / rate;
        float unit = (float) (UNIT / rate);
        Var f1 = low.abs().mul(unit).add((float) minHz);
        Var f2 = f1.add(band.abs().mul(unit)).add((float) minHz).clamp(0f, 0.5f);
        Var arg1 = f1.mul(n).mul((float) (2 * Math.PI));
        Var arg2 = f2.mul(n).mul((float) (2 * Math.PI));
        Var off = arg2.sin().sub(arg1.sin()).div(n.mul((float) Math.PI));
        Var h = off.mul(centre.neg().add(1f)).add(f2.sub(f1).mul(2f).mul(centre));
        return h.mul(window).reshape(channels, 1, kernel);
    }

    /**
     * Frame energies.
     *
     * @param waves {@code [batch, samples]}, samples a multiple of {@code hop}
     * @return {@code [batch, samples / hop, channels]}
     */
    public Var energies(Var waves) {
        int b = waves.shape().dim(0);
        int samples = waves.shape().dim(1);
        int frames = samples / hop;
        int half = kernel / 2;
        Var y = Functional.conv(waves.reshape(b, 1, samples), filters(), null, new int[] {stride},
                new int[] {half}, new int[] {half}, new int[] {1}, 1);                 // [b, C, samples / stride]
        int per = hop / stride;
        return y.square().slice(2, 0, frames * per).reshape(b, channels, frames, per).mean(3, false)
                .permute(0, 2, 1);
    }

    @Override
    public Var forward(Var waves, boolean training) {
        return energies(waves).add(1e-6f).log().mul(0.2f);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    /**
     * The filters' cut-offs in Hz.
     *
     * @return {@code [channels][2]}
     */
    public double[][] cutoffs() {
        float[] l = low.value().toArray();
        float[] w = band.value().toArray();
        double[][] out = new double[channels][2];
        for (int c = 0; c < channels; c++) {
            double f1 = Math.abs(l[c]) * UNIT / rate + 30 / rate;
            out[c][0] = f1 * rate;
            out[c][1] = Math.min(0.5, f1 + Math.abs(w[c]) * UNIT / rate + 30 / rate) * rate;
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        return List.of(low, band);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(-1) / hop, channels);
    }

    @Override
    public String toString() {
        return "SincFilterbank(" + channels + " x " + kernel + ", stride " + stride + ", hop " + hop + ")";
    }
}
