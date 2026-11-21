package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.Conv;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Lin;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

import static io.github.zzuegg.brainiac.audio.ChatterboxNets.conv;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.lin;

/**
 * CosyVoice's HiFT vocoder as Chatterbox uses it: a pitch predictor, a neural source-filter
 * excitation (sines at the pitch and 8 harmonics), and a HiFi-GAN generator with Snake
 * activations whose last layer gives the magnitude and phase of a 16-point inverse STFT,
 * turning an 80-band mel at 50 Hz into 24 kHz audio.
 */
final class HiftVocoder {

    static final int SAMPLE_RATE = 24_000;
    private static final int[] UP_RATES = {8, 5, 3};
    private static final int N_FFT = 16;
    private static final int HOP = 4;
    private static final float VOICED = 10f;

    private record Snake(Var alpha) {
        Var apply(Var x) {
            Var a = alpha.reshape(alpha.shape().dim(0), 1);
            return x.add(x.mul(a).sin().square().div(a.add(1e-9f)));
        }
    }

    private record ResBlock(List<Conv> convs1, List<Conv> convs2, List<Snake> act1, List<Snake> act2,
            int[] dilations) {
        Var apply(Var x) {
            for (int i = 0; i < convs1.size(); i++) {
                Var t = convs1.get(i).same(act1.get(i).apply(x), dilations[i]);
                t = convs2.get(i).same(act2.get(i).apply(t), 1);
                x = Vocos.settle(t.add(x));
            }
            return x;
        }
    }

    private final Device device;
    private final DType dtype;
    private final List<Conv> f0Convs = new ArrayList<>();
    private final Lin f0Out;
    private final float[] sourceW;
    private final float sourceB;
    private final Conv pre;
    private final List<Conv> ups = new ArrayList<>();
    private final List<Conv> sourceDowns = new ArrayList<>();
    private final List<ResBlock> sourceBlocks = new ArrayList<>();
    private final List<ResBlock> blocks = new ArrayList<>();
    private final Conv post;
    private final int kernels;

    HiftVocoder(Weights w, String p) {
        device = w.device();
        dtype = w.dtype();
        for (int i = 0; w.contains(p + "f0_predictor.condnet." + (2 * i) + ".bias"); i++) {
            f0Convs.add(conv(w, p + "f0_predictor.condnet." + (2 * i)));
        }
        f0Out = lin(w, p + "f0_predictor.classifier");
        sourceW = w.floats(p + "m_source.l_linear.weight");
        sourceB = w.floats(p + "m_source.l_linear.bias")[0];
        pre = conv(w, p + "conv_pre");
        for (int i = 0; i < UP_RATES.length; i++) {
            ups.add(conv(w, p + "ups." + i));
            sourceDowns.add(conv(w, p + "source_downs." + i));
            sourceBlocks.add(resBlock(w, p + "source_resblocks." + i));
        }
        int k = 0;
        while (w.contains(p + "resblocks." + k + ".activations1.0.alpha")) {
            blocks.add(resBlock(w, p + "resblocks." + k));
            k++;
        }
        kernels = k / UP_RATES.length;
        post = conv(w, p + "conv_post");
    }

    private static ResBlock resBlock(Weights w, String p) {
        List<Conv> c1 = new ArrayList<>();
        List<Conv> c2 = new ArrayList<>();
        List<Snake> a1 = new ArrayList<>();
        List<Snake> a2 = new ArrayList<>();
        for (int i = 0; w.contains(p + ".activations1." + i + ".alpha"); i++) {
            c1.add(conv(w, p + ".convs1." + i));
            c2.add(conv(w, p + ".convs2." + i));
            a1.add(new Snake(w.read(p + ".activations1." + i + ".alpha")));
            a2.add(new Snake(w.read(p + ".activations2." + i + ".alpha")));
        }
        int[] dil = new int[c1.size()];
        for (int i = 0; i < dil.length; i++) {
            dil[i] = 2 * i + 1;   // 1, 3, 5
        }
        return new ResBlock(c1, c2, a1, a2, dil);
    }

    /** Output samples per mel frame. */
    static int hop() {
        int h = HOP;
        for (int r : UP_RATES) {
            h *= r;
        }
        return h;
    }

    /**
     * The predicted pitch of each frame, in Hz.
     *
     * @param mel {@code [80, frames]}
     */
    float[] pitch(Var mel) {
        Var x = mel.reshape(1, mel.shape().dim(0), mel.shape().dim(1)).to(dtype);
        for (Conv c : f0Convs) {
            x = c.same(x, 1);
            x = x.relu().add(x.minimum(0).exp().sub(1));      // ELU
        }
        Var f = f0Out.apply(x.permute(0, 2, 1)).abs();
        return f.to(DType.FLOAT32).value().toArray();
    }

    /**
     * The excitation: sines at the pitch and its harmonics (random starting phases above the
     * first), noise where unvoiced, merged by a learned weight and {@code tanh}.
     *
     * @param f0 the pitch of each frame
     * @param phases the starting phase of each sine (the first 0)
     * @param noise standard normal values {@code [harmonics + 1][samples]}, or {@code null} for none
     * @return {@code samples} values
     */
    float[] source(float[] f0, float[] phases, float[][] noise) {
        int up = hop();
        int n = f0.length * up;
        int sines = sourceW.length;
        float[] out = new float[n];
        double[] acc = new double[sines];
        float[] merged = new float[n];
        java.util.Arrays.fill(merged, sourceB);
        for (int i = 0; i < n; i++) {
            float f = f0[i / up];
            boolean voiced = f > VOICED;
            float amp = voiced ? 0.003f : 0.1f / 3;
            for (int h = 0; h < sines; h++) {
                acc[h] += f * (h + 1) / SAMPLE_RATE;
                float cum = (float) acc[h];
                float frac = cum % 1f;
                float theta = (float) (2 * Math.PI) * frac;
                float s = 0.1f * (float) Math.sin(theta + phases[h]);
                s = (voiced ? s : 0) + (noise == null ? 0 : amp * noise[h][i]);
                merged[i] += sourceW[h] * s;
            }
            out[i] = (float) Math.tanh(merged[i]);
        }
        return out;
    }

    /**
     * Audio from a mel: the pitch, a random excitation, the generator.
     *
     * @param mel {@code [80, frames]}
     * @param random the random phases and noise
     * @return 24 kHz samples
     */
    float[] synthesize(Var mel, RandomGenerator random) {
        float[] f0 = pitch(mel);
        int sines = sourceW.length;
        float[] phases = new float[sines];
        for (int h = 1; h < sines; h++) {
            phases[h] = (float) ((random.nextDouble() * 2 - 1) * Math.PI);
        }
        int n = f0.length * hop();
        float[][] noise = new float[sines][n];
        for (float[] row : noise) {
            for (int i = 0; i < n; i++) {
                row[i] = (float) random.nextGaussian();
            }
        }
        return decode(mel, source(f0, phases, noise));
    }

    /**
     * The generator: the mel and the excitation's STFT to audio, clamped to +-0.99.
     *
     * @param mel {@code [80, frames]}
     * @param source the excitation, {@code frames * 480} samples
     */
    float[] decode(Var mel, float[] source) {
        Var s = Var.of(Tensor.of(Shape.of(1, source.length), source, device).to(dtype));
        Var spec = Signal.stft(s, N_FFT, HOP);                                 // [1, frames, 9, 2]
        int sf = spec.shape().dim(1);
        int bins = N_FFT / 2 + 1;
        Var re = spec.slice(3, 0, 1).reshape(1, sf, bins).permute(0, 2, 1);
        Var im = spec.slice(3, 1, 2).reshape(1, sf, bins).permute(0, 2, 1);
        Var sStft = Vocos.settle(Var.concat(1, re, im));                       // [1, 18, sf]
        Var x = pre.same(mel.reshape(1, mel.shape().dim(0), mel.shape().dim(1)).to(dtype), 1);
        for (int i = 0; i < UP_RATES.length; i++) {
            x = x.leakyRelu(0.1f);
            Conv u = ups.get(i);
            int k = u.kernel();
            int r = UP_RATES[i];
            int pad = (k - r) / 2;
            x = Functional.convTranspose(x, u.w(), u.b(), new int[] {r}, new int[] {pad}, new int[] {pad},
                    new int[] {0}, new int[] {1}, 1);
            if (i == UP_RATES.length - 1) {
                x = Functional.padAxis(x, 2, 1, 0, PadMode.REFLECT, 0);
            }
            Conv sd = sourceDowns.get(i);
            int stride = sd.kernel() == 1 ? 1 : sd.kernel() / 2;
            int sp = stride == 1 ? 0 : stride / 2;
            Var si = sourceBlocks.get(i).apply(sd.apply(sStft, stride, sp, sp, 1));
            x = Vocos.settle(x.add(si));
            Var sum = null;
            for (int j = 0; j < kernels; j++) {
                Var y = blocks.get(i * kernels + j).apply(x);
                sum = sum == null ? y : Vocos.settle(sum.add(y));
            }
            x = sum.mul(1f / kernels);
        }
        x = post.same(x.leakyRelu(0.01f), 1);                                   // [1, 18, frames]
        int frames = x.shape().dim(2);
        Var mag = x.slice(1, 0, bins).exp().clamp(0f, 100f);
        Var ph = x.slice(1, bins, 2 * bins).sin();
        Var cplx = Var.concat(3, mag.mul(ph.cos()).permute(0, 2, 1).reshape(1, frames, bins, 1),
                mag.mul(ph.sin()).permute(0, 2, 1).reshape(1, frames, bins, 1));
        Var audio = Signal.istft(cplx, N_FFT, HOP, HOP * (frames - 1)).clamp(-0.99f, 0.99f);
        return audio.to(DType.FLOAT32).value().toArray();
    }

    /** The reference's fade-in: 20 ms of silence, then 20 ms of a raised cosine. */
    static void fadeIn(float[] audio) {
        int n = SAMPLE_RATE / 50;
        for (int i = 0; i < Math.min(2 * n, audio.length); i++) {
            float g = 0;
            if (i >= n) {
                double t = Math.PI + (0 - Math.PI) * (i - n) / (double) (n - 1);
                g = (float) ((Math.cos(t) + 1) / 2);
            }
            audio[i] *= g;
        }
    }
}
