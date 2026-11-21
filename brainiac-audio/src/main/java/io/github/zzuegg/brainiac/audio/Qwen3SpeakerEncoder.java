package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The speaker encoder of the Qwen3-TTS Base models: an ECAPA-TDNN (Desplanques et al., 2020)
 * that turns a 128-band log-mel spectrogram of 24 kHz speech into one vector, which the talker
 * reads in place of a named speaker's embedding to speak in that voice.
 */
final class Qwen3SpeakerEncoder {

    static final int SAMPLE_RATE = 24_000;
    static final int N_FFT = 1024;
    static final int HOP = 256;
    static final int MELS = 128;

    private record Tdnn(Var w, Var b, int dilation) {
        Var apply(Var x) {
            int k = w.shape().dim(2);
            int total = dilation * (k - 1);
            Var in = total == 0 ? x : Functional.padAxis(x, 2, total / 2, total - total / 2, PadMode.REFLECT, 0);
            return Functional.conv(in, w, b, new int[] {1}, new int[] {0}, new int[] {0}, new int[] {dilation}, 1);
        }
    }

    private record Block(Tdnn tdnn1, List<Tdnn> res2net, int scale, Tdnn tdnn2, Tdnn se1, Tdnn se2) {
    }

    private final Device device;
    private final DType dtype;
    private final Tdnn first;
    private final List<Block> blocks = new ArrayList<>();
    private final Tdnn mfa;
    private final Tdnn aspTdnn;
    private final Tdnn aspConv;
    private final Tdnn fc;
    private final Tensor melBasis;

    Qwen3SpeakerEncoder(Weights w, Map<String, Object> config) {
        this.device = w.device();
        this.dtype = w.dtype();
        String p = "speaker_encoder.";
        // the released configs name only enc_dim; the rest are the reference's defaults
        List<Object> dilations = config.get("enc_dilations") == null
                ? List.of(1, 2, 3, 4, 1) : Json.array(config.get("enc_dilations"));
        int scale = 1;
        while (w.contains(p + "blocks.1.res2net_block.blocks." + (scale - 1) + ".conv.weight")) {
            scale++;
        }
        first = tdnn(w, p + "blocks.0.conv", dil(dilations, 0));
        for (int i = 1; i < dilations.size() - 1; i++) {
            String b = p + "blocks." + i + ".";
            List<Tdnn> parts = new ArrayList<>();
            for (int j = 0; j < scale - 1; j++) {
                parts.add(tdnn(w, b + "res2net_block.blocks." + j + ".conv", dil(dilations, i)));
            }
            blocks.add(new Block(tdnn(w, b + "tdnn1.conv", 1), parts, scale, tdnn(w, b + "tdnn2.conv", 1),
                    tdnn(w, b + "se_block.conv1", 1), tdnn(w, b + "se_block.conv2", 1)));
        }
        mfa = tdnn(w, p + "mfa.conv", dil(dilations, dilations.size() - 1));
        aspTdnn = tdnn(w, p + "asp.tdnn.conv", 1);
        aspConv = tdnn(w, p + "asp.conv", 1);
        fc = tdnn(w, p + "fc", 1);
        melBasis = Signal.melFilterbank(SAMPLE_RATE, N_FFT, MELS, 0, 12_000, Signal.MelScale.SLANEY, true)
                .to(device);
    }

    private static int dil(List<Object> d, int i) {
        return ((Number) d.get(i)).intValue();
    }

    private static Tdnn tdnn(Weights w, String name, int dilation) {
        return new Tdnn(w.read(name + ".weight"), w.read(name + ".bias"), dilation);
    }

    /**
     * The log-mel spectrogram the encoder reads, as the reference's {@code mel_spectrogram}:
     * the signal reflect-padded by 384 samples, uncentred 1024-sample Hann frames every 256,
     * magnitudes, Slaney mel filters up to 12 kHz and the log of at least 1e-5.
     *
     * @param audio 24 kHz samples, more than 384
     * @return {@code [frames, 128]}
     */
    Var mel(float[] audio) {
        int pad = (N_FFT - HOP) / 2;
        Var x = Var.of(Tensor.of(Shape.of(1, 1, audio.length), audio, device));
        Var padded = Functional.padAxis(x, 2, pad, pad, PadMode.REFLECT, 0).reshape(audio.length + 2 * pad);
        int frames = (audio.length + 2 * pad - N_FFT) / HOP + 1;
        float[] idx = new float[frames * N_FFT];
        for (int f = 0; f < frames; f++) {
            for (int j = 0; j < N_FFT; j++) {
                idx[f * N_FFT + j] = f * HOP + j;
            }
        }
        Var spec = padded.gather(0, Tensor.of(Shape.of(frames, N_FFT), idx, device))
                .mul(Var.of(Signal.hann(N_FFT, device))).rfft();             // [frames, bins, 2]
        Var mag = spec.square().sum(2, false).add(1e-9f).sqrt();
        return mag.matmul(Var.of(melBasis)).clamp(1e-5f, Float.MAX_VALUE).log();
    }

    /**
     * The speaker embedding of a voice.
     *
     * @param audio 24 kHz samples of speech
     * @return the embedding, float32
     */
    float[] embed(float[] audio) {
        return Var.noGrad(() -> {
            Var m = mel(audio);
            int frames = m.shape().dim(0);
            Var x = m.permute(1, 0).reshape(1, MELS, frames).to(dtype);
            x = first.apply(x).relu();
            List<Var> outs = new ArrayList<>();
            for (Block b : blocks) {
                Var h = b.tdnn1().apply(x).relu();
                int c = h.shape().dim(1) / b.scale();
                Var y = null;
                List<Var> parts = new ArrayList<>();
                for (int i = 0; i < b.scale(); i++) {
                    Var part = h.slice(1, i * c, (i + 1) * c);
                    if (i == 0) {
                        y = part;
                    } else if (i == 1) {
                        y = b.res2net().get(0).apply(part).relu();
                    } else {
                        y = b.res2net().get(i - 1).apply(part.add(y)).relu();
                    }
                    parts.add(y);
                }
                h = Var.concat(1, parts.toArray(Var[]::new));
                h = b.tdnn2().apply(h).relu();
                Var s = b.se2().apply(b.se1().apply(h.mean(2, true)).relu()).sigmoid();
                x = Vocos.settle(x.add(h.mul(s)));
                outs.add(x);
            }
            x = mfa.apply(Var.concat(1, outs.toArray(Var[]::new))).relu();
            int t = x.shape().dim(2);
            Var mean = x.mean(2, true);
            Var std = x.sub(mean).square().mean(2, true).clamp(1e-12f, Float.MAX_VALUE).sqrt();
            Var att = Var.concat(1, x, mean.expand(Shape.of(1, mean.shape().dim(1), t)),
                    std.expand(Shape.of(1, std.shape().dim(1), t)));
            att = aspConv.apply(aspTdnn.apply(att).relu().tanh()).softmax(2);
            Var wMean = x.mul(att).sum(2, true);
            Var wStd = att.mul(x.sub(wMean).square()).sum(2, true).clamp(1e-12f, Float.MAX_VALUE).sqrt();
            Var pooled = Var.concat(1, wMean, wStd);
            return fc.apply(pooled);
        }).to(DType.FLOAT32).value().toArray();
    }
}
