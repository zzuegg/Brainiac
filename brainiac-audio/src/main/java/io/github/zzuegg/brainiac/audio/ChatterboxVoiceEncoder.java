package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.Lin;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Resemble's voice encoder, which gives Chatterbox's T3 the speaker: 40-band mels of 16 kHz
 * speech (silence trimmed), cut into overlapping 160-frame windows, each through a 3-layer LSTM
 * to a normalized embedding, averaged.
 */
final class ChatterboxVoiceEncoder {

    static final int SAMPLE_RATE = 16_000;
    private static final int N_FFT = 400;
    private static final int HOP = 160;
    private static final int WINDOW = 160;
    private static final double RATE = 1.3;
    private static final double MIN_COVERAGE = 0.8;

    private final Device device;
    private final DType dtype;
    private final List<Recurrent> lstm = new ArrayList<>();
    private final Lin proj;
    private final Tensor melBasis;
    private final int mels;

    ChatterboxVoiceEncoder(Weights w) {
        device = w.device();
        dtype = w.dtype();
        for (int l = 0; w.contains("lstm.weight_ih_l" + l); l++) {
            lstm.add(w.lstm("lstm", l));
        }
        proj = ChatterboxNets.lin(w, "proj");
        mels = w.shape("lstm.weight_ih_l0").dim(1);
        melBasis = Signal.melFilterbank(SAMPLE_RATE, N_FFT, mels, 0, 8000, Signal.MelScale.SLANEY, true).to(device);
    }

    /**
     * Where speech is: librosa's {@code effects.trim} with {@code top_db = 20} (frames of 2048
     * samples every 512 whose power is within 20 dB of the loudest).
     *
     * @return {@code {start, end}} sample indices
     */
    static int[] trim(float[] y) {
        int frameLength = 2048;
        int hop = 512;
        int frames = 1 + y.length / hop;
        double[] power = new double[frames];
        double max = 0;
        for (int f = 0; f < frames; f++) {
            int start = f * hop - frameLength / 2;
            double sum = 0;
            for (int i = 0; i < frameLength; i++) {
                int j = start + i;
                if (j >= 0 && j < y.length) {
                    sum += (double) y[j] * y[j];
                }
            }
            power[f] = sum / frameLength;
            max = Math.max(max, power[f]);
        }
        double ref = 10 * Math.log10(Math.max(1e-10, max));
        int first = -1;
        int last = -1;
        for (int f = 0; f < frames; f++) {
            double db = 10 * Math.log10(Math.max(1e-10, power[f])) - ref;
            if (db > -20) {
                if (first < 0) {
                    first = f;
                }
                last = f;
            }
        }
        if (first < 0) {
            return new int[] {0, 0};
        }
        return new int[] {first * hop, Math.min(y.length, (last + 1) * hop)};
    }

    /** The power mel spectrogram, {@code [frames, mels]}. */
    Var mel(float[] audio) {
        Var x = Var.of(Tensor.of(Shape.of(audio.length), audio, device));
        Var power = Signal.stft(x, N_FFT, HOP).square().sum(2, false);    // [frames, 201]
        return power.matmul(Var.of(melBasis));
    }

    /**
     * The speaker embedding of some speech.
     *
     * @param audio 16 kHz samples
     * @return a unit vector (256 values in the real model)
     */
    float[] embed(float[] audio) {
        int[] span = trim(audio);
        float[] y = java.util.Arrays.copyOfRange(audio, span[0], span[1]);
        if (y.length <= N_FFT / 2) {
            throw new IllegalArgumentException("the recording holds no speech loud enough to read a voice from");
        }
        return Var.noGrad(() -> {
            float[] m = mel(y).to(DType.FLOAT32).value().toArray();
            int frames = m.length / mels;
            int step = (int) Math.round((SAMPLE_RATE / RATE) / HOP);
            int wins = Math.max(frames - WINDOW + step, 0) / step;
            int rem = Math.max(frames - WINDOW + step, 0) % step;
            if (wins == 0 || (rem + (WINDOW - step)) / (double) WINDOW >= MIN_COVERAGE) {
                wins++;
            }
            int target = WINDOW + step * (wins - 1);
            float[] padded = java.util.Arrays.copyOf(m, Math.max(target, frames) * mels);
            float[] parts = new float[wins * WINDOW * mels];
            for (int i = 0; i < wins; i++) {
                System.arraycopy(padded, i * step * mels, parts, i * WINDOW * mels, WINDOW * mels);
            }
            Var x = Var.of(Tensor.of(Shape.of(wins, WINDOW, mels), parts, device).to(dtype));
            for (Recurrent r : lstm) {
                x = r.forward(x);
            }
            Var last = x.slice(1, WINDOW - 1, WINDOW).reshape(wins, x.shape().dim(2));
            Var e = proj.apply(last).relu();
            e = e.div(e.square().sum(1, true).sqrt());
            Var mean = e.mean(0, false);
            mean = mean.div(mean.square().sum().sqrt());
            return mean.to(DType.FLOAT32).value().toArray();
        });
    }
}
