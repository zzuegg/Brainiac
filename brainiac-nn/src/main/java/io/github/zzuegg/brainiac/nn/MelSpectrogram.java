package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A log-mel spectrogram as a layer: raw audio {@code [samples]} in, {@code [mels, frames]}
 * out, channels first so convolutions and transformers can follow. It has no parameters
 * and passes gradients, so it can sit at the front of any model (a speech classifier, a
 * keyword spotter, an audio encoder).
 *
 * <pre>{@code
 * Model kws = Model.sequential()
 *         .input(16_000)                         // one second at 16 kHz
 *         .melSpectrogram(16_000, 400, 160, 40)  // [40, 101]
 *         .conv1d(64, 3, Activation.GELU)
 *         .globalAvgPool()
 *         .dense(12, Activation.SOFTMAX)
 *         .build();
 * }</pre>
 */
public final class MelSpectrogram implements Layer {

    /** How the mel energies are compressed. */
    public enum Compression {
        /** {@code ln(mel + 1e-6)}, with {@code samples / hop + 1} frames. */
        LOG,
        /**
         * The mel of the magnitude (not the power) spectrum, {@code ln(max(mel, 1e-5))}, with
         * {@code samples / hop + 1} frames: the features of Vocos and F5-TTS.
         */
        LOG_AMPLITUDE,
        /**
         * Whisper's: {@code log10} clamped at {@code 1e-10}, floored at 8 below the maximum,
         * {@code (x + 4) / 4}, with the last frame dropped ({@code samples / hop} frames).
         */
        WHISPER
    }

    private final int sampleRate;
    private final int nFft;
    private final int hop;
    private final int mels;
    private final Signal.MelScale scale;
    private final boolean slaneyNorm;
    private final Compression compression;

    /**
     * Creates the layer.
     *
     * @param sampleRate the audio's sample rate
     * @param nFft the STFT frame length
     * @param hop the step between frames
     * @param mels the number of mel bands
     * @param scale the mel scale
     * @param slaneyNorm whether filters are area-normalized
     * @param compression how energies are compressed
     */
    public MelSpectrogram(int sampleRate, int nFft, int hop, int mels, Signal.MelScale scale, boolean slaneyNorm,
            Compression compression) {
        if (sampleRate <= 0 || nFft < 2 || hop < 1 || mels < 1) {
            throw new IllegalArgumentException("mel spectrogram needs a positive sample rate, nFft >= 2, hop >= 1"
                    + " and mels >= 1");
        }
        this.sampleRate = sampleRate;
        this.nFft = nFft;
        this.hop = hop;
        this.mels = mels;
        this.scale = scale;
        this.slaneyNorm = slaneyNorm;
        this.compression = compression;
    }

    /**
     * A natural-log mel spectrogram with HTK mels (as {@link Signal#logMel}).
     *
     * @param sampleRate the sample rate
     * @param nFft the frame length
     * @param hop the step between frames
     * @param mels the number of bands
     * @return the layer
     */
    public static MelSpectrogram of(int sampleRate, int nFft, int hop, int mels) {
        return new MelSpectrogram(sampleRate, nFft, hop, mels, Signal.MelScale.HTK, false, Compression.LOG);
    }

    /**
     * The log-amplitude mel features of Vocos and F5-TTS (torchaudio's {@code MelSpectrogram}
     * with {@code power=1}, HTK mels, no filter norm, then {@code ln(max(x, 1e-5))}).
     *
     * @param sampleRate the sample rate, 24000 for both
     * @param nFft the frame length, 1024
     * @param hop the step between frames, 256
     * @param mels the bands, 100
     * @return the layer
     */
    public static MelSpectrogram vocos(int sampleRate, int nFft, int hop, int mels) {
        return new MelSpectrogram(sampleRate, nFft, hop, mels, Signal.MelScale.HTK, false, Compression.LOG_AMPLITUDE);
    }

    /**
     * Whisper's front end for 16 kHz audio (see {@link Signal#whisperLogMel}).
     *
     * @param mels 80, or 128 for large-v3
     * @return the layer
     */
    public static MelSpectrogram whisper(int mels) {
        return new MelSpectrogram(16_000, 400, 160, mels, Signal.MelScale.SLANEY, true, Compression.WHISPER);
    }

    /**
     * Returns the number of mel bands.
     *
     * @return the bands
     */
    public int mels() {
        return mels;
    }

    @Override
    public Var forward(Var input) {
        if (compression == Compression.WHISPER && sampleRate == 16_000 && nFft == 400 && hop == 160
                && scale == Signal.MelScale.SLANEY && slaneyNorm) {
            return Signal.whisperLogMel(input, mels);
        }
        Tensor fb = Signal.melFilterbank(sampleRate, nFft, mels, 0, sampleRate / 2.0, scale, slaneyNorm)
                .to(input.value().device()).to(input.dtype());
        Var p = Signal.power(Signal.stft(input, nFft, hop));
        if (compression == Compression.LOG_AMPLITUDE) {
            p = p.add(1e-20f).sqrt();
        }
        int rank = p.shape().rank();
        if (compression == Compression.WHISPER) {
            p = p.slice(rank - 2, 0, p.shape().dim(rank - 2) - 1);
        }
        Var mel = p.matmul(Var.of(fb));
        Var out;
        if (compression == Compression.LOG) {
            out = mel.add(1e-6f).log();
        } else if (compression == Compression.LOG_AMPLITUDE) {
            out = mel.maximum(1e-5f).log();
        } else {
            Var log = mel.maximum(1e-10f).log().mul((float) (1 / Math.log(10)));
            Var max = log.max(rank - 1, true).max(rank - 2, true);
            out = log.maximum(max.sub(8f)).add(4f).mul(0.25f);
        }
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return out.permute(axes);
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] d = input.toArray();
        int samples = d[d.length - 1];
        int frames = compression == Compression.WHISPER ? samples / hop : samples / hop + 1;
        int[] out = java.util.Arrays.copyOf(d, d.length + 1);
        out[d.length - 1] = mels;
        out[d.length] = frames;
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return "MelSpectrogram[" + mels + " mels, " + sampleRate + " Hz, nFft " + nFft + ", hop " + hop + ", "
                + compression + "]";
    }

    static Config write(MelSpectrogram m) {
        return Config.builder().put("sampleRate", m.sampleRate).put("nFft", m.nFft).put("hop", m.hop)
                .put("mels", m.mels).put("scale", m.scale.name()).put("slaneyNorm", m.slaneyNorm)
                .put("compression", m.compression.name()).build();
    }

    static MelSpectrogram read(Config c) {
        return new MelSpectrogram(c.getInt("sampleRate"), c.getInt("nFft"), c.getInt("hop"), c.getInt("mels"),
                Signal.MelScale.valueOf(c.getString("scale")), c.getBoolean("slaneyNorm"),
                Compression.valueOf(c.getString("compression")));
    }
}
