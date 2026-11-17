package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Whisper's input: the log-mel spectrogram of 16 kHz audio, computed as OpenAI's
 * {@code whisper.audio.log_mel_spectrogram}. A 400-sample Hann window every 160 samples
 * (100 frames a second), the power spectrum through Slaney-normalized mel filters (80 or
 * 128 bands), {@code log10}, clamped to 8 below the maximum, then {@code (x + 4) / 4}.
 */
public final class WhisperFeatures {

    /** The sample rate Whisper expects. */
    public static final int SAMPLE_RATE = 16_000;
    /** The STFT frame length. */
    public static final int N_FFT = 400;
    /** The step between frames. */
    public static final int HOP = 160;
    /** The samples of one 30-second window. */
    public static final int WINDOW_SAMPLES = 30 * SAMPLE_RATE;
    /** The frames of one 30-second window. */
    public static final int WINDOW_FRAMES = WINDOW_SAMPLES / HOP;

    private WhisperFeatures() {
    }

    /**
     * The log-mel spectrogram of audio, without padding: {@code samples / 160} frames.
     * It is {@link Signal#whisperLogMel} (also a layer, {@link
     * io.github.zzuegg.brainiac.nn.MelSpectrogram#whisper}) on the CPU.
     *
     * @param samples 16 kHz mono samples
     * @param mels the number of mel bands (80, or 128 for large-v3)
     * @return {@code [mels][frames]}
     */
    public static float[][] logMel(float[] samples, int mels) {
        int frames = samples.length / HOP;
        float[][] out = new float[mels][frames];
        if (frames == 0) {
            return out;
        }
        Tensor x = Tensor.of(Shape.of(samples.length), samples, Device.cpu());
        float[] flat = Signal.whisperLogMel(Var.of(x), mels).value().toArray();
        for (int m = 0; m < mels; m++) {
            System.arraycopy(flat, m * frames, out[m], 0, frames);
        }
        return out;
    }

    /**
     * The spectrogram of audio for transcription: the audio followed by 30 seconds of
     * silence, as Whisper's {@code transcribe} computes it, so the last window is full.
     *
     * @param samples 16 kHz mono samples
     * @param mels the number of mel bands
     * @return {@code [mels][samples / 160 + 3000]}
     */
    public static float[][] padded(float[] samples, int mels) {
        float[] p = java.util.Arrays.copyOf(samples, samples.length + WINDOW_SAMPLES);
        return logMel(p, mels);
    }

    /**
     * One 30-second window of a spectrogram, zero-padded or cut to 3000 frames.
     *
     * @param mel {@code [mels][frames]}
     * @param start the first frame
     * @return {@code [mels][3000]}
     */
    public static float[][] window(float[][] mel, int start) {
        float[][] out = new float[mel.length][WINDOW_FRAMES];
        for (int m = 0; m < mel.length; m++) {
            int n = Math.max(0, Math.min(WINDOW_FRAMES, mel[m].length - start));
            System.arraycopy(mel[m], start, out[m], 0, n);
        }
        return out;
    }
}
