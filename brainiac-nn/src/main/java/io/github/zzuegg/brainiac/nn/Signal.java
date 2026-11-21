package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Audio front ends: short-time Fourier transforms and their inverse, mel filterbanks and
 * log-mel spectrograms, SpecAugment masking and resampling. The transforms are {@link Var}
 * operations, so they run on any device and pass gradients (for spectral losses and
 * vocoders).
 */
public final class Signal {

    private Signal() {
    }

    /**
     * The periodic Hann window, as STFTs use.
     *
     * @param n its length
     * @param device where it goes
     * @return {@code [n]}
     */
    public static Tensor hann(int n, Device device) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
        }
        return Tensor.of(Shape.of(n), w, device);
    }

    /**
     * The short-time Fourier transform of signals along their last axis, with a Hann window
     * and the signal reflect-padded by {@code nFft / 2} on both sides so frame {@code t} is
     * centred on sample {@code t * hop} (as librosa and PyTorch with {@code center=true}).
     *
     * @param x signals {@code [..., samples]}
     * @param nFft the frame length
     * @param hop the step between frames
     * @return complex values {@code [..., frames, nFft / 2 + 1, 2]}, frames =
     *     {@code samples / hop + 1}
     */
    public static Var stft(Var x, int nFft, int hop) {
        int rank = x.shape().rank();
        int samples = x.shape().dim(-1);
        if (samples <= nFft / 2) {
            throw new IllegalArgumentException("a signal of " + samples + " samples is too short for nFft " + nFft);
        }
        Var padded = Functional.padAxis(x, rank - 1, nFft / 2, nFft / 2, PadMode.REFLECT, 0);
        int frames = samples / hop + 1;
        float[] idx = new float[frames * nFft];
        for (int f = 0; f < frames; f++) {
            for (int j = 0; j < nFft; j++) {
                idx[f * nFft + j] = f * hop + j;
            }
        }
        Device device = x.value().device();
        Var framed = padded.gather(rank - 1, Tensor.of(Shape.of(frames, nFft), idx, device));
        return framed.mul(Var.of(hann(nFft, device).to(x.dtype()))).rfft();
    }

    /**
     * The inverse of {@link #stft}: windowed overlap-add of the frames' inverse transforms,
     * divided by the summed squared window.
     *
     * @param spec complex values {@code [..., frames, nFft / 2 + 1, 2]}
     * @param nFft the frame length
     * @param hop the step between frames
     * @param samples the length of the signal
     * @return {@code [..., samples]}
     */
    public static Var istft(Var spec, int nFft, int hop, int samples) {
        return overlapAdd(spec, nFft, hop, nFft / 2, samples);
    }

    /**
     * The inverse STFT with "same" padding, as Vocos decodes: frames overlap-added without
     * centring, then {@code (nFft - hop) / 2} samples trimmed from both ends, so {@code frames}
     * frames give exactly {@code frames * hop} samples.
     *
     * @param spec complex values {@code [..., frames, nFft / 2 + 1, 2]}
     * @param nFft the frame length
     * @param hop the step between frames
     * @return {@code [..., frames * hop]}
     */
    public static Var istftSame(Var spec, int nFft, int hop) {
        int frames = spec.shape().dim(-3);
        return overlapAdd(spec, nFft, hop, (nFft - hop) / 2, frames * hop);
    }

    private static Var overlapAdd(Var spec, int nFft, int hop, int start, int samples) {
        Shape s = spec.shape();
        int rank = s.rank();
        int frames = s.dim(rank - 3);
        Device device = spec.value().device();
        Tensor window = hann(nFft, device).to(spec.dtype());
        Var framesOut = spec.irfft(nFft).mul(Var.of(window));   // [..., frames, nFft]
        int full = (frames - 1) * hop + nFft;
        float[] idx = new float[frames * nFft];
        double[] norm = new double[full];
        float[] w = hann(nFft, Device.cpu()).toArray();
        for (int f = 0; f < frames; f++) {
            for (int j = 0; j < nFft; j++) {
                idx[f * nFft + j] = f * hop + j;
                norm[f * hop + j] += (double) w[j] * w[j];
            }
        }
        int[] flat = new int[rank - 2];
        System.arraycopy(s.toArray(), 0, flat, 0, rank - 3);
        flat[rank - 3] = frames * nFft;
        Var summed = framesOut.reshape(flat).scatterAdd(rank - 3, Tensor.of(Shape.of(frames * nFft), idx, device), full);
        float[] inv = new float[full];
        for (int i = 0; i < full; i++) {
            inv[i] = norm[i] > 1e-10 ? (float) (1 / norm[i]) : 0;
        }
        Var signal = summed.mul(Var.of(Tensor.of(Shape.of(full), inv, device).to(spec.dtype())));
        return signal.slice(rank - 3, start, Math.min(full, start + samples));
    }

    /**
     * A mel filterbank (HTK mel scale, triangular filters with unit peak).
     *
     * @param sampleRate the sample rate
     * @param nFft the frame length of the STFT
     * @param mels the number of filters
     * @param minHz the lowest frequency
     * @param maxHz the highest frequency, at most half the sample rate
     * @return {@code [nFft / 2 + 1, mels]}: power spectra times this give mel energies
     */
    public static Tensor melFilterbank(int sampleRate, int nFft, int mels, double minHz, double maxHz) {
        int bins = nFft / 2 + 1;
        double lo = hzToMel(minHz);
        double hi = hzToMel(maxHz);
        double[] edges = new double[mels + 2];
        for (int i = 0; i < edges.length; i++) {
            edges[i] = melToHz(lo + (hi - lo) * i / (mels + 1));
        }
        float[] fb = new float[bins * mels];
        for (int b = 0; b < bins; b++) {
            double hz = (double) b * sampleRate / nFft;
            for (int m = 0; m < mels; m++) {
                double left = edges[m];
                double centre = edges[m + 1];
                double right = edges[m + 2];
                double v = hz <= left || hz >= right ? 0 : hz <= centre ? (hz - left) / (centre - left)
                        : (right - hz) / (right - centre);
                fb[b * mels + m] = (float) v;
            }
        }
        return Tensor.of(Shape.of(bins, mels), fb);
    }

    private static double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    private static double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }

    /** How hertz map to mels. */
    public enum MelScale {
        /** {@code 2595 log10(1 + hz / 700)}, as HTK and torchaudio's default. */
        HTK,
        /** Slaney's: linear below 1 kHz, logarithmic above; librosa's default, and Whisper's. */
        SLANEY
    }

    /**
     * A mel filterbank of either mel scale, optionally with Slaney's area normalization
     * (each filter divided by its width in hertz, so filters carry equal energy), as
     * librosa's {@code filters.mel}.
     *
     * @param sampleRate the sample rate
     * @param nFft the frame length of the STFT
     * @param mels the number of filters
     * @param minHz the lowest frequency
     * @param maxHz the highest frequency, at most half the sample rate
     * @param scale the mel scale
     * @param slaneyNorm whether to normalize each filter's area
     * @return {@code [nFft / 2 + 1, mels]}: power spectra times this give mel energies
     */
    public static Tensor melFilterbank(int sampleRate, int nFft, int mels, double minHz, double maxHz,
            MelScale scale, boolean slaneyNorm) {
        int bins = nFft / 2 + 1;
        java.util.function.DoubleUnaryOperator toMel = scale == MelScale.HTK ? Signal::hzToMel : Signal::slaneyMel;
        java.util.function.DoubleUnaryOperator toHz = scale == MelScale.HTK ? Signal::melToHz : Signal::slaneyHz;
        double lo = toMel.applyAsDouble(minHz);
        double hi = toMel.applyAsDouble(maxHz);
        double[] hz = new double[mels + 2];
        for (int i = 0; i < hz.length; i++) {
            hz[i] = toHz.applyAsDouble(lo + (hi - lo) * i / (mels + 1));
        }
        float[] fb = new float[bins * mels];
        for (int m = 0; m < mels; m++) {
            double norm = slaneyNorm ? 2 / (hz[m + 2] - hz[m]) : 1;
            for (int b = 0; b < bins; b++) {
                double f = (double) b * sampleRate / nFft;
                double rise = (f - hz[m]) / (hz[m + 1] - hz[m]);
                double fall = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
                fb[b * mels + m] = (float) (Math.max(0, Math.min(rise, fall)) * norm);
            }
        }
        return Tensor.of(Shape.of(bins, mels), fb);
    }

    private static final double SLANEY_F_SP = 200.0 / 3;
    private static final double SLANEY_MIN_LOG_HZ = 1000;
    private static final double SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
    private static final double SLANEY_LOG_STEP = Math.log(6.4) / 27;

    private static double slaneyMel(double hz) {
        return hz >= SLANEY_MIN_LOG_HZ ? SLANEY_MIN_LOG_MEL + Math.log(hz / SLANEY_MIN_LOG_HZ) / SLANEY_LOG_STEP
                : hz / SLANEY_F_SP;
    }

    private static double slaneyHz(double mel) {
        return mel >= SLANEY_MIN_LOG_MEL ? SLANEY_MIN_LOG_HZ * Math.exp(SLANEY_LOG_STEP * (mel - SLANEY_MIN_LOG_MEL))
                : SLANEY_F_SP * mel;
    }

    /**
     * Whisper's log-mel spectrogram, as OpenAI's {@code log_mel_spectrogram}: a 400-sample
     * Hann STFT every 160 samples of 16 kHz audio (the last frame dropped), the power
     * through Slaney mel filters, {@code log10} clamped at {@code 1e-10}, floored at 8
     * below each signal's maximum, then {@code (x + 4) / 4}.
     *
     * @param x 16 kHz signals {@code [..., samples]}
     * @param mels the number of mel bands (80, or 128 for large-v3)
     * @return {@code [..., mels, samples / 160]}, channels first as convolutions take them
     */
    public static Var whisperLogMel(Var x, int mels) {
        int frames = x.shape().dim(-1) / 160;
        Tensor fb = melFilterbank(16_000, 400, mels, 0, 8_000, MelScale.SLANEY, true)
                .to(x.value().device()).to(x.dtype());
        Var p = power(stft(x, 400, 160));   // [..., frames + 1, 201]
        int rank = p.shape().rank();
        Var mel = p.slice(rank - 2, 0, frames).matmul(Var.of(fb));   // [..., frames, mels]
        Var log = mel.maximum(1e-10f).log().mul((float) (1 / Math.log(10)));
        Var max = log.max(rank - 1, true).max(rank - 2, true);
        return log.maximum(max.sub(8f)).add(4f).mul(0.25f).permute(swapLast(rank));
    }

    private static int[] swapLast(int rank) {
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return axes;
    }

    /**
     * The power of complex values: {@code re^2 + im^2}.
     *
     * @param spec {@code [..., 2]}
     * @return {@code [...]}
     */
    public static Var power(Var spec) {
        return spec.square().sum(-1, false);
    }

    /**
     * A log-mel spectrogram, the usual input of speech models: {@code log(power(stft) x
     * filterbank + 1e-6)}.
     *
     * @param x signals {@code [..., samples]}
     * @param sampleRate their sample rate
     * @param nFft the frame length
     * @param hop the step between frames
     * @param mels the number of mel bands
     * @return {@code [..., frames, mels]}
     */
    public static Var logMel(Var x, int sampleRate, int nFft, int hop, int mels) {
        Tensor fb = melFilterbank(sampleRate, nFft, mels, 0, sampleRate / 2.0).to(x.value().device()).to(x.dtype());
        return power(stft(x, nFft, hop)).matmul(Var.of(fb)).add(1e-6f).log();
    }

    /**
     * SpecAugment (Park et al., 2019): zeroes random bands of frequencies and spans of time,
     * a training-time augmentation for speech.
     *
     * @param spec spectrograms {@code [..., frames, bins]}
     * @param freqMasks how many frequency bands to zero
     * @param maxFreq the widest band
     * @param timeMasks how many time spans to zero
     * @param maxTime the longest span
     * @param random the source of the masks
     * @return the masked spectrograms
     */
    public static Var specAugment(Var spec, int freqMasks, int maxFreq, int timeMasks, int maxTime,
            RandomGenerator random) {
        int frames = spec.shape().dim(-2);
        int bins = spec.shape().dim(-1);
        float[] keep = new float[frames * bins];
        java.util.Arrays.fill(keep, 1);
        for (int i = 0; i < freqMasks; i++) {
            int w = random.nextInt(Math.min(maxFreq, bins) + 1);
            int f0 = random.nextInt(bins - w + 1);
            for (int t = 0; t < frames; t++) {
                for (int f = f0; f < f0 + w; f++) {
                    keep[t * bins + f] = 0;
                }
            }
        }
        for (int i = 0; i < timeMasks; i++) {
            int w = random.nextInt(Math.min(maxTime, frames) + 1);
            int t0 = random.nextInt(frames - w + 1);
            for (int t = t0; t < t0 + w; t++) {
                for (int f = 0; f < bins; f++) {
                    keep[t * bins + f] = 0;
                }
            }
        }
        return spec.mask(Tensor.of(Shape.of(frames, bins), keep, spec.value().device()));
    }

    /**
     * Changes the sample rate with windowed-sinc interpolation (a Hann-windowed low-pass at
     * the lower of the two Nyquist frequencies).
     *
     * @param x the samples
     * @param from their rate
     * @param to the new rate
     * @return the resampled signal, {@code round(x.length * to / from)} samples
     */
    public static float[] resample(float[] x, int from, int to) {
        if (from == to) {
            return x.clone();
        }
        int n = (int) Math.round((double) x.length * to / from);
        float[] out = new float[n];
        double ratio = (double) to / from;
        double cutoff = Math.min(1, ratio);   // relative to the input's Nyquist
        int half = (int) Math.ceil(16 / cutoff);
        for (int i = 0; i < n; i++) {
            double centre = i / ratio;
            int j0 = (int) Math.floor(centre) - half + 1;
            double sum = 0;
            for (int j = j0; j < j0 + 2 * half; j++) {
                if (j < 0 || j >= x.length) {
                    continue;
                }
                double d = centre - j;
                double sinc = d == 0 ? 1 : Math.sin(Math.PI * cutoff * d) / (Math.PI * cutoff * d);
                double win = Math.abs(d) >= half ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * d / half);
                sum += x[j] * cutoff * sinc * win;
            }
            out[i] = (float) sum;
        }
        return out;
    }

    /**
     * Speeds speech up or slows it down without changing its pitch: WSOLA (Verhelst and
     * Roelands, 1993). Hann windows of 30 ms overlap by half in the output; each is taken from
     * near where the rate puts it in the input, shifted by up to 10 ms to line up best (by
     * cross-correlation) with the natural continuation of the window before it.
     *
     * @param x the samples
     * @param rate their sample rate
     * @param speed how much faster, such as 1.25; 1 returns a copy
     * @return about {@code x.length / speed} samples
     */
    public static float[] timeStretch(float[] x, int rate, double speed) {
        if (!(speed > 0)) {
            throw new IllegalArgumentException("speed must be positive, got " + speed);
        }
        if (speed == 1 || x.length == 0) {
            return x.clone();
        }
        int n = Math.max(4, (int) Math.round(rate * 0.03)) & ~1;
        int hop = n / 2;
        int tolerance = Math.max(1, (int) Math.round(rate * 0.01));
        int stride = Math.max(1, rate / 8000);                    // correlate at about 8 kHz
        int out = (int) Math.round(x.length / speed);
        float[] y = new float[out + n];
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);     // periodic: halves sum to 1
        }
        int previous = 0;
        for (int k = 0; (long) k * hop < out; k++) {
            int nominal = (int) Math.round(k * hop * speed);
            int at = nominal;
            if (k > 0) {
                int natural = previous + hop;
                double best = Double.NEGATIVE_INFINITY;
                for (int d = -tolerance; d <= tolerance; d++) {
                    int c = nominal + d;
                    if (c < 0) {
                        continue;
                    }
                    double sum = 0;
                    for (int i = 0; i < n; i += stride) {
                        sum += at(x, c + i) * at(x, natural + i);
                    }
                    if (sum > best) {
                        best = sum;
                        at = c;
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                y[k * hop + i] += (float) (w[i] * at(x, at + i));
            }
            previous = at;
        }
        // the first half window has no partner before it
        for (int i = 0; i < Math.min(hop, out); i++) {
            y[i] = x[Math.min(i, x.length - 1)];
        }
        return java.util.Arrays.copyOf(y, out);
    }

    private static float at(float[] x, int i) {
        return i >= 0 && i < x.length ? x[i] : 0;
    }
}
