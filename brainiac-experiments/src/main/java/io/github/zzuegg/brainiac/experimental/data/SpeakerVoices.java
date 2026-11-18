package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A source-filter synthesiser of spoken digits with enough speaker factors that speakers are
 * separable, for speaker verification, diarisation, voice activity detection and speech
 * enhancement. The digit content follows the formant table of the wave 7 {@code TalkingDigits}
 * / {@code SpokenDigitRecognition} synthesiser (two vowel-like segments per digit,
 * {@code F1 = 300 + 110 d}, {@code F2 = 2500 - 170 ((7 d) mod 10)}), so recognisers built for it
 * still apply; that synthesiser had only two speaker factors (formant scale and rate).
 *
 * <p>Here each segment is a glottal pulse train (exponentially decaying pulses) at the
 * speaker's pitch, with a slow pitch contour, cycle-to-cycle jitter and breath noise, passed
 * through a one-pole spectral-tilt filter and a parallel bank of three two-pole formant
 * resonators. A {@link Voice} holds the speaker factors: pitch and pitch range, vocal-tract
 * scale and per-formant offsets, formant bandwidth, spectral tilt, breathiness, jitter and
 * speaking rate. Utterances come with their sample-level speech activity, and
 * {@link #conversation} mixes speakers' turns with silences, chosen overlap and background
 * noise (white, pink, or babble from other synthetic speakers) into a recording with
 * frame-level "who spoke when" labels.
 */
public final class SpeakerVoices {

    /** Samples per second. */
    public static final int RATE = 8000;
    /** Samples per frame of the activity labels (16 ms). */
    public static final int FRAME = 128;

    private SpeakerVoices() {
    }

    /**
     * A speaker.
     *
     * @param pitch mean fundamental frequency in Hz
     * @param pitchRange relative depth of the pitch contour
     * @param tractScale multiplies every formant (vocal tract length)
     * @param formantOffsets per-formant multipliers (three)
     * @param bandwidth formant bandwidth in Hz
     * @param tilt the spectral-tilt pole in [0, 1) (higher is darker)
     * @param breathiness the share of aspiration noise in the source
     * @param jitter relative cycle-to-cycle pitch perturbation
     * @param rate speaking rate (durations are divided by it)
     */
    public record Voice(double pitch, double pitchRange, double tractScale, double[] formantOffsets,
            double bandwidth, double tilt, double breathiness, double jitter, double rate) {
    }

    /**
     * Draws a speaker.
     *
     * @param random the source
     * @return the voice
     */
    public static Voice voice(SplittableRandom random) {
        double pitch = 85 * Math.pow(250 / 85.0, random.nextDouble());
        double[] off = new double[3];
        for (int i = 0; i < 3; i++) {
            off[i] = 1 + 0.16 * (random.nextDouble() - 0.5);
        }
        return new Voice(pitch, 0.05 + 0.2 * random.nextDouble(), 0.88 + 0.24 * random.nextDouble(), off,
                60 + 120 * random.nextDouble(), 0.2 + 0.7 * random.nextDouble(), 0.4 * random.nextDouble(),
                0.03 * random.nextDouble(), 0.8 + 0.45 * random.nextDouble());
    }

    /**
     * Draws a population of speakers.
     *
     * @param n the number of speakers
     * @param seed the seed
     * @return the voices
     */
    public static List<Voice> speakers(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Voice> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(voice(r));
        }
        return out;
    }

    /**
     * A clean utterance.
     *
     * @param wave the samples
     * @param active per sample whether the speaker is voicing (pauses are false)
     * @param digits the spoken digits
     */
    public record Utterance(float[] wave, boolean[] active, int[] digits) {
    }

    private static double[] formants(int digit, int segment, Voice v) {
        double f1 = 300 + 110 * digit;
        double f2 = 2500 - 170 * ((digit * 7) % 10);
        double a = segment == 0 ? f1 : f1 * 1.35;
        double b = segment == 0 ? f2 : f2 * 0.75 + 200 * (digit % 3);
        double c = 3000 + 150 * (digit % 4);
        double s = v.tractScale();
        double nyquist = RATE / 2.0 - 200;
        return new double[] {Math.min(nyquist, a * s * v.formantOffsets()[0]),
            Math.min(nyquist, b * s * v.formantOffsets()[1]), Math.min(nyquist, c * s * v.formantOffsets()[2])};
    }

    /**
     * Speaks a digit string (no added noise).
     *
     * @param digits the digits
     * @param v the voice
     * @param random varies durations, loudness, pitch contour and noise
     * @return the utterance
     */
    public static Utterance speak(int[] digits, Voice v, SplittableRandom random) {
        List<float[]> parts = new ArrayList<>();
        List<Boolean> voiced = new ArrayList<>();
        parts.add(new float[FRAME * (1 + random.nextInt(2))]);
        voiced.add(false);
        for (int d : digits) {
            double amp = 0.4 + 0.4 * random.nextDouble();
            for (int seg = 0; seg < 2; seg++) {
                int n = (int) ((FRAME * (2 + random.nextInt(2)) + random.nextInt(FRAME)) / v.rate());
                parts.add(segment(formants(d, seg, v), n, amp, v, random));
                voiced.add(true);
            }
            parts.add(new float[FRAME * (1 + random.nextInt(2))]);
            voiced.add(false);
        }
        int total = 0;
        for (float[] p : parts) {
            total += p.length;
        }
        float[] wave = new float[total];
        boolean[] active = new boolean[total];
        int o = 0;
        for (int i = 0; i < parts.size(); i++) {
            float[] p = parts.get(i);
            System.arraycopy(p, 0, wave, o, p.length);
            if (voiced.get(i)) {
                Arrays.fill(active, o, o + p.length, true);
            }
            o += p.length;
        }
        return new Utterance(wave, active, digits.clone());
    }

    private static float[] segment(double[] f, int n, double amp, Voice v, SplittableRandom random) {
        double[] source = new double[n];
        double phase = random.nextDouble();
        double contourPhase = random.nextDouble() * 2 * Math.PI;
        double pulse = 0;
        double period = RATE / v.pitch();
        double t = 0;
        double lp = 0;
        for (int s = 0; s < n; s++) {
            double f0 = v.pitch() * (1 + v.pitchRange() * Math.sin(2 * Math.PI * s / (double) n + contourPhase) * 0.5);
            phase += f0 / RATE;
            if (phase >= 1) {
                phase -= 1;
                pulse = 1;
                period = RATE / f0 * (1 + v.jitter() * random.nextGaussian());
                t = 0;
            }
            double glottal = pulse * Math.exp(-t / Math.max(1, 0.15 * period));
            t++;
            double x = (1 - v.breathiness()) * glottal + v.breathiness() * 0.5 * random.nextGaussian();
            lp = (1 - v.tilt()) * x + v.tilt() * lp;
            source[s] = lp;
        }
        double[] gains = {1.0, 0.6, 0.25};
        double[] out = new double[n];
        for (int k = 0; k < 3; k++) {
            double r = Math.exp(-Math.PI * v.bandwidth() * (1 + 0.5 * k) / RATE);
            double a1 = 2 * r * Math.cos(2 * Math.PI * f[k] / RATE);
            double a2 = -r * r;
            double y1 = 0;
            double y2 = 0;
            for (int s = 0; s < n; s++) {
                double y = a1 * y1 + a2 * y2 + (1 - r) * source[s];
                y2 = y1;
                y1 = y;
                out[s] += gains[k] * y;
            }
        }
        double power = 1e-12;
        for (double x : out) {
            power += x * x / n;
        }
        double scale = amp * 0.3 / Math.sqrt(power);
        float[] w = new float[n];
        for (int s = 0; s < n; s++) {
            double env = Math.min(1, Math.min(s, n - s) / 60.0);
            w[s] = (float) (out[s] * scale * env);
        }
        return w;
    }

    /** Background noise types. */
    public enum Noise {
        /** White Gaussian noise. */
        WHITE,
        /** Pink (1/f) noise. */
        PINK,
        /** Babble: several other synthetic speakers talking at once. */
        BABBLE
    }

    /**
     * Noise of a type, unit power.
     *
     * @param kind the type
     * @param n samples
     * @param random the source
     * @return the noise
     */
    public static float[] noise(Noise kind, int n, SplittableRandom random) {
        double[] x = new double[n];
        switch (kind) {
            case WHITE -> {
                for (int i = 0; i < n; i++) {
                    x[i] = random.nextGaussian();
                }
            }
            case PINK -> {
                // Paul Kellet's economy filter
                double b0 = 0;
                double b1 = 0;
                double b2 = 0;
                for (int i = 0; i < n; i++) {
                    double w = random.nextGaussian();
                    b0 = 0.99765 * b0 + w * 0.0990460;
                    b1 = 0.96300 * b1 + w * 0.2965164;
                    b2 = 0.57000 * b2 + w * 1.0526913;
                    x[i] = b0 + b1 + b2 + w * 0.1848;
                }
            }
            default -> {
                int talkers = 4;
                for (int k = 0; k < talkers; k++) {
                    Voice v = voice(random);
                    int o = -random.nextInt(FRAME * 4);
                    while (o < n) {
                        int[] digits = {random.nextInt(10), random.nextInt(10), random.nextInt(10)};
                        float[] w = speak(digits, v, random).wave();
                        for (int i = 0; i < w.length; i++) {
                            if (o + i >= 0 && o + i < n) {
                                x[o + i] += w[i];
                            }
                        }
                        o += w.length;
                    }
                }
            }
        }
        double p = 1e-12;
        for (double v : x) {
            p += v * v / n;
        }
        float[] out = new float[n];
        double s = 1 / Math.sqrt(p);
        for (int i = 0; i < n; i++) {
            out[i] = (float) (x[i] * s);
        }
        return out;
    }

    /**
     * Adds noise at a signal-to-noise ratio, the signal power measured over its active samples
     * (or all samples when {@code active} is {@code null}).
     *
     * @param clean the signal
     * @param active its activity, or {@code null}
     * @param kind the noise type
     * @param snrDb the SNR in dB
     * @param random the source
     * @return {@code {noisy, noise}}
     */
    public static float[][] addNoise(float[] clean, boolean[] active, Noise kind, double snrDb, SplittableRandom random) {
        double p = 1e-12;
        int count = 0;
        for (int i = 0; i < clean.length; i++) {
            if (active == null || active[i]) {
                p += clean[i] * clean[i];
                count++;
            }
        }
        p /= Math.max(1, count);
        float[] nz = noise(kind, clean.length, random);
        double g = Math.sqrt(p / Math.pow(10, snrDb / 10));
        float[] noisy = new float[clean.length];
        for (int i = 0; i < clean.length; i++) {
            nz[i] *= (float) g;
            noisy[i] = clean[i] + nz[i];
        }
        return new float[][] {noisy, nz};
    }

    /**
     * Frame-level activity: a frame is active when at least a quarter of its samples are.
     *
     * @param active per-sample activity
     * @return per-frame activity
     */
    public static boolean[] frames(boolean[] active) {
        int f = active.length / FRAME;
        boolean[] out = new boolean[f];
        for (int t = 0; t < f; t++) {
            int c = 0;
            for (int i = t * FRAME; i < (t + 1) * FRAME; i++) {
                if (active[i]) {
                    c++;
                }
            }
            out[t] = c * 4 >= FRAME;
        }
        return out;
    }

    /**
     * A multi-speaker recording.
     *
     * @param wave the noisy mixture
     * @param clean the noise-free mixture
     * @param activity {@code [speaker][frame]} who is speaking
     * @param speakers the voices, in the order of {@code activity}
     * @param overlap the fraction of speech frames with two or more active speakers
     */
    public record Conversation(float[] wave, float[] clean, boolean[][] activity, List<Voice> speakers,
            double overlap) {

        /** @return the number of frames */
        public int frames() {
            return activity[0].length;
        }

        /** @return per frame whether anyone speaks */
        public boolean[] speech() {
            boolean[] s = new boolean[frames()];
            for (boolean[] a : activity) {
                for (int t = 0; t < s.length; t++) {
                    s[t] |= a[t];
                }
            }
            return s;
        }
    }

    /**
     * Builds a conversation: speakers take turns (each turn one to three digits, the speaker
     * different from the previous one), separated by silences of 0.1-0.4 s; with probability
     * {@code overlapChance} a turn instead starts while the previous one is still running, at
     * 40-80 percent of its length.
     *
     * @param voices the speakers
     * @param turns the number of turns
     * @param overlapChance the chance that a turn overlaps the previous one
     * @param noise the background noise type
     * @param snrDb the SNR in dB
     * @param random the source
     * @return the conversation
     */
    public static Conversation conversation(List<Voice> voices, int turns, double overlapChance, Noise noise,
            double snrDb, SplittableRandom random) {
        record Turn(int speaker, int start, Utterance u) {
        }
        List<Turn> list = new ArrayList<>();
        int cursor = FRAME * 4;
        int prevSpeaker = -1;
        int prevStart = 0;
        int prevLen = 0;
        for (int t = 0; t < turns; t++) {
            int s;
            do {
                s = random.nextInt(voices.size());
            } while (s == prevSpeaker && voices.size() > 1);
            int len = 1 + random.nextInt(3);
            int[] digits = new int[len];
            for (int i = 0; i < len; i++) {
                digits[i] = random.nextInt(10);
            }
            Utterance u = speak(digits, voices.get(s), random);
            int start = cursor;
            if (t > 0 && random.nextDouble() < overlapChance) {
                start = prevStart + (int) (prevLen * (0.4 + 0.4 * random.nextDouble()));
            }
            list.add(new Turn(s, start, u));
            cursor = Math.max(cursor, start + u.wave().length) + (int) (RATE * (0.1 + 0.3 * random.nextDouble()));
            prevSpeaker = s;
            prevStart = start;
            prevLen = u.wave().length;
        }
        int total = ((cursor + FRAME * 4) / FRAME + 1) * FRAME;
        float[] clean = new float[total];
        boolean[][] sampleActive = new boolean[voices.size()][total];
        for (Turn t : list) {
            float[] w = t.u().wave();
            for (int i = 0; i < w.length; i++) {
                clean[t.start() + i] += w[i];
                sampleActive[t.speaker()][t.start() + i] |= t.u().active()[i];
            }
        }
        boolean[][] activity = new boolean[voices.size()][];
        boolean[] any = new boolean[total];
        for (int s = 0; s < voices.size(); s++) {
            activity[s] = frames(sampleActive[s]);
            for (int i = 0; i < total; i++) {
                any[i] |= sampleActive[s][i];
            }
        }
        int speechFrames = 0;
        int overlapped = 0;
        for (int f = 0; f < activity[0].length; f++) {
            int c = 0;
            for (boolean[] a : activity) {
                if (a[f]) {
                    c++;
                }
            }
            if (c > 0) {
                speechFrames++;
            }
            if (c > 1) {
                overlapped++;
            }
        }
        float[] wave = addNoise(clean, any, noise, snrDb, random)[0];
        return new Conversation(wave, clean, activity, List.copyOf(voices), overlapped / (double) Math.max(1,
                speechFrames));
    }
}
