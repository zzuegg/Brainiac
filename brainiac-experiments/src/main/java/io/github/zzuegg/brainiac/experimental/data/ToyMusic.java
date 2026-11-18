package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic music with exact labels, at 8 kHz: monophonic melodies with their F0 track,
 * speech-like voiced utterances with a known F0 (timbre from a {@link SpeakerVoices.Voice}),
 * polyphonic songs with a chord progression, a bass line, passing tones and drums at a tempo
 * of 60-180 BPM (optionally changing, with expressive timing jitter), and sound-event scenes
 * that overlap 1-4 event types. Every clip carries its F0 and voicing per 10 ms frame, its
 * note and drum onsets, its beats, its chord per frame and its active events per frame.
 *
 * <p>Instruments are additive harmonic tones (a per-timbre harmonic profile and envelope:
 * plucked, sustained or breathy) and simple drums (a pitch-swept kick, a noisy snare, a
 * high-passed hi-hat). Degradations are separate: white noise at an SNR, an exponentially
 * decaying noise reverb, a one-pole low-pass, and a speed change by resampling.
 *
 * <p>{@code HarmonicNoiseSynth} renders the same harmonic-plus-noise model, but its noise
 * bank costs {@code O(samples^2)} to build, which rules it out for multi-second clips; the
 * oscillators here are the same sum of harmonics with a running phase.
 */
public final class ToyMusic {

    /** Samples per second. */
    public static final int RATE = 8000;
    /** Samples per label frame (10 ms). */
    public static final int HOP = 80;
    /** The chord label of a frame without a chord; chords are {@code root + 12 * minor}. */
    public static final int NO_CHORD = 24;
    /** Sound event classes. */
    public static final List<String> EVENTS = List.of("piano", "flute", "kick", "snare", "speech");

    private ToyMusic() {
    }

    /**
     * A labelled clip.
     *
     * @param wave the samples
     * @param f0 per frame the fundamental of the lead voice in Hz, 0 when unvoiced
     * @param onsets note and drum onsets in seconds, sorted
     * @param beats beat times in seconds
     * @param chords per frame the chord label ({@link #NO_CHORD} for none)
     * @param events per frame which {@link #EVENTS} are active
     * @param tempo the starting tempo in BPM
     */
    public record Clip(float[] wave, float[] f0, double[] onsets, double[] beats, int[] chords, boolean[][] events,
            double tempo) {

        /** @return the number of label frames */
        public int frames() {
            return f0.length;
        }

        /** @return per frame whether the lead voice is voiced */
        public boolean[] voiced() {
            boolean[] v = new boolean[f0.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = f0[i] > 0;
            }
            return v;
        }

        /**
         * @param wave a replacement signal of the same length (a degraded copy)
         * @return this clip with another signal
         */
        public Clip withWave(float[] wave) {
            return new Clip(wave, f0, onsets, beats, chords, events, tempo);
        }
    }

    /**
     * @param midi a MIDI note number
     * @return its frequency in Hz (A4 = 69 = 440 Hz)
     */
    public static double hz(double midi) {
        return 440 * Math.pow(2, (midi - 69) / 12.0);
    }

    /** A harmonic profile for a timbre id: odd/even balance, spectral tilt and a weak or strong fundamental. */
    private static double[] profile(int timbre, int harmonics) {
        SplittableRandom r = new SplittableRandom(1000 + timbre);
        double tilt = 0.3 + 1.2 * r.nextDouble();
        double odd = 0.3 + 0.7 * r.nextDouble();
        double[] a = new double[harmonics];
        for (int k = 1; k <= harmonics; k++) {
            a[k - 1] = Math.pow(k, -tilt) * (k % 2 == 1 ? 1 : odd) * (0.7 + 0.6 * r.nextDouble());
        }
        a[0] *= 0.15 + 0.85 * r.nextDouble();
        return a;
    }

    /** Envelope kinds. */
    private enum Envelope { PLUCK, SUSTAIN, BREATHY }

    /**
     * Adds a harmonic tone.
     *
     * @param out the signal
     * @param start first sample
     * @param len samples
     * @param f0 per-sample fundamental supplier (sample index from start)
     */
    private static void tone(float[] out, int start, int len, java.util.function.IntToDoubleFunction f0,
            double[] amps, Envelope env, double gain, SplittableRandom r) {
        double phase = r.nextDouble() * 2 * Math.PI;
        int attack = env == Envelope.PLUCK ? 40 : 200;
        for (int n = 0; n < len && start + n < out.length; n++) {
            double f = f0.applyAsDouble(n);
            phase += 2 * Math.PI * f / RATE;
            double e = Math.min(1, n / (double) attack) * Math.min(1, (len - n) / 160.0);
            if (env == Envelope.PLUCK) {
                e *= Math.exp(-3.0 * n / RATE);
            }
            double s = 0;
            for (int k = 1; k <= amps.length; k++) {
                if (k * f >= RATE / 2.0 - 200) {
                    break;
                }
                double decay = env == Envelope.PLUCK ? Math.exp(-(k - 1) * 2.0 * n / RATE) : 1;
                s += amps[k - 1] * decay * Math.sin(k * phase);
            }
            if (env == Envelope.BREATHY) {
                s += 0.15 * r.nextGaussian() * Math.abs(Math.sin(phase));
            }
            out[start + n] += (float) (gain * e * s);
        }
    }

    private static void kick(float[] out, int start, double gain) {
        double phase = 0;
        for (int n = 0; n < 1600 && start + n < out.length; n++) {
            double f = 50 + 90 * Math.exp(-n / 300.0);
            phase += 2 * Math.PI * f / RATE;
            out[start + n] += (float) (gain * Math.exp(-n / 900.0) * Math.sin(phase) * Math.min(1, n / 8.0));
        }
    }

    private static void snare(float[] out, int start, double gain, SplittableRandom r) {
        double lp = 0;
        for (int n = 0; n < 1200 && start + n < out.length; n++) {
            double noise = r.nextGaussian();
            double hp = noise - lp;
            lp = 0.6 * lp + 0.4 * noise;
            double body = Math.sin(2 * Math.PI * 185 * n / RATE) * Math.exp(-n / 300.0);
            out[start + n] += (float) (gain * (0.5 * hp * Math.exp(-n / 600.0) + 0.6 * body) * Math.min(1, n / 8.0));
        }
    }

    private static void hat(float[] out, int start, double gain, SplittableRandom r) {
        double prev = 0;
        for (int n = 0; n < 300 && start + n < out.length; n++) {
            double noise = r.nextGaussian();
            out[start + n] += (float) (gain * 0.3 * (noise - prev) * Math.exp(-n / 120.0));
            prev = noise;
        }
    }

    private static int frames(int samples) {
        return samples / HOP + 1;
    }

    private static double[] sortedUnique(List<Double> times) {
        double[] t = times.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        List<Double> out = new ArrayList<>();
        for (double v : t) {
            if (out.isEmpty() || v - out.get(out.size() - 1) > 0.03) {
                out.add(v);
            }
        }
        return out.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /** Beat times for a start tempo, an optional linear tempo drift and timing jitter. */
    private static double[] beatGrid(double seconds, double bpm, boolean change, double jitter, SplittableRandom r) {
        double end = bpm * (change ? (r.nextBoolean() ? 1.3 : 0.75) : 1);
        List<Double> beats = new ArrayList<>();
        double t = 0.1 + 0.1 * r.nextDouble();
        while (t < seconds - 0.05) {
            beats.add(Math.max(0, t + jitter * r.nextGaussian()));
            double frac = t / seconds;
            double tempo = bpm + (end - bpm) * frac;
            t += 60 / tempo;
        }
        return beats.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * A monophonic melody: notes of half, one or two beats between MIDI 45 and 81
     * (110-880 Hz) moving by small intervals, 20% rests, with optional 5.5 Hz vibrato of
     * 30 cents.
     *
     * @param seconds the length
     * @param vibrato whether to add vibrato
     * @param timbre the instrument (any int; 0-3 differ clearly)
     * @param random the source
     * @return the clip (no chords, the lead voice as the only event)
     */
    public static Clip melody(double seconds, boolean vibrato, int timbre, SplittableRandom random) {
        int samples = (int) (seconds * RATE);
        float[] wave = new float[samples];
        double[] f0s = new double[samples];
        double bpm = 60 + 120 * random.nextDouble();
        double beat = 60 / bpm;
        double[] amps = profile(timbre, 14);
        Envelope env = Envelope.values()[Math.floorMod(timbre, 3)];
        List<Double> onsets = new ArrayList<>();
        double t = 0.05;
        int note = 52 + random.nextInt(20);
        while (t < seconds - 0.1) {
            double[] durs = {0.5, 1, 1, 2};
            double dur = Math.max(0.12, durs[random.nextInt(durs.length)] * beat);
            dur = Math.min(dur, 0.8);
            if (random.nextDouble() < 0.2) {
                t += dur;
                continue;
            }
            note = Math.max(45, Math.min(81, note + random.nextInt(9) - 4));
            int start = (int) (t * RATE);
            int len = Math.min(samples - start, (int) (dur * RATE * 0.92));
            double base = hz(note + 0.3 * (random.nextDouble() - 0.5));
            double vphase = random.nextDouble() * 2 * Math.PI;
            java.util.function.IntToDoubleFunction f = n -> base * (vibrato
                    ? Math.pow(2, 0.3 / 12 * Math.sin(2 * Math.PI * 5.5 * n / RATE + vphase)) : 1);
            tone(wave, start, len, f, amps, env, 0.25, random);
            for (int n = 0; n < len; n++) {
                f0s[start + n] = f.applyAsDouble(n);
            }
            onsets.add(t);
            t += dur;
        }
        return finish(wave, f0s, onsets, new double[0], null, bpm);
    }

    /**
     * A speech-like utterance with a known F0: voiced syllables of 120-320 ms with a falling
     * or rising pitch contour around the speaker's pitch, separated by pauses and short
     * unvoiced (noise) consonants, with harmonics shaped by three formants.
     *
     * @param voice the speaker (pitch, range, formant scale, tilt, breathiness)
     * @param seconds the length
     * @param random the source
     * @return the clip (speech as the only event)
     */
    public static Clip voice(SpeakerVoices.Voice voice, double seconds, SplittableRandom random) {
        int samples = (int) (seconds * RATE);
        float[] wave = new float[samples];
        double[] f0s = new double[samples];
        List<Double> onsets = new ArrayList<>();
        double t = 0.05 + 0.1 * random.nextDouble();
        while (t < seconds - 0.15) {
            int cons = (int) ((0.02 + 0.04 * random.nextDouble()) * RATE);
            int s0 = (int) (t * RATE);
            for (int n = 0; n < cons && s0 + n < samples; n++) {
                wave[s0 + n] += (float) (0.04 * random.nextGaussian());
            }
            int start = s0 + cons;
            int len = Math.min(samples - start, (int) ((0.12 + 0.2 * random.nextDouble()) * RATE / voice.rate()));
            if (len <= 0) {
                break;
            }
            double f1 = (350 + 450 * random.nextDouble()) * voice.tractScale();
            double f2 = (900 + 1400 * random.nextDouble()) * voice.tractScale();
            double f3 = 2600 * voice.tractScale();
            double base = voice.pitch() * (1 + voice.pitchRange() * (random.nextDouble() - 0.5));
            double slope = (random.nextDouble() - 0.6) * voice.pitchRange();
            final int length = len;
            java.util.function.IntToDoubleFunction f = n -> base * (1 + slope * n / (double) length);
            double[] amps = new double[30];
            double bw = voice.bandwidth() + 60;
            for (int k = 1; k <= amps.length; k++) {
                double fk = k * base;
                double g = 0;
                for (double[] fm : new double[][] {{f1, 1}, {f2, 0.6}, {f3, 0.25}}) {
                    g += fm[1] / (1 + Math.pow((fk - fm[0]) / bw, 2));
                }
                amps[k - 1] = (0.15 + g) * Math.pow(k, -0.5 - voice.tilt());
            }
            tone(wave, start, len, f, amps, voice.breathiness() > 0.2 ? Envelope.BREATHY : Envelope.SUSTAIN, 0.35,
                    random);
            for (int n = 0; n < len; n++) {
                f0s[start + n] = f.applyAsDouble(n);
            }
            onsets.add(start / (double) RATE);
            t = (start + len) / (double) RATE + (random.nextDouble() < 0.3 ? 0.15 + 0.2 * random.nextDouble() : 0.02);
        }
        Clip c = finish(wave, f0s, onsets, new double[0], null, 0);
        boolean[][] ev = c.events();
        for (int i = 0; i < ev.length; i++) {
            ev[i][0] = false;
            ev[i][4] = c.f0()[i] > 0;
        }
        return c;
    }

    private static final int[][] PROGRESSIONS = {{0, 7, 9, 5}, {0, 5, 7, 0}, {9, 5, 0, 7}, {0, 9, 2, 7}, {2, 7, 0, 9}};
    private static final boolean[][] MINOR = {{false, false, true, false}, {false, false, false, false},
        {true, false, false, false}, {false, true, true, false}, {true, false, false, true}};

    /**
     * A polyphonic song: one chord per bar (four beats) from a pop progression transposed to
     * a random key, played as block chords or arpeggios with random inversions, a bass root
     * on every beat, a lead line with passing tones on eighth notes, and optional drums (kick
     * on beats 1 and 3, snare on 2 and 4, hi-hat on eighths).
     *
     * @param seconds the length
     * @param bpm the starting tempo (60-180)
     * @param tempoChange whether the tempo drifts by 25-30% over the clip
     * @param jitter the standard deviation of expressive timing in seconds
     * @param drums whether to add drums
     * @param timbre the chord instrument
     * @param random the source
     * @return the clip; the lead line gives the F0 track
     */
    public static Clip song(double seconds, double bpm, boolean tempoChange, double jitter, boolean drums, int timbre,
            SplittableRandom random) {
        int samples = (int) (seconds * RATE);
        float[] wave = new float[samples];
        double[] f0s = new double[samples];
        double[] beats = beatGrid(seconds, bpm, tempoChange, jitter, random);
        int prog = random.nextInt(PROGRESSIONS.length);
        int key = random.nextInt(12);
        int bar0 = random.nextInt(4);
        double[] chordAmps = profile(timbre, 10);
        double[] leadAmps = profile(timbre + 17, 8);
        double[] bassAmps = profile(timbre + 31, 6);
        boolean arpeggio = random.nextBoolean();
        int frames = frames(samples);
        int[] chords = new int[frames];
        Arrays.fill(chords, NO_CHORD);
        List<Double> onsets = new ArrayList<>();
        boolean[][] events = new boolean[frames][EVENTS.size()];
        for (int b = 0; b < beats.length; b++) {
            int bar = (b + bar0) / 4;
            int pos = (b + bar0) % 4;
            int deg = PROGRESSIONS[prog][bar % 4];
            boolean minor = MINOR[prog][bar % 4];
            int root = (key + deg) % 12;
            double tb = beats[b];
            double next = b + 1 < beats.length ? beats[b + 1] : Math.min(seconds, tb + (tb - beats[Math.max(0, b - 1)]));
            double beatLen = Math.max(0.15, next - tb);
            int label = root + (minor ? 12 : 0);
            int f0 = (int) (tb * RATE / HOP);
            int f1 = Math.min(frames, (int) (next * RATE / HOP));
            for (int f = Math.max(0, f0); f < f1; f++) {
                chords[f] = label;
                events[f][0] = true;
            }
            int[] tones = {0, minor ? 3 : 4, 7};
            int inversion = random.nextInt(3);
            int start = (int) (tb * RATE);
            int len = Math.min(samples - start, (int) (beatLen * RATE * 0.95));
            if (len <= 0) {
                continue;
            }
            // chord tones
            for (int i = 0; i < 3; i++) {
                int midi = 55 + Math.floorMod(root + tones[(i + inversion) % 3] - 7, 12) + (i >= 3 - inversion ? 12 : 0);
                double f = hz(midi);
                int offset = arpeggio ? (int) (i * len / 3.0) : 0;
                tone(wave, start + offset, len - offset, n -> f, chordAmps, Envelope.PLUCK, 0.12, random);
            }
            onsets.add(tb);
            if (arpeggio) {
                onsets.add(tb + len / 3.0 / RATE);
                onsets.add(tb + 2 * len / 3.0 / RATE);
            }
            // bass
            double fb = hz(40 + Math.floorMod(root - 4, 12));
            tone(wave, start, len, n -> fb, bassAmps, Envelope.PLUCK, 0.12, random);
            // lead: chord tone then a passing tone on the off-beat
            int leadRoot = 72 + Math.floorMod(root - 12, 12) - 12;
            int[] leadNotes = {leadRoot + tones[random.nextInt(3)], leadRoot + tones[random.nextInt(3)]
                + (random.nextDouble() < 0.5 ? 2 : 0)};
            for (int h = 0; h < 2; h++) {
                int ls = start + h * len / 2;
                int ll = len / 2 - 20;
                double fl = hz(leadNotes[h]);
                tone(wave, ls, ll, n -> fl, leadAmps, Envelope.SUSTAIN, 0.1, random);
                for (int n = 0; n < ll && ls + n < samples; n++) {
                    f0s[ls + n] = fl;
                }
                if (h == 1) {
                    onsets.add(ls / (double) RATE);
                }
            }
            if (drums) {
                if (pos % 2 == 0) {
                    kick(wave, start, 0.5);
                    mark(events, start, 1600, 2);
                } else {
                    snare(wave, start, 0.35, random);
                    mark(events, start, 1200, 3);
                }
                hat(wave, start, 0.2, random);
                hat(wave, start + len / 2, 0.15, random);
            }
        }
        Clip c = finish(wave, f0s, onsets, beats, chords, bpm);
        return new Clip(c.wave(), c.f0(), c.onsets(), c.beats(), chords, events, bpm);
    }

    private static void mark(boolean[][] events, int start, int len, int cls) {
        for (int f = start / HOP; f <= (start + len) / HOP && f < events.length; f++) {
            if (f >= 0) {
                events[f][cls] = true;
            }
        }
    }

    /**
     * A sound-event scene: 3-8 events of 1-4 types drawn from {@link #EVENTS} (piano and flute
     * notes, kicks, snares, speech syllable runs) at random times, overlapping freely.
     *
     * @param seconds the length
     * @param types how many event types may occur (1-5)
     * @param speaker the voice of the speech events
     * @param random the source
     * @return the clip with per-frame event activity
     */
    public static Clip scene(double seconds, int types, SpeakerVoices.Voice speaker, SplittableRandom random) {
        int samples = (int) (seconds * RATE);
        float[] wave = new float[samples];
        int frames = frames(samples);
        boolean[][] events = new boolean[frames][EVENTS.size()];
        List<Double> onsets = new ArrayList<>();
        int[] allowed = new int[EVENTS.size()];
        for (int i = 0; i < allowed.length; i++) {
            allowed[i] = i;
        }
        for (int i = allowed.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = allowed[i];
            allowed[i] = allowed[j];
            allowed[j] = tmp;
        }
        int count = 3 + random.nextInt(6);
        for (int e = 0; e < count; e++) {
            int cls = allowed[random.nextInt(Math.max(1, Math.min(types, allowed.length)))];
            double t = random.nextDouble() * (seconds - 0.4);
            int start = (int) (t * RATE);
            onsets.add(t);
            switch (cls) {
                case 0, 1 -> {
                    int len = (int) ((0.3 + 0.6 * random.nextDouble()) * RATE);
                    double f = hz(50 + random.nextInt(30));
                    tone(wave, start, len, n -> f, profile(cls == 0 ? 3 : 8, 10), cls == 0 ? Envelope.PLUCK
                            : Envelope.BREATHY, 0.25, random);
                    mark(events, start, Math.min(len, cls == 0 ? (int) (0.6 * RATE) : len), cls);
                }
                case 2 -> {
                    kick(wave, start, 0.6);
                    mark(events, start, 1200, 2);
                }
                case 3 -> {
                    snare(wave, start, 0.45, random);
                    mark(events, start, 900, 3);
                }
                default -> {
                    Clip v = voice(speaker, 0.5 + 0.5 * random.nextDouble(), random);
                    for (int n = 0; n < v.wave().length && start + n < samples; n++) {
                        wave[start + n] += v.wave()[n];
                    }
                    for (int f = 0; f < v.frames(); f++) {
                        int g = start / HOP + f;
                        if (g < frames && v.f0()[f] > 0) {
                            events[g][4] = true;
                        }
                    }
                }
            }
        }
        return new Clip(wave, new float[frames], sortedUnique(onsets), new double[0], filled(frames), events, 0);
    }

    private static int[] filled(int frames) {
        int[] c = new int[frames];
        Arrays.fill(c, NO_CHORD);
        return c;
    }

    private static Clip finish(float[] wave, double[] f0s, List<Double> onsets, double[] beats, int[] chords,
            double bpm) {
        int frames = frames(wave.length);
        float[] f0 = new float[frames];
        boolean[][] events = new boolean[frames][EVENTS.size()];
        for (int f = 0; f < frames; f++) {
            int n = Math.min(wave.length - 1, f * HOP);
            // voiced only if the note covers the whole analysis neighbourhood centre
            f0[f] = (float) f0s[n];
            events[f][0] = f0[f] > 0;
        }
        return new Clip(wave, f0, sortedUnique(onsets), beats, chords == null ? filled(frames) : chords, events, bpm);
    }

    // ------------------------------------------------------------------ degradations

    /**
     * @param x a signal
     * @return its mean power
     */
    public static double power(float[] x) {
        double p = 0;
        for (float v : x) {
            p += v * (double) v;
        }
        return p / Math.max(1, x.length);
    }

    /**
     * Adds white noise at a signal-to-noise ratio.
     *
     * @param x the clean signal
     * @param snrDb the SNR in dB ({@code Infinity} returns a copy)
     * @param random the source
     * @return the noisy signal
     */
    public static float[] addNoise(float[] x, double snrDb, SplittableRandom random) {
        float[] y = x.clone();
        if (Double.isInfinite(snrDb)) {
            return y;
        }
        double std = Math.sqrt(power(x) / Math.pow(10, snrDb / 10));
        for (int i = 0; i < y.length; i++) {
            y[i] += (float) (std * random.nextGaussian());
        }
        return y;
    }

    /**
     * Reverb: convolution with exponentially decaying noise (a direct path plus a diffuse
     * tail reaching -60 dB after {@code rt60} seconds), mixed at -6 dB.
     *
     * @param x the dry signal
     * @param rt60 the reverberation time in seconds
     * @param random the source
     * @return the wet signal, same length
     */
    public static float[] reverb(float[] x, double rt60, SplittableRandom random) {
        int len = (int) (Math.min(rt60, 0.4) * RATE);
        float[] ir = new float[len];
        for (int i = 1; i < len; i += 3) {
            ir[i] = (float) (random.nextGaussian() * Math.pow(10, -3.0 * i / (rt60 * RATE)) * 0.12);
        }
        float[] y = x.clone();
        for (int n = 0; n < x.length; n++) {
            float v = x[n];
            if (v == 0) {
                continue;
            }
            for (int k = 1; k < len && n + k < x.length; k += 3) {
                y[n + k] += v * ir[k];
            }
        }
        return y;
    }

    /**
     * A one-pole low-pass filter.
     *
     * @param x the signal
     * @param cutoff the cutoff in Hz
     * @return the filtered signal
     */
    public static float[] lowPass(float[] x, double cutoff) {
        double a = Math.exp(-2 * Math.PI * cutoff / RATE);
        float[] y = new float[x.length];
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s = (1 - a) * x[i] + a * s;
            y[i] = (float) s;
        }
        // second pass for a steeper slope
        double t = 0;
        for (int i = 0; i < y.length; i++) {
            t = (1 - a) * y[i] + a * t;
            y[i] = (float) t;
        }
        return y;
    }

    /**
     * Plays a signal faster or slower by linear-interpolation resampling (pitch and tempo
     * both change, as with a tape).
     *
     * @param x the signal
     * @param factor the speed, such as 1.02
     * @return the resampled signal, {@code length / factor} samples
     */
    public static float[] speed(float[] x, double factor) {
        int n = (int) (x.length / factor);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            double p = i * factor;
            int a = (int) p;
            double f = p - a;
            float va = a < x.length ? x[a] : 0;
            float vb = a + 1 < x.length ? x[a + 1] : 0;
            y[i] = (float) (va * (1 - f) + vb * f);
        }
        return y;
    }
}
