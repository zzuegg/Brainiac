package io.github.zzuegg.brainiac.experiments.music;

import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experimental.data.ToyMusic;
import io.github.zzuegg.brainiac.experimental.music.Crepe;
import io.github.zzuegg.brainiac.experimental.music.EquivariantPitch;
import io.github.zzuegg.brainiac.experimental.music.MusicMetrics;
import io.github.zzuegg.brainiac.experimental.music.PitchTrackers;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 11 items 11.5.1-11.5.2: pitch trackers on {@code ToyMusic} melodies with noise at
 * several SNRs, and on speech-like voices for transfer: the spectral-peak and harmonic-sum
 * baselines, autocorrelation, YIN, pYIN (11.5.1), a supervised CREPE and the self-supervised
 * transposition-equivariant model with and without its pseudo-labelled voicing classifier
 * (11.5.2), scored by raw pitch and chroma accuracy, voicing F1 and octave errors.
 */
public final class PitchShootout {

    private PitchShootout() {
    }

    /**
     * The budget.
     *
     * @param trainClips unlabelled (PESTO) or labelled (CREPE) training melodies
     * @param testClips test clips per condition
     * @param seconds clip length
     * @param steps training steps of each network
     * @param snrs the test SNRs in dB
     */
    public record Budget(int trainClips, int testClips, double seconds, int steps, double[] snrs) {
    }

    /**
     * One row.
     *
     * @param condition the test condition
     * @param method the tracker
     * @param labels training labels used (labelled frames)
     * @param score the melody scores
     */
    public record Row(String condition, String method, int labels, MusicMetrics.Pitch score) {
    }

    private static final double FMIN = 60;
    private static final double FMAX = 1000;

    private static MusicMetrics.Pitch score(List<ToyMusic.Clip> clips, Function<float[], PitchTrackers.Track> t) {
        List<Float> ref = new ArrayList<>();
        List<Float> est = new ArrayList<>();
        List<Boolean> v = new ArrayList<>();
        for (ToyMusic.Clip c : clips) {
            PitchTrackers.Track tr = t.apply(c.wave());
            for (int i = 0; i < c.frames(); i++) {
                ref.add(c.f0()[i]);
                est.add(tr.f0()[i]);
                v.add(tr.voiced()[i]);
            }
        }
        float[] r = new float[ref.size()];
        float[] e = new float[ref.size()];
        boolean[] vv = new boolean[ref.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = ref.get(i);
            e[i] = est.get(i);
            vv[i] = v.get(i);
        }
        return MusicMetrics.pitch(r, e, vv);
    }

    /**
     * Runs the shoot-out.
     *
     * @param b the budget
     * @param seed the seed
     * @return the rows, condition by condition
     */
    public static List<Row> run(Budget b, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<ToyMusic.Clip> train = new ArrayList<>();
        for (int i = 0; i < b.trainClips(); i++) {
            train.add(ToyMusic.melody(b.seconds(), i % 2 == 0, i, r));
        }
        // CREPE: supervised on the clean labelled frames (plus noisy copies)
        Crepe crepe = new Crepe(55, FMAX, 8, seed);
        List<float[]> frames = new ArrayList<>();
        List<Float> labels = new ArrayList<>();
        List<float[]> waves = new ArrayList<>();
        List<float[]> f0s = new ArrayList<>();
        for (ToyMusic.Clip c : train) {
            waves.add(ToyMusic.addNoise(c.wave(), 5 + 20 * r.nextDouble(), r));
            f0s.add(c.f0());
        }
        Crepe.collect(waves, f0s, ToyMusic.HOP, 2, frames, labels);
        crepe.train(frames, labels, b.steps() / 2, 16, 3e-3f, seed);
        int crepeLabels = frames.size();
        // PESTO: unlabelled audio only, with noise so that the frame weights have work to do
        EquivariantPitch pesto = new EquivariantPitch(55, 4, 8, seed);
        List<float[][]> fe = new ArrayList<>();
        List<float[]> en = new ArrayList<>();
        int total = 0;
        for (float[] w : waves) {
            float[] e = new float[w.length / ToyMusic.HOP + 1];
            float[][] f = EquivariantPitch.frontEnd(w, ToyMusic.RATE, ToyMusic.HOP, 55, pesto.bins(), e);
            fe.add(f);
            en.add(e);
            total += (f.length + 1) / 2;
        }
        float[][] all = new float[total][];
        float[] energy = new float[total];
        int o = 0;
        for (int i = 0; i < fe.size(); i++) {
            for (int t = 0; t < fe.get(i).length; t += 2) {
                all[o] = fe.get(i)[t];
                energy[o++] = en.get(i)[t];
            }
        }
        pesto.train(all, energy, b.steps(), 3, 16, 3e-3f, seed + 1);
        pesto.calibrate(ToyMusic.RATE, ToyMusic.HOP, 110, 196, 330, 523);

        List<Row> rows = new ArrayList<>();
        List<String> conditions = new ArrayList<>();
        List<List<ToyMusic.Clip>> sets = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 99);
        List<ToyMusic.Clip> clean = new ArrayList<>();
        for (int i = 0; i < b.testClips(); i++) {
            clean.add(ToyMusic.melody(b.seconds(), i % 2 == 1, 100 + i, tr));
        }
        for (double snr : b.snrs()) {
            List<ToyMusic.Clip> noisy = new ArrayList<>();
            for (ToyMusic.Clip c : clean) {
                noisy.add(c.withWave(ToyMusic.addNoise(c.wave(), snr, tr)));
            }
            conditions.add(String.format("melody %.0fdB", snr));
            sets.add(noisy);
        }
        List<ToyMusic.Clip> voices = new ArrayList<>();
        List<SpeakerVoices.Voice> speakers = SpeakerVoices.speakers(b.testClips(), seed + 5);
        for (SpeakerVoices.Voice v : speakers) {
            ToyMusic.Clip c = ToyMusic.voice(v, b.seconds(), tr);
            voices.add(c.withWave(ToyMusic.addNoise(c.wave(), 15, tr)));
        }
        conditions.add("voice 15dB");
        sets.add(voices);
        for (int s = 0; s < sets.size(); s++) {
            List<ToyMusic.Clip> set = sets.get(s);
            String cond = conditions.get(s);
            for (PitchTrackers.Method m : PitchTrackers.Method.values()) {
                rows.add(new Row(cond, m.name(), 0, score(set,
                        x -> PitchTrackers.track(m, x, ToyMusic.RATE, ToyMusic.HOP, FMIN, FMAX))));
            }
            rows.add(new Row(cond, "CREPE", crepeLabels, score(set, x -> crepe.track(x, ToyMusic.HOP, 0.5))));
            rows.add(new Row(cond, "PESTO", 0, score(set, x -> pesto.track(x, ToyMusic.RATE, ToyMusic.HOP, false))));
            rows.add(new Row(cond, "PESTO+SCE voicing", 0,
                    score(set, x -> pesto.track(x, ToyMusic.RATE, ToyMusic.HOP, true))));
        }
        return rows;
    }
}
