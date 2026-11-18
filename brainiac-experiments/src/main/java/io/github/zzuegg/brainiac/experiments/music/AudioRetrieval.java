package io.github.zzuegg.brainiac.experiments.music;

import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experimental.data.ToyMusic;
import io.github.zzuegg.brainiac.experimental.ml.DynamicTimeWarping;
import io.github.zzuegg.brainiac.experimental.music.LandmarkFingerprint;
import io.github.zzuegg.brainiac.experimental.music.MusicFeatures;
import io.github.zzuegg.brainiac.experimental.music.MusicMetrics;
import io.github.zzuegg.brainiac.experimental.music.NeuralFingerprint;
import io.github.zzuegg.brainiac.experimental.music.SoundEventDetector;
import io.github.zzuegg.brainiac.experimental.nn.Mfcc;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 11 items 11.5.5-11.5.6.
 *
 * <ul>
 *   <li>{@link #fingerprinting} (11.5.5): top-1 identification of 2 s degraded queries
 *       (noise at 20 and 0 dB, a 1 kHz low-pass, a 2% speed-up, reverb) against a database
 *       of {@code ToyMusic} songs, melodies and voices, for landmark hashes, contrastive
 *       neural fingerprints, MFCC + DTW template matching and mean-pooled MFCC nearest
 *       neighbours, with milliseconds per query.
 *   <li>{@link #eventDetection} (11.5.6): event-based F1 (200 ms onset collar) and 1 s
 *       segment error rate of the CRNN against a frame MLP and a windowed CNN without
 *       recurrence, on scenes overlapping piano, flute, kick, snare and speech events.
 * </ul>
 */
public final class AudioRetrieval {

    private AudioRetrieval() {
    }

    /**
     * A row.
     *
     * @param condition the query degradation or test set
     * @param method the method
     * @param score top-1 rate or event F1
     * @param second milliseconds per query or segment error rate
     */
    public record Row(String condition, String method, double score, double second) {
    }

    private static final int RATE = ToyMusic.RATE;

    private static List<float[]> database(int tracks, double seconds, SplittableRandom r) {
        List<float[]> db = new ArrayList<>();
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(tracks, 77);
        for (int i = 0; i < tracks; i++) {
            ToyMusic.Clip c = switch (i % 3) {
                case 0 -> ToyMusic.song(seconds, 80 + 80 * r.nextDouble(), false, 0.01, r.nextBoolean(), i, r);
                case 1 -> ToyMusic.melody(seconds, r.nextBoolean(), i, r);
                default -> ToyMusic.voice(voices.get(i), seconds, r);
            };
            db.add(c.wave());
        }
        return db;
    }

    private static float[] degrade(String kind, float[] x, SplittableRandom r) {
        return switch (kind) {
            case "noise 20dB" -> ToyMusic.addNoise(x, 20, r);
            case "noise 0dB" -> ToyMusic.addNoise(x, 0, r);
            case "low-pass 1kHz" -> ToyMusic.addNoise(ToyMusic.lowPass(x, 1000), 30, r);
            case "speed +2%" -> ToyMusic.addNoise(ToyMusic.speed(x, 1.02), 30, r);
            case "reverb" -> ToyMusic.addNoise(ToyMusic.reverb(x, 0.3, r), 30, r);
            default -> x.clone();
        };
    }

    private static float[] randomDegradation(float[] x, SplittableRandom r) {
        float[] y = x;
        if (r.nextDouble() < 0.5) {
            y = ToyMusic.lowPass(y, 800 + 1500 * r.nextDouble());
        }
        if (r.nextDouble() < 0.5) {
            float[] s = ToyMusic.speed(y, 0.97 + 0.06 * r.nextDouble());
            y = java.util.Arrays.copyOf(s, x.length);
        }
        if (r.nextDouble() < 0.4) {
            y = ToyMusic.reverb(y, 0.1 + 0.3 * r.nextDouble(), r);
        }
        return ToyMusic.addNoise(y, -2 + 25 * r.nextDouble(), r);
    }

    private static int mfccDtw(List<float[][]> db, float[][] q, int hop) {
        int best = -1;
        double bestD = Double.MAX_VALUE;
        for (int t = 0; t < db.size(); t++) {
            float[][] ref = db.get(t);
            for (int s = 0; s + q.length <= ref.length; s += hop) {
                double d = DynamicTimeWarping.distance(q, java.util.Arrays.copyOfRange(ref, s, s + q.length), 8);
                if (d < bestD) {
                    bestD = d;
                    best = t;
                }
            }
        }
        return best;
    }

    private static float[] mean(float[][] m, int from, int to) {
        float[] v = new float[m[0].length];
        for (int t = from; t < to; t++) {
            for (int j = 0; j < v.length; j++) {
                v[j] += m[t][j] / (to - from);
            }
        }
        return v;
    }

    private static int meanMfcc(List<float[][]> db, float[][] q, int hop) {
        float[] qm = mean(q, 0, q.length);
        int best = -1;
        double bestD = Double.MAX_VALUE;
        for (int t = 0; t < db.size(); t++) {
            float[][] ref = db.get(t);
            for (int s = 0; s + q.length <= ref.length; s += hop) {
                float[] rm = mean(ref, s, s + q.length);
                double d = 0;
                for (int j = 0; j < qm.length; j++) {
                    d += (qm[j] - rm[j]) * (qm[j] - rm[j]);
                }
                if (d < bestD) {
                    bestD = d;
                    best = t;
                }
            }
        }
        return best;
    }

    private static float[][] mfcc(float[] x) {
        return Mfcc.compute(x, RATE, 256, 128, 26, 13, false);
    }

    /**
     * Audio fingerprinting.
     *
     * @param tracks database recordings
     * @param queries queries per degradation
     * @param steps neural fingerprint training steps
     * @param seed the seed
     * @return rows: per condition and method the top-1 rate and ms per query
     */
    public static List<Row> fingerprinting(int tracks, int queries, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double seconds = 5;
        List<float[]> db = database(tracks, seconds, r);
        LandmarkFingerprint landmarks = new LandmarkFingerprint();
        db.forEach(landmarks::add);
        NeuralFingerprint neural = new NeuralFingerprint(RATE, RATE, 32, seed);
        neural.train(db, AudioRetrieval::randomDegradation, steps, 12, 2e-3f, seed);
        neural.index(db, RATE / 4);
        List<float[][]> dbMfcc = new ArrayList<>();
        for (float[] x : db) {
            dbMfcc.add(mfcc(x));
        }
        String[] conditions = {"noise 20dB", "noise 0dB", "low-pass 1kHz", "speed +2%", "reverb"};
        List<Row> rows = new ArrayList<>();
        int qLen = 2 * RATE;
        for (String cond : conditions) {
            SplittableRandom qr = new SplittableRandom(seed + cond.hashCode());
            List<float[]> qs = new ArrayList<>();
            int[] truth = new int[queries];
            for (int i = 0; i < queries; i++) {
                truth[i] = qr.nextInt(db.size());
                int start = qr.nextInt(db.get(truth[i]).length - qLen);
                qs.add(degrade(cond, java.util.Arrays.copyOfRange(db.get(truth[i]), start, start + qLen), qr));
            }
            List<Function<float[], Integer>> methods = List.of(q -> landmarks.identify(q)[0],
                    q -> neural.identify(q, RATE / 4, 3), q -> mfccDtw(dbMfcc, mfcc(q), 16),
                    q -> meanMfcc(dbMfcc, mfcc(q), 8));
            String[] names = {"landmarks", "neural fingerprint", "MFCC+DTW", "mean MFCC kNN"};
            for (int m = 0; m < methods.size(); m++) {
                long t0 = System.nanoTime();
                int hit = 0;
                for (int i = 0; i < queries; i++) {
                    hit += methods.get(m).apply(qs.get(i)) == truth[i] ? 1 : 0;
                }
                double ms = (System.nanoTime() - t0) / 1e6 / queries;
                rows.add(new Row(cond, names[m], hit / (double) queries, ms));
            }
        }
        return rows;
    }

    private static final int SED_HOP = 160;
    private static final int SED_MELS = 32;

    private static float[][] sedFeatures(float[] x) {
        float[][] m = MusicFeatures.logMel(x, RATE, 512, SED_HOP, SED_MELS);
        for (float[] row : m) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (row[j] + 6f) * 0.25f;
            }
        }
        return m;
    }

    private static boolean[][] sedLabels(ToyMusic.Clip c, int frames) {
        boolean[][] y = new boolean[frames][ToyMusic.EVENTS.size()];
        int step = SED_HOP / ToyMusic.HOP;
        for (int t = 0; t < frames; t++) {
            y[t] = c.events()[Math.min(c.events().length - 1, t * step)].clone();
        }
        return y;
    }

    /**
     * Sound event detection.
     *
     * @param trainScenes training scenes
     * @param testScenes test scenes
     * @param steps training steps per model
     * @param seed the seed
     * @return rows: per model event F1 and segment error rate
     */
    public static List<Row> eventDetection(int trainScenes, int testScenes, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(8, seed);
        List<float[][]> x = new ArrayList<>();
        List<boolean[][]> y = new ArrayList<>();
        for (int i = 0; i < trainScenes; i++) {
            ToyMusic.Clip c = ToyMusic.scene(4, 1 + r.nextInt(4), voices.get(i % 4), r);
            float[] w = ToyMusic.addNoise(c.wave(), 5 + 20 * r.nextDouble(), r);
            float[][] f = sedFeatures(w);
            x.add(f);
            y.add(sedLabels(c, f.length));
        }
        List<float[][]> tx = new ArrayList<>();
        List<boolean[][]> ty = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 1);
        for (int i = 0; i < testScenes; i++) {
            ToyMusic.Clip c = ToyMusic.scene(4, 1 + tr.nextInt(4), voices.get(4 + i % 4), tr);
            float[] w = ToyMusic.addNoise(c.wave(), 5 + 20 * tr.nextDouble(), tr);
            float[][] f = sedFeatures(w);
            tx.add(f);
            ty.add(sedLabels(c, f.length));
        }
        List<Row> rows = new ArrayList<>();
        int frames = x.get(0).length;
        int collar = (int) Math.round(0.2 * RATE / SED_HOP);
        int segment = RATE / SED_HOP;
        for (SoundEventDetector.Architecture a : SoundEventDetector.Architecture.values()) {
            SoundEventDetector d = new SoundEventDetector(a, SED_MELS, ToyMusic.EVENTS.size(), frames, seed);
            d.train(x, y, steps, 4, 3e-3f, seed);
            double f1 = 0;
            double er = 0;
            for (int i = 0; i < tx.size(); i++) {
                boolean[][] est = d.detect(tx.get(i), 5, 0.5);
                f1 += MusicMetrics.eventF1(ty.get(i), est, collar) / tx.size();
                er += MusicMetrics.segmentErrorRate(ty.get(i), est, segment) / tx.size();
            }
            rows.add(new Row("scenes", a.name(), f1, er));
        }
        return rows;
    }
}
