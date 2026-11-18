package io.github.zzuegg.brainiac.experiments.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ToyMusic;
import io.github.zzuegg.brainiac.experimental.music.BeatTracker;
import io.github.zzuegg.brainiac.experimental.music.Chroma;
import io.github.zzuegg.brainiac.experimental.music.ChordRecognizer;
import io.github.zzuegg.brainiac.experimental.music.MusicFeatures;
import io.github.zzuegg.brainiac.experimental.music.MusicMetrics;
import io.github.zzuegg.brainiac.experimental.music.NeuralBeatTracker;
import io.github.zzuegg.brainiac.experimental.music.OnsetDetection;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 11 items 11.5.3-11.5.4 on {@code ToyMusic} songs.
 *
 * <ul>
 *   <li>{@link #rhythm} (11.5.3): onset F1 (50 ms) of the energy envelope, spectral flux and
 *       superflux, each with its peak threshold tuned on training clips; beat F1 (70 ms) and
 *       CMLt of Ellis's DP tracker on the superflux envelope and of the neural tracker trained
 *       with the shift-tolerant loss or with plain BCE, on steady clips with and without
 *       drums and on clips whose tempo drifts by 25-30%.
 *   <li>{@link #harmony} (11.5.4): weighted chord symbol recall and segmentation quality of
 *       frame-wise and HMM-smoothed templates, the root-tied Gaussian HMM, a CNN-CRF on
 *       log-frequency spectra and a frame-wise MLP on log-mel frames, with the instrument
 *       timbres of the test songs unseen in training.
 * </ul>
 */
public final class RhythmAndHarmony {

    private RhythmAndHarmony() {
    }

    /**
     * One row.
     *
     * @param condition the test set
     * @param method the method
     * @param f1 onset or beat F1, or chord recall
     * @param second CMLt for beats, segmentation for chords, NaN for onsets
     */
    public record Row(String condition, String method, double f1, double second) {
    }

    private static final double SECONDS = 4;

    private static List<ToyMusic.Clip> songs(int n, boolean change, boolean drums, int timbre, double minBpm,
            double maxBpm, SplittableRandom r) {
        List<ToyMusic.Clip> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double bpm = minBpm + (maxBpm - minBpm) * r.nextDouble();
            ToyMusic.Clip c = ToyMusic.song(SECONDS, bpm, change, 0.01, drums, timbre + i, r);
            out.add(c.withWave(ToyMusic.addNoise(c.wave(), 25, r)));
        }
        return out;
    }

    private static double onsetF1(List<ToyMusic.Clip> clips, List<float[]> functions, double delta) {
        double s = 0;
        for (int i = 0; i < clips.size(); i++) {
            int[] peaks = MusicMetrics.pickPeaks(functions.get(i), 3, 10, delta, 3);
            double[] t = new double[peaks.length];
            for (int j = 0; j < peaks.length; j++) {
                t[j] = peaks[j] * ToyMusic.HOP / (double) ToyMusic.RATE;
            }
            s += MusicMetrics.matchF1(clips.get(i).onsets(), t, 0.05)[2];
        }
        return s / clips.size();
    }

    private static List<float[]> functions(List<ToyMusic.Clip> clips, OnsetDetection.Function f) {
        List<float[]> out = new ArrayList<>();
        for (ToyMusic.Clip c : clips) {
            out.add(OnsetDetection.standardise(OnsetDetection.function(f, c.wave(), ToyMusic.RATE, ToyMusic.HOP)));
        }
        return out;
    }

    /**
     * Onsets and beats.
     *
     * @param trainClips training songs (threshold tuning and the neural trackers)
     * @param testClips songs per test condition
     * @param steps neural tracker steps
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> rhythm(int trainClips, int testClips, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<ToyMusic.Clip> train = new ArrayList<>();
        for (int i = 0; i < trainClips; i++) {
            train.addAll(songs(1, i % 2 == 0, i % 3 != 0, i, 70, 170, r));
        }
        List<Row> rows = new ArrayList<>();
        double fps = ToyMusic.RATE / (double) ToyMusic.HOP;
        double[] deltas = {0.3, 0.6, 1.0, 1.5, 2.0};
        double[] tuned = new double[OnsetDetection.Function.values().length];
        for (OnsetDetection.Function f : OnsetDetection.Function.values()) {
            double best = -1;
            List<float[]> fs = functions(train, f);
            for (double d : deltas) {
                double v = onsetF1(train, fs, d);
                if (v > best) {
                    best = v;
                    tuned[f.ordinal()] = d;
                }
            }
        }
        int mels = 24;
        List<float[][]> feats = new ArrayList<>();
        List<double[]> beats = new ArrayList<>();
        for (ToyMusic.Clip c : train) {
            feats.add(NeuralBeatTracker.features(c.wave(), ToyMusic.RATE, ToyMusic.HOP, mels));
            beats.add(c.beats());
        }
        NeuralBeatTracker tolerant = new NeuralBeatTracker(mels, 16, seed);
        tolerant.train(feats, beats, fps, NeuralBeatTracker.Loss.SHIFT_TOLERANT, steps, 4, 3e-3f, seed);
        NeuralBeatTracker plain = new NeuralBeatTracker(mels, 16, seed);
        plain.train(feats, beats, fps, NeuralBeatTracker.Loss.PLAIN, steps, 4, 3e-3f, seed);
        BeatTracker dp = new BeatTracker(100, 120);

        SplittableRandom tr = new SplittableRandom(seed + 1);
        String[] names = {"steady+drums", "steady no drums", "tempo change"};
        List<List<ToyMusic.Clip>> sets = List.of(songs(testClips, false, true, 500, 70, 170, tr),
                songs(testClips, false, false, 600, 70, 170, tr), songs(testClips, true, true, 700, 70, 170, tr));
        for (int s = 0; s < sets.size(); s++) {
            List<ToyMusic.Clip> set = sets.get(s);
            for (OnsetDetection.Function f : OnsetDetection.Function.values()) {
                rows.add(new Row(names[s], "onset " + f, onsetF1(set, functions(set, f), tuned[f.ordinal()]),
                        Double.NaN));
            }
            List<Function<ToyMusic.Clip, double[]>> trackers = List.of(
                    c -> dp.beats(OnsetDetection.function(OnsetDetection.Function.SUPERFLUX, c.wave(), ToyMusic.RATE,
                            ToyMusic.HOP), fps),
                    c -> tolerant.beats(NeuralBeatTracker.features(c.wave(), ToyMusic.RATE, ToyMusic.HOP, mels), fps,
                            0.5),
                    c -> plain.beats(NeuralBeatTracker.features(c.wave(), ToyMusic.RATE, ToyMusic.HOP, mels), fps, 0.5));
            String[] tn = {"beat Ellis DP", "beat neural shift-tolerant", "beat neural plain BCE"};
            for (int k = 0; k < trackers.size(); k++) {
                double f1 = 0;
                double cml = 0;
                for (ToyMusic.Clip c : set) {
                    double[] est = trackers.get(k).apply(c);
                    f1 += MusicMetrics.matchF1(c.beats(), est, 0.07)[2];
                    cml += MusicMetrics.cmlt(c.beats(), est);
                }
                rows.add(new Row(names[s], tn[k], f1 / set.size(), cml / set.size()));
            }
        }
        return rows;
    }

    private static final int CHORD_HOP = 400;

    private static int[] chordLabels(ToyMusic.Clip c, int frames) {
        int[] y = new int[frames];
        int step = CHORD_HOP / ToyMusic.HOP;
        for (int t = 0; t < frames; t++) {
            y[t] = c.chords()[Math.min(c.chords().length - 1, t * step)];
        }
        return y;
    }

    private static float[][] logSpec(float[] x) {
        return MusicFeatures.logFrequency(MusicFeatures.magnitude(x, Chroma.N_FFT, CHORD_HOP), ToyMusic.RATE,
                Chroma.FMIN, 12, 60, true);
    }

    /**
     * Chord recognition.
     *
     * @param trainClips training songs
     * @param testClips test songs (unseen timbres)
     * @param steps CNN-CRF and MLP steps
     * @param seed the seed
     * @return the rows
     */
    public static List<Row> harmony(int trainClips, int testClips, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<ToyMusic.Clip> train = songs(trainClips, false, r.nextBoolean(), 0, 110, 170, r);
        List<ToyMusic.Clip> test = songs(testClips, false, true, 300, 110, 170, new SplittableRandom(seed + 3));
        List<float[][]> trChroma = new ArrayList<>();
        List<float[]> trEnergy = new ArrayList<>();
        List<int[]> trLabels = new ArrayList<>();
        List<float[][]> trSpec = new ArrayList<>();
        List<float[][]> trMel = new ArrayList<>();
        for (ToyMusic.Clip c : train) {
            float[] e = new float[c.wave().length / CHORD_HOP + 1];
            float[][] ch = Chroma.compute(c.wave(), ToyMusic.RATE, CHORD_HOP, 1, e);
            trChroma.add(ch);
            trEnergy.add(e);
            trLabels.add(chordLabels(c, ch.length));
            trSpec.add(logSpec(c.wave()));
            trMel.add(MusicFeatures.logMel(c.wave(), ToyMusic.RATE, 1024, CHORD_HOP, 32));
        }
        double medianEnergy = median(trEnergy);
        ChordRecognizer.GaussianHmm hmm = ChordRecognizer.GaussianHmm.fit(trChroma, trEnergy, trLabels);
        ChordRecognizer.CnnCrf crf = new ChordRecognizer.CnnCrf(60, 24, seed);
        crf.train(trSpec, trLabels, steps, 4, 5e-3f, seed);
        SimpleMlp mlp = SimpleMlp.relu(seed, 32, 64, ChordRecognizer.LABELS);
        trainMlp(mlp, trMel, trLabels, steps, seed);

        String[] names = {"template frame-wise", "template + HMM", "Gaussian frame-wise", "Gaussian HMM", "CNN-CRF",
            "MLP on mel frames"};
        double[] recall = new double[names.length];
        double[] seg = new double[names.length];
        for (ToyMusic.Clip c : test) {
            float[] e = new float[c.wave().length / CHORD_HOP + 1];
            float[][] ch = Chroma.compute(c.wave(), ToyMusic.RATE, CHORD_HOP, 1, e);
            int[] gold = chordLabels(c, ch.length);
            float[][] mel = MusicFeatures.logMel(c.wave(), ToyMusic.RATE, 1024, CHORD_HOP, 32);
            int[][] est = {ChordRecognizer.template(ch, e, 0.05 * medianEnergy, 0),
                ChordRecognizer.template(ch, e, 0.05 * medianEnergy, 0.9), hmm.decode(ch, e, false),
                hmm.decode(ch, e, true), crf.decode(logSpec(c.wave())), mlpDecode(mlp, mel)};
            for (int k = 0; k < names.length; k++) {
                recall[k] += MusicMetrics.chordRecall(gold, est[k]) / test.size();
                seg[k] += MusicMetrics.segmentation(gold, est[k]) / test.size();
            }
        }
        List<Row> rows = new ArrayList<>();
        for (int k = 0; k < names.length; k++) {
            rows.add(new Row("unseen timbres", names[k], recall[k], seg[k]));
        }
        return rows;
    }

    private static double median(List<float[]> xs) {
        List<Float> all = new ArrayList<>();
        for (float[] x : xs) {
            for (float v : x) {
                all.add(v);
            }
        }
        all.sort(null);
        return all.get(all.size() / 2);
    }

    private static void trainMlp(SimpleMlp mlp, List<float[][]> x, List<int[]> y, int steps, long seed) {
        List<float[]> rows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            for (int t = 0; t < x.get(i).length; t++) {
                rows.add(x.get(i)[t]);
                labels.add(y.get(i)[t]);
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            float[][] xb = new float[64][];
            float[] yb = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(rows.size());
                xb[i] = rows.get(j);
                yb[i] = labels.get(j);
            }
            mlp.parameters().forEach(Var::zeroGrad);
            Var loss = Functional.crossEntropy(mlp.forward(Var.of(scale(MusicFeatures.stack(xb)))),
                    Tensor.of(Shape.of(64), yb));
            loss.backward();
            opt.step(mlp.parameters());
        }
        mlp.parameters().forEach(Var::zeroGrad);
    }

    private static Tensor scale(Tensor logMel) {
        return logMel.add(6f).mul(0.25f);
    }

    private static int[] mlpDecode(SimpleMlp mlp, float[][] mel) {
        float[] z = mlp.predict(scale(MusicFeatures.stack(mel))).toArray();
        int c = ChordRecognizer.LABELS;
        int[] out = new int[mel.length];
        for (int t = 0; t < mel.length; t++) {
            int b = 0;
            for (int k = 1; k < c; k++) {
                if (z[t * c + k] > z[t * c + b]) {
                    b = k;
                }
            }
            out[t] = b;
        }
        return out;
    }
}
