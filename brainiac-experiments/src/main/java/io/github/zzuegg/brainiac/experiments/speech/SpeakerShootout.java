package io.github.zzuegg.brainiac.experiments.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SpeakerVoices;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.nn.Mfcc;
import io.github.zzuegg.brainiac.experimental.speech.Diarization;
import io.github.zzuegg.brainiac.experimental.speech.SpeakerEmbedding;
import io.github.zzuegg.brainiac.experimental.speech.SpeechFeatures;
import io.github.zzuegg.brainiac.experimental.speech.VoiceActivity;
import io.github.zzuegg.brainiac.experiments.audio.SpokenDigitRecognition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Speaker verification and diarisation on {@link SpeakerVoices} (wave 10, items 10.6.5 and
 * 10.6.8).
 *
 * <p>{@link #verification}: x-vectors ({@link SpeakerEmbedding}) trained with a normalised
 * softmax, with additive angular margin (AAM), with AAM and attentive statistics pooling, and
 * with GE2E, scored by cosine and by two-covariance PLDA, against mean-MFCC vectors with cosine
 * scoring, a GMM-UBM baseline (a {@link GaussianMixture} universal background model, per
 * utterance MAP-adapted means as a supervector, cosine scoring) and the encoder of
 * {@link SpokenDigitRecognition}'s digit recogniser trained on the digits (a content model)
 * mean-pooled as a speaker embedding. Training and held-out speakers are disjoint; the metrics
 * are the equal error rate and minDCF (p = 0.05) over all test pairs, also against utterance
 * length.
 *
 * <p>{@link #diarization}: the modular pipeline (Sohn VAD with HMM hangover or oracle VAD,
 * 0.4 s windows, x-vector or mean-MFCC window embeddings, agglomerative or spectral clustering
 * with an eigengap or oracle speaker count) against end-to-end neural diarisation
 * ({@link Diarization.Eend}) and the one-speaker baseline, scored by the diarisation error
 * rate on two- and three-speaker conversations with chosen overlap.
 */
public final class SpeakerShootout {

    static final int MELS = 24;
    static final int HOP = SpeakerVoices.FRAME;
    static final int CROP = 48;

    private SpeakerShootout() {
    }

    /** Verification systems. */
    public enum Verifier {
        /** Mean MFCC vector, cosine. */
        MEAN_MFCC,
        /** GMM-UBM MAP-adapted mean supervector, cosine. */
        GMM_UBM,
        /** The digit recogniser's encoder, mean-pooled. */
        DIGIT_ENCODER,
        /** x-vector trained with the normalised softmax (no margin). */
        XVECTOR_SOFTMAX,
        /** x-vector trained with additive angular margin softmax. */
        XVECTOR_AAM,
        /** x-vector with attentive statistics pooling and AAM. */
        XVECTOR_ATTENTIVE_AAM,
        /** x-vector network trained with the GE2E loss. */
        GE2E
    }

    /**
     * A verification result.
     *
     * @param method the system
     * @param eer the equal error rate with cosine scoring
     * @param minDcf minDCF (p = 0.05) with cosine scoring
     * @param eerPlda the EER with PLDA scoring (NaN when not applicable)
     */
    public record VerificationRow(String method, double eer, double minDcf, double eerPlda) {
        @Override
        public String toString() {
            return String.format("%s EER %.3f minDCF %.3f%s", method, eer, minDcf,
                    Double.isNaN(eerPlda) ? "" : String.format(" (PLDA EER %.3f)", eerPlda));
        }
    }

    /** An utterance with its speaker. */
    record Utt(float[] wave, float[][] features, int speaker, int[] digits) {
    }

    static float[][] features(float[] wave) {
        float[][] lm = SpeechFeatures.logMel(wave, SpeakerVoices.RATE, HOP, MELS);
        for (float[] f : lm) {
            for (int b = 0; b < f.length; b++) {
                f[b] = Math.max(-3, f[b] / 5);
            }
        }
        return lm;
    }

    static List<Utt> utterances(List<SpeakerVoices.Voice> voices, int offset, int count, int perSpeaker, double snrDb,
            SplittableRandom r) {
        List<Utt> out = new ArrayList<>();
        for (int s = 0; s < count; s++) {
            for (int u = 0; u < perSpeaker; u++) {
                int[] digits = new int[2 + r.nextInt(3)];
                for (int i = 0; i < digits.length; i++) {
                    digits[i] = r.nextInt(10);
                }
                SpeakerVoices.Utterance clean = SpeakerVoices.speak(digits, voices.get(offset + s), r);
                float[] wave = SpeakerVoices.addNoise(clean.wave(), clean.active(), SpeakerVoices.Noise.WHITE, snrDb, r)[0];
                out.add(new Utt(wave, features(wave), s, digits));
            }
        }
        return out;
    }

    /** A trained embedding function, with optional PLDA. */
    interface Embedder {
        double[] embed(Utt u, int maxFrames);
    }

    /**
     * Trains an x-vector network.
     *
     * @param train training utterances
     * @param speakers the number of training speakers
     * @param verifier which loss and pooling
     * @param steps training steps
     * @param seed the seed
     * @return the network
     */
    static SpeakerEmbedding trainXvector(List<Utt> train, int speakers, Verifier verifier, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SpeakerEmbedding net = new SpeakerEmbedding(MELS, 32, 32, verifier == Verifier.XVECTOR_ATTENTIVE_AAM, r);
        Var classes = Var.param(Tensor.normal(Shape.of(32, speakers), 0, 0.2f, r));
        Var w = Var.param(Tensor.scalar(10f));
        Var b = Var.param(Tensor.scalar(-5f));
        List<List<Utt>> bySpeaker = new ArrayList<>();
        for (int s = 0; s < speakers; s++) {
            bySpeaker.add(new ArrayList<>());
        }
        for (Utt u : train) {
            bySpeaker.get(u.speaker()).add(u);
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        List<Var> params = new ArrayList<>(net.parameters());
        if (verifier == Verifier.GE2E) {
            params.add(w);
            params.add(b);
        } else {
            params.add(classes);
        }
        for (int step = 0; step < steps; step++) {
            List<float[][]> crops = new ArrayList<>();
            int[] labels;
            Var loss;
            if (verifier == Verifier.GE2E) {
                int n = 8;
                int m = 4;
                labels = new int[n * m];
                for (int i = 0; i < n; i++) {
                    List<Utt> pool = bySpeaker.get(r.nextInt(speakers));
                    for (int j = 0; j < m; j++) {
                        crops.add(crop(pool.get(r.nextInt(pool.size())).features(), r));
                    }
                }
                Var e = net.forward(Var.of(SpeechFeatures.batch(crops)));
                loss = SpeakerEmbedding.ge2e(e, n, m, w.relu().add(1e-3f), b);
            } else {
                labels = new int[32];
                for (int i = 0; i < labels.length; i++) {
                    Utt u = train.get(r.nextInt(train.size()));
                    labels[i] = u.speaker();
                    crops.add(crop(u.features(), r));
                }
                Var e = net.forward(Var.of(SpeechFeatures.batch(crops)));
                loss = SpeakerEmbedding.aamSoftmax(e, classes, labels, 16f,
                        verifier == Verifier.XVECTOR_SOFTMAX ? 0f : 0.2f);
            }
            loss.backward();
            opt.step(params);
        }
        return net;
    }

    private static float[][] crop(float[][] f, SplittableRandom r) {
        int start = f.length > CROP ? r.nextInt(f.length - CROP + 1) : 0;
        return SpeechFeatures.window(f, start, CROP);
    }

    private static float[][] head(float[][] f, int maxFrames) {
        return maxFrames <= 0 || f.length <= maxFrames ? f : SpeechFeatures.window(f, 0, maxFrames);
    }

    /** GMM-UBM supervectors. */
    static final class GmmUbm implements Embedder {
        private final int k;
        private final double[][] means;
        private final double[][] vars;
        private final double[] logWeights;

        GmmUbm(List<Utt> train, int k, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            List<double[]> frames = new ArrayList<>();
            for (int i = 0; i < Math.min(train.size(), 200); i++) {
                float[][] m = mfcc(train.get(r.nextInt(train.size())).wave());
                for (int j = 0; j < 15; j++) {
                    frames.add(SpeechFeatures.toDouble(m[r.nextInt(m.length)]));
                }
            }
            GaussianMixture g = GaussianMixture.fit(frames.toArray(double[][]::new), k, 30, 1e-3, seed);
            this.k = g.components();
            int d = frames.get(0).length;
            means = new double[this.k][];
            vars = new double[this.k][d];
            logWeights = new double[this.k];
            for (int j = 0; j < this.k; j++) {
                means[j] = g.mean(j).clone();
                for (int i = 0; i < d; i++) {
                    vars[j][i] = Math.max(1e-4, g.covariance(j)[i][i]);
                }
                logWeights[j] = Math.log(Math.max(1e-12, g.weights()[j]));
            }
        }

        @Override
        public double[] embed(Utt u, int maxFrames) {
            float[][] f = head(mfcc(u.wave()), maxFrames);
            int d = means[0].length;
            double[] n = new double[k];
            double[][] sum = new double[k][d];
            for (float[] x : f) {
                double[] lp = new double[k];
                double max = -Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double s = logWeights[j];
                    for (int i = 0; i < d; i++) {
                        double z = x[i] - means[j][i];
                        s -= 0.5 * (z * z / vars[j][i] + Math.log(vars[j][i]));
                    }
                    lp[j] = s;
                    max = Math.max(max, s);
                }
                double total = 0;
                for (int j = 0; j < k; j++) {
                    lp[j] = Math.exp(lp[j] - max);
                    total += lp[j];
                }
                for (int j = 0; j < k; j++) {
                    double g = lp[j] / total;
                    n[j] += g;
                    for (int i = 0; i < d; i++) {
                        sum[j][i] += g * x[i];
                    }
                }
            }
            double relevance = 4;
            double[] sv = new double[k * d];
            for (int j = 0; j < k; j++) {
                for (int i = 0; i < d; i++) {
                    double adapted = (sum[j][i] + relevance * means[j][i]) / (n[j] + relevance);
                    sv[j * d + i] = (adapted - means[j][i]) * Math.exp(0.5 * logWeights[j]) / Math.sqrt(vars[j][i]);
                }
            }
            return sv;
        }
    }

    static float[][] mfcc(float[] wave) {
        return Mfcc.compute(wave, SpeakerVoices.RATE, 2 * HOP, HOP, MELS, 13, false);
    }

    /**
     * Speaker verification.
     *
     * @param trainSpeakers training speakers (100 in the item)
     * @param testSpeakers held-out speakers (30)
     * @param perSpeaker utterances per speaker
     * @param snrDb the SNR (20 or 5)
     * @param steps training steps of each network
     * @param maxFrames test utterance length cap in frames (0 for full; 31 frames are 0.5 s)
     * @param seed the seed
     * @param methods the systems
     * @return one row per system
     */
    public static List<VerificationRow> verification(int trainSpeakers, int testSpeakers, int perSpeaker,
            double snrDb, int steps, int maxFrames, long seed, Verifier... methods) {
        SplittableRandom r = new SplittableRandom(seed);
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(trainSpeakers + testSpeakers, seed);
        List<Utt> train = utterances(voices, 0, trainSpeakers, perSpeaker, snrDb, r);
        List<Utt> test = utterances(voices, trainSpeakers, testSpeakers, Math.max(3, perSpeaker / 2), snrDb, r);
        List<VerificationRow> rows = new ArrayList<>();
        for (Verifier v : methods) {
            Embedder emb;
            boolean plda = false;
            switch (v) {
                case MEAN_MFCC -> {
                    double[] global = new double[13];
                    for (Utt u : train) {
                        double[] m = SpeechFeatures.mean(mfcc(u.wave()));
                        for (int i = 0; i < 13; i++) {
                            global[i] += m[i] / train.size();
                        }
                    }
                    emb = (u, max) -> {
                        double[] m = SpeechFeatures.mean(head(mfcc(u.wave()), max));
                        for (int i = 0; i < m.length; i++) {
                            m[i] -= global[i];
                        }
                        return m;
                    };
                }
                case GMM_UBM -> emb = new GmmUbm(train, 16, seed);
                case DIGIT_ENCODER -> emb = digitEncoder(train, steps, seed);
                default -> {
                    SpeakerEmbedding net = trainXvector(train, trainSpeakers, v, steps, seed + v.ordinal());
                    emb = (u, max) -> net.embed(head(u.features(), max));
                    plda = true;
                }
            }
            List<double[]> e = new ArrayList<>();
            for (Utt u : test) {
                e.add(emb.embed(u, maxFrames));
            }
            int pairs = test.size() * (test.size() - 1) / 2;
            double[] cos = new double[pairs];
            double[] pl = new double[pairs];
            boolean[] target = new boolean[pairs];
            SpeakerEmbedding.Plda model = null;
            if (plda) {
                List<double[]> te = new ArrayList<>();
                int[] spk = new int[train.size()];
                for (int i = 0; i < train.size(); i++) {
                    te.add(emb.embed(train.get(i), 0));
                    spk[i] = train.get(i).speaker();
                }
                model = SpeakerEmbedding.Plda.fit(te, spk);
            }
            int p = 0;
            for (int i = 0; i < test.size(); i++) {
                for (int j = i + 1; j < test.size(); j++) {
                    cos[p] = SpeechFeatures.cosine(e.get(i), e.get(j));
                    if (model != null) {
                        pl[p] = model.score(e.get(i), e.get(j));
                    }
                    target[p] = test.get(i).speaker() == test.get(j).speaker();
                    p++;
                }
            }
            rows.add(new VerificationRow(v.name(), SpeakerEmbedding.equalErrorRate(cos, target),
                    SpeakerEmbedding.minDcf(cos, target, 0.05),
                    model != null ? SpeakerEmbedding.equalErrorRate(pl, target) : Double.NaN));
        }
        return rows;
    }

    private static Embedder digitEncoder(List<Utt> train, int steps, long seed) {
        int frames = 96;
        SpokenDigitRecognition.Recognizer rec = new SpokenDigitRecognition.Recognizer(
                SpokenDigitRecognition.Encoder.MLP, SpokenDigitRecognition.width(SpokenDigitRecognition.Features.LOG_MEL),
                frames, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        for (int s = 0; s < steps; s++) {
            List<float[][]> feats = new ArrayList<>();
            List<int[]> digits = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                Utt u = train.get(r.nextInt(train.size()));
                feats.add(SpokenDigitRecognition.features(SpokenDigitRecognition.Features.LOG_MEL, u.wave(), frames));
                digits.add(u.digits());
            }
            rec.trainCtc(feats, digits, false, r);
        }
        return (u, max) -> {
            float[] wave = u.wave();
            if (max > 0 && wave.length > max * HOP) {
                wave = java.util.Arrays.copyOf(wave, max * HOP);
            }
            int used = Math.min(frames, Math.max(1, wave.length / HOP));
            float[][] f = SpokenDigitRecognition.features(SpokenDigitRecognition.Features.LOG_MEL, wave, frames);
            float[] h = Var.noGrad(() -> rec.encode(List.<float[][]>of(f), false)).value().toArray();
            int width = h.length / frames;
            double[] m = new double[width];
            for (int t = 0; t < used; t++) {
                for (int i = 0; i < width; i++) {
                    m[i] += h[t * width + i] / used;
                }
            }
            return m;
        };
    }

    /**
     * EER against test utterance length for one system (10.6.5).
     *
     * @param verifier the system
     * @param lengths frame caps
     * @param steps training steps
     * @param seed the seed
     * @return per cap the EER
     */
    public static Map<Integer, Double> eerByLength(Verifier verifier, int[] lengths, int steps, long seed) {
        Map<Integer, Double> out = new LinkedHashMap<>();
        for (int len : lengths) {
            out.put(len, verification(100, 30, 8, 20, steps, len, seed, verifier).get(0).eer());
        }
        return out;
    }

    // ---------------------------------------------------------------- 10.6.8 diarisation

    /** Diarisation systems. */
    public enum Diarizer {
        /** Everything is one speaker (speech from the VAD). */
        ONE_SPEAKER,
        /** Mean-MFCC window embeddings, spectral clustering with the eigengap count. */
        MFCC_SPECTRAL,
        /** x-vector windows, agglomerative clustering with the eigengap count. */
        XVECTOR_AHC,
        /** x-vector windows, spectral clustering with the eigengap count. */
        XVECTOR_SPECTRAL,
        /** x-vector windows with oracle VAD and the oracle speaker count (spectral). */
        XVECTOR_ORACLE,
        /** End-to-end neural diarisation. */
        EEND
    }

    /**
     * A diarisation result.
     *
     * @param method the system
     * @param der the mean diarisation error rate
     * @param missed its missed-speech part
     * @param falseAlarm its false-alarm part
     * @param confusion its speaker-confusion part
     * @param overlap the measured fraction of overlapped speech frames
     */
    public record DiarizationRow(String method, double der, double missed, double falseAlarm, double confusion,
            double overlap) {
        @Override
        public String toString() {
            return String.format("%s DER %.3f (miss %.3f, FA %.3f, conf %.3f)", method, der, missed, falseAlarm,
                    confusion);
        }
    }

    /**
     * Diarisation of synthetic conversations at 15 dB SNR (white noise).
     *
     * @param speakers speakers per conversation (2 or 3)
     * @param overlapChance the chance that a turn overlaps the previous one
     * @param conversations test conversations
     * @param steps training steps of the x-vector and EEND networks
     * @param seed the seed
     * @param methods the systems
     * @return one row per system
     */
    public static List<DiarizationRow> diarization(int speakers, double overlapChance, int conversations, int steps,
            long seed, Diarizer... methods) {
        SplittableRandom r = new SplittableRandom(seed);
        int trainSpeakers = 60;
        List<SpeakerVoices.Voice> voices = SpeakerVoices.speakers(trainSpeakers + 30, seed);
        List<SpeakerVoices.Voice> trainVoices = voices.subList(0, trainSpeakers);
        List<SpeakerVoices.Voice> testVoices = voices.subList(trainSpeakers, voices.size());
        List<SpeakerVoices.Conversation> tests = new ArrayList<>();
        SplittableRandom tr = new SplittableRandom(seed + 77);
        for (int c = 0; c < conversations; c++) {
            List<SpeakerVoices.Voice> who = new ArrayList<>();
            for (int s = 0; s < speakers; s++) {
                who.add(testVoices.get((c * speakers + s) % testVoices.size()));
            }
            tests.add(SpeakerVoices.conversation(who, 3 * speakers, overlapChance, SpeakerVoices.Noise.WHITE, 15, tr));
        }
        double meanOverlap = 0;
        for (SpeakerVoices.Conversation c : tests) {
            meanOverlap += c.overlap() / tests.size();
        }
        SpeakerEmbedding xvec = null;
        Diarization.Eend eend = null;
        List<DiarizationRow> rows = new ArrayList<>();
        for (Diarizer m : methods) {
            if (m.name().startsWith("XVECTOR") && xvec == null) {
                xvec = trainXvector(utterances(trainVoices, 0, trainSpeakers, 4, 15, r), trainSpeakers,
                        Verifier.XVECTOR_AAM, steps, seed + 5);
            }
            if (m == Diarizer.EEND && eend == null) {
                eend = trainEend(trainVoices, steps, seed + 9);
            }
            double[] sum = new double[4];
            for (SpeakerVoices.Conversation c : tests) {
                float[][] feats = features(c.wave());
                int frames = Math.min(feats.length, c.frames());
                boolean[][] ref = new boolean[c.activity().length][];
                for (int s = 0; s < ref.length; s++) {
                    ref[s] = java.util.Arrays.copyOf(c.activity()[s], frames);
                }
                boolean[] speech;
                if (m == Diarizer.XVECTOR_ORACLE) {
                    speech = java.util.Arrays.copyOf(c.speech(), frames);
                } else {
                    double[] post = VoiceActivity.hmmHangover(VoiceActivity.sohn(c.wave(), HOP), 0.95);
                    speech = java.util.Arrays.copyOf(VoiceActivity.decide(post, 0.5), frames);
                }
                boolean[][] hyp;
                if (m == Diarizer.EEND) {
                    hyp = eend.diarize(SpeechFeatures.window(feats, 0, frames), 0.5);
                } else if (m == Diarizer.ONE_SPEAKER) {
                    hyp = new boolean[][] {speech};
                } else {
                    int window = 24;
                    List<Integer> starts = Diarization.windows(speech, window, 8);
                    List<double[]> emb = new ArrayList<>();
                    for (int s : starts) {
                        float[][] w = SpeechFeatures.window(feats, s, window);
                        if (m == Diarizer.MFCC_SPECTRAL) {
                            float[] seg = java.util.Arrays.copyOfRange(c.wave(), s * HOP, (s + window) * HOP);
                            emb.add(SpeechFeatures.mean(mfcc(seg)));
                        } else {
                            emb.add(xvec.embed(w));
                        }
                    }
                    int k = m == Diarizer.XVECTOR_ORACLE ? speakers : 0;
                    int[] labels;
                    if (starts.size() < 2) {
                        labels = new int[starts.size()];
                    } else if (m == Diarizer.XVECTOR_AHC) {
                        int est = Diarization.eigengap(sharpened(emb), 6);
                        labels = Diarization.agglomerative(emb, est, 0);
                    } else {
                        labels = Diarization.spectral(emb, k, seed);
                    }
                    int count = 1;
                    for (int l : labels) {
                        count = Math.max(count, l + 1);
                    }
                    hyp = Diarization.frameLabels(speech, starts, window, labels, count);
                }
                double[] e = Diarization.errorRate(hyp, ref);
                for (int i = 0; i < 4; i++) {
                    sum[i] += e[i] / tests.size();
                }
            }
            rows.add(new DiarizationRow(m.name(), sum[0], sum[1], sum[2], sum[3], meanOverlap));
        }
        return rows;
    }

    private static double[][] sharpened(List<double[]> emb) {
        double[][] a = Diarization.affinity(emb);
        for (double[] row : a) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.pow(row[j], 4);
            }
        }
        return a;
    }

    private static Diarization.Eend trainEend(List<SpeakerVoices.Voice> voices, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int slots = 3;
        Diarization.Eend net = new Diarization.Eend(MELS, 32, 2, slots, r);
        Optimizer opt = Optimizer.adam(2e-3f);
        int chunk = 160;
        for (int s = 0; s < steps; s++) {
            List<float[][]> feats = new ArrayList<>();
            float[] labels = new float[8 * chunk * slots];
            for (int i = 0; i < 8; i++) {
                int k = 2 + r.nextInt(2);
                List<SpeakerVoices.Voice> who = new ArrayList<>();
                while (who.size() < k) {
                    SpeakerVoices.Voice v = voices.get(r.nextInt(voices.size()));
                    if (!who.contains(v)) {
                        who.add(v);
                    }
                }
                SpeakerVoices.Conversation c = SpeakerVoices.conversation(who, 2 * k, 0.3 * r.nextDouble() * 1.5,
                        SpeakerVoices.Noise.WHITE, 10 + 10 * r.nextDouble(), r);
                float[][] f = features(c.wave());
                int frames = Math.min(f.length, c.frames());
                int start = frames > chunk ? r.nextInt(frames - chunk + 1) : 0;
                feats.add(SpeechFeatures.window(f, start, chunk));
                for (int t = 0; t < chunk; t++) {
                    int ft = start + t;
                    for (int sp = 0; sp < k; sp++) {
                        if (ft < frames && c.activity()[sp][ft]) {
                            labels[(i * chunk + t) * slots + sp] = 1;
                        }
                    }
                }
            }
            Var loss = net.loss(SpeechFeatures.batch(feats), Tensor.of(Shape.of(8, chunk, slots), labels));
            loss.backward();
            opt.step(net.parameters());
        }
        return net;
    }

    /**
     * Prints the long-budget runs.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (double snr : new double[] {20, 5}) {
            System.out.println("verification " + snr + " dB: "
                    + verification(100, 30, 8, snr, 1500, 0, 1, Verifier.values()));
        }
        System.out.println("EER by length (AAM): " + eerByLength(Verifier.XVECTOR_AAM, new int[] {31, 62, 125}, 1500,
                1));
        for (int k : new int[] {2, 3}) {
            for (double ov : new double[] {0, 0.25, 0.6}) {
                System.out.println("diarization " + k + " speakers, overlap chance " + ov + ": "
                        + diarization(k, ov, 12, 1500, 1, Diarizer.values()));
            }
        }
    }
}
