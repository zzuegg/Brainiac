package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContrastivePairLoss;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Neural audio fingerprinting (Chang et al.): a small convolutional encoder maps a segment's
 * log-mel spectrogram to a unit-norm embedding, trained contrastively with the InfoNCE loss
 * of {@link ContrastivePairLoss} so that a segment and a degraded copy of it (noise,
 * filtering, a small time shift, a speed change, reverb, supplied by the caller) are nearest
 * neighbours while other segments in the batch are pushed away. The database holds the
 * embeddings of overlapping segments of every reference in a cosine {@link KnnIndex}; a query
 * embeds its own segments, and each segment's top hits vote for their tracks.
 *
 * <p>Paper: Chang, Lee, Park, Jang, Kim, Kim, Choi, "Neural Audio Fingerprint for
 * High-specific Audio Retrieval based on Contrastive Learning" (ICASSP 2021),
 * <a href="https://arxiv.org/abs/2010.11910">arXiv:2010.11910</a>.
 */
public final class NeuralFingerprint {

    /** Mel bands. */
    public static final int MELS = 32;
    /** STFT size. */
    public static final int N_FFT = 256;
    /** Hop. */
    public static final int HOP = 128;

    private final int rate;
    private final int segment;
    private final int frames;
    private final int dim;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var proj;
    private final List<Var> params;
    private final ContrastivePairLoss loss = ContrastivePairLoss.clip(0.1f);
    private KnnIndex index;
    private final List<Integer> owners = new ArrayList<>();

    /**
     * Creates an untrained encoder.
     *
     * @param rate the sample rate
     * @param segment samples per segment
     * @param dim the embedding size
     * @param seed the seed
     */
    public NeuralFingerprint(int rate, int segment, int dim, long seed) {
        this.rate = rate;
        this.segment = segment;
        this.frames = segment / HOP + 1;
        this.dim = dim;
        SplittableRandom r = new SplittableRandom(seed);
        w1 = VisionOps.convWeight(r, 8, 1, 3);
        b1 = VisionOps.zeros(8);
        w2 = VisionOps.convWeight(r, 16, 8, 3);
        b2 = VisionOps.zeros(16);
        int flat = 16 * ((frames + 3) / 4) * (MELS / 4);
        proj = Var.param(Tensor.normal(Shape.of(flat, dim), 0, (float) Math.sqrt(1.0 / flat), r));
        params = List.of(w1, b1, w2, b2, proj);
        index = KnnIndex.create(dim, 1, KnnIndex.Metric.COSINE);
    }

    /** @return samples per segment */
    public int segment() {
        return segment;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return params;
    }

    private float[] features(float[] x) {
        float[][] m = MusicFeatures.logMel(x, rate, N_FFT, HOP, MELS);
        float[] v = new float[frames * MELS];
        for (int t = 0; t < frames && t < m.length; t++) {
            for (int j = 0; j < MELS; j++) {
                v[t * MELS + j] = (m[t][j] + 6f) * 0.25f;
            }
        }
        return v;
    }

    /**
     * Embeds segments.
     *
     * @param segments signals of {@link #segment()} samples
     * @return unit-norm embeddings {@code [n, dim]}
     */
    public Var embed(List<float[]> segments) {
        int n = segments.size();
        float[] v = new float[n * frames * MELS];
        for (int i = 0; i < n; i++) {
            System.arraycopy(features(segments.get(i)), 0, v, i * frames * MELS, frames * MELS);
        }
        Var x = Var.of(Tensor.of(Shape.of(n, 1, frames, MELS), v));
        Var h = VisionOps.conv(x, w1, b1, 2, 1).relu();
        h = VisionOps.conv(h, w2, b2, 2, 1).relu();
        return h.reshape(n, -1).matmul(proj).normalize(1, 1e-6f);
    }

    private static float[] cut(float[] x, int start, int len) {
        float[] s = new float[len];
        for (int i = 0; i < len; i++) {
            int j = start + i;
            s[i] = j >= 0 && j < x.length ? x[j] : 0;
        }
        return s;
    }

    /**
     * Trains contrastively on random segments of reference recordings.
     *
     * @param references the recordings
     * @param degrade makes a degraded view of a segment ({@code segment, random -> view} of
     *     the same length)
     * @param steps steps
     * @param batch pairs per step
     * @param lr learning rate
     * @param seed the seed
     * @return the last loss
     */
    public float train(List<float[]> references, BiFunction<float[], SplittableRandom, float[]> degrade, int steps,
            int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        float last = 0;
        for (int s = 0; s < steps; s++) {
            List<float[]> a = new ArrayList<>();
            List<float[]> b = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                float[] ref = references.get(r.nextInt(references.size()));
                int start = r.nextInt(Math.max(1, ref.length - segment));
                a.add(cut(ref, start, segment));
                int shift = r.nextInt(segment / 8 + 1) - segment / 16;
                b.add(degrade.apply(cut(ref, start + shift, segment), r));
            }
            params.forEach(Var::zeroGrad);
            Var l = loss.compute(embed(a), embed(b));
            last = l.value().item();
            l.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        return last;
    }

    /**
     * Builds the database.
     *
     * @param references the recordings; track ids are their positions
     * @param hop samples between database segments
     */
    public void index(List<float[]> references, int hop) {
        index = KnnIndex.create(dim, 1, KnnIndex.Metric.COSINE);
        owners.clear();
        for (int t = 0; t < references.size(); t++) {
            float[] ref = references.get(t);
            List<float[]> segs = new ArrayList<>();
            for (int start = 0; start + segment <= ref.length; start += hop) {
                segs.add(cut(ref, start, segment));
            }
            float[] e = Var.noGrad(() -> embed(segs)).value().toArray();
            for (int i = 0; i < segs.size(); i++) {
                index.add(java.util.Arrays.copyOfRange(e, i * dim, (i + 1) * dim), new float[] {t});
                owners.add(t);
            }
        }
    }

    /**
     * Identifies a query by segment-wise nearest-neighbour votes.
     *
     * @param query the signal (at least one segment long)
     * @param hop samples between query segments
     * @param k neighbours per segment
     * @return the best track
     */
    public int identify(float[] query, int hop, int k) {
        List<float[]> segs = new ArrayList<>();
        for (int start = 0; start + segment <= query.length; start += hop) {
            segs.add(cut(query, start, segment));
        }
        if (segs.isEmpty()) {
            segs.add(cut(query, 0, segment));
        }
        float[] e = Var.noGrad(() -> embed(segs)).value().toArray();
        Map<Integer, Double> votes = new HashMap<>();
        for (int i = 0; i < segs.size(); i++) {
            KnnIndex.Hit[] hits = index.search(java.util.Arrays.copyOfRange(e, i * dim, (i + 1) * dim), k);
            for (KnnIndex.Hit h : hits) {
                votes.merge(owners.get(h.row()), (double) Math.max(0, h.score()), Double::sum);
            }
        }
        int best = -1;
        double bestV = -1;
        for (Map.Entry<Integer, Double> v : votes.entrySet()) {
            if (v.getValue() > bestV) {
                bestV = v.getValue();
                best = v.getKey();
            }
        }
        return best;
    }
}
