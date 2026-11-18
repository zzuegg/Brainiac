package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Self-supervised masked speech pretraining for any frame encoder
 * {@code [batch, frames, features] -> [batch, frames, width]}. Spans of input frames are
 * replaced by a learned mask vector and the encoder must recover what was there:
 *
 * <ul>
 *   <li>{@link Mode#HUBERT}: predict, at masked frames only, the k-means cluster of the
 *       original frame (clusters fitted once on unlabelled features, as HuBERT's first
 *       iteration fits them on MFCCs), by cross-entropy through a linear head;</li>
 *   <li>{@link Mode#WAV2VEC2}: a Gumbel-softmax quantizer (one codebook, straight-through
 *       hard codes) turns the unmasked input frames into targets; at each masked frame the
 *       projected context must pick its own quantized frame among all frames of the
 *       utterance (InfoNCE with cosine similarity and temperature 0.1), plus a diversity
 *       penalty, the negative entropy of the batch-averaged code distribution.</li>
 * </ul>
 * After pretraining, the encoder's parameters are fine-tuned by the caller.
 *
 * <p>Papers: Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked
 * Prediction of Hidden Units" (2021), <a href="https://arxiv.org/abs/2106.07447">arXiv:2106.07447</a>;
 * Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
 * Representations" (NeurIPS 2020), <a href="https://arxiv.org/abs/2006.11477">arXiv:2006.11477</a>.
 */
public final class MaskedSpeechPretraining {

    /** The pretraining objective. */
    public enum Mode {
        /** Masked prediction of k-means units. */
        HUBERT,
        /** Contrastive prediction of Gumbel-quantized frames. */
        WAV2VEC2
    }

    private static final int CLUSTERS = 24;
    private static final int CODES = 32;
    private static final int CODE_DIM = 24;

    private final Mode mode;
    private final Function<Var, Var> encoder;
    private final List<Var> parameters = new ArrayList<>();
    private final int features;
    private final long seed;
    private final Var maskVector;
    private final Var head;
    private final Var quantizer;
    private final Var codebook;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private KMeans clusters;

    /**
     * Creates the objective.
     *
     * @param mode the objective
     * @param encoder the frame encoder, {@code [b, t, features] -> [b, t, width]}
     * @param encoderParameters the encoder's trainable values (updated in place)
     * @param features input features per frame
     * @param width the encoder's output width
     * @param seed the seed
     */
    public MaskedSpeechPretraining(Mode mode, Function<Var, Var> encoder, List<Var> encoderParameters,
            int features, int width, long seed) {
        this.mode = mode;
        this.encoder = encoder;
        this.features = features;
        this.seed = seed;
        SplittableRandom r = new SplittableRandom(seed);
        maskVector = Var.param(Tensor.normal(Shape.of(features), 0, 0.1f, r));
        int out = mode == Mode.HUBERT ? CLUSTERS : CODE_DIM;
        head = Var.param(Tensor.normal(Shape.of(width, out), 0, (float) Math.sqrt(1.0 / width), r));
        quantizer = Var.param(Tensor.normal(Shape.of(features, CODES), 0, (float) Math.sqrt(1.0 / features), r));
        codebook = Var.param(Tensor.normal(Shape.of(CODES, CODE_DIM), 0, 1f, r));
        parameters.addAll(encoderParameters);
        parameters.add(maskVector);
        parameters.add(head);
        if (mode == Mode.WAV2VEC2) {
            parameters.add(quantizer);
            parameters.add(codebook);
        }
    }

    /**
     * Fits the HuBERT targets: k-means over all frames (a no-op for wav2vec 2.0).
     *
     * @param unlabelled per utterance {@code [frames][features]}
     */
    public void fitTargets(List<float[][]> unlabelled) {
        if (mode != Mode.HUBERT) {
            return;
        }
        List<double[]> rows = new ArrayList<>();
        for (float[][] u : unlabelled) {
            for (float[] f : u) {
                double[] d = new double[f.length];
                for (int k = 0; k < f.length; k++) {
                    d[k] = f[k];
                }
                rows.add(d);
            }
        }
        clusters = KMeans.fit(rows.toArray(new double[0][]), CLUSTERS, KMeans.Init.PLUS_PLUS, 1, 30, seed);
    }

    /**
     * Draws span masks: spans of {@code span} frames starting at random frames until about
     * {@code fraction} of the frames are covered.
     *
     * @param batch utterances
     * @param frames frames per utterance
     * @param span span length
     * @param fraction target masked fraction
     * @param r the random source
     * @return {@code [batch][frames]}, true where masked
     */
    public static boolean[][] spanMask(int batch, int frames, int span, double fraction, SplittableRandom r) {
        boolean[][] m = new boolean[batch][frames];
        for (int b = 0; b < batch; b++) {
            int covered = 0;
            int guard = 0;
            while (covered < fraction * frames && guard++ < 100) {
                int s = r.nextInt(Math.max(1, frames - span + 1));
                for (int t = s; t < Math.min(frames, s + span); t++) {
                    if (!m[b][t]) {
                        m[b][t] = true;
                        covered++;
                    }
                }
            }
        }
        return m;
    }

    /**
     * One pretraining step.
     *
     * @param batch per utterance {@code [frames][features]}, all of one length
     * @param r the random source of the masks (and the Gumbel noise)
     * @return the loss
     */
    public float trainStep(List<float[][]> batch, SplittableRandom r) {
        int b = batch.size();
        int t = batch.get(0).length;
        float[] x = new float[b * t * features];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                System.arraycopy(batch.get(i)[j], 0, x, (i * t + j) * features, features);
            }
        }
        boolean[][] mask = spanMask(b, t, 4, 0.4, r);
        float[] keep = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                keep[i * t + j] = mask[i][j] ? 0 : 1;
            }
        }
        Tensor clean = Tensor.of(Shape.of(b, t, features), x);
        Tensor keepT = Tensor.of(Shape.of(b, t, 1), keep);
        Tensor dropT = Tensor.of(Shape.of(b, t, 1), keep).neg().add(1f);
        Var masked = Var.of(clean).mul(keepT).add(maskVector.reshape(1, 1, features).mul(dropT));
        Var h = encoder.apply(masked);
        int width = h.shape().dim(2);
        Var loss;
        if (mode == Mode.HUBERT) {
            if (clusters == null) {
                throw new IllegalStateException("fitTargets first");
            }
            float[] ids = new float[b * t];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < t; j++) {
                    double[] row = new double[features];
                    for (int k = 0; k < features; k++) {
                        row[k] = batch.get(i)[j][k];
                    }
                    ids[i * t + j] = mask[i][j] ? KMeans.nearest(clusters.centres(), row) : -1;
                }
            }
            Var logits = h.reshape(b * t, width).matmul(head);
            loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(b * t), ids));
        } else {
            Var codeLogits = Var.of(clean).matmul(quantizer);
            Var hard = Functional.gumbelSoftmax(codeLogits, 0.5f, true, r);
            Var q = normalise(hard.matmul(codebook));
            Var c = normalise(h.matmul(head));
            Var sim = c.matmul(q.transpose()).mul(10f).reshape(b * t, t);
            float[] ids = new float[b * t];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < t; j++) {
                    ids[i * t + j] = mask[i][j] ? j : -1;
                }
            }
            Var contrastive = Functional.crossEntropy(sim, Tensor.of(Shape.of(b * t), ids));
            Var probs = codeLogits.sub(codeLogits.logsumexp(2, true)).exp().reshape(b * t, CODES).mean(0, false);
            Var negEntropy = probs.mul(probs.add(1e-6f).log()).sum();
            loss = contrastive.add(negEntropy.mul(0.1f));
        }
        float v = loss.value().item();
        loss.backward();
        optimizer.step(parameters);
        return v;
    }

    private static Var normalise(Var v) {
        return v.div(v.square().sum(-1, true).add(1e-6f).sqrt());
    }

    /**
     * Code usage of the wav2vec 2.0 quantizer, a collapse check.
     *
     * @param frames per utterance {@code [frames][features]}
     * @return the number of distinct codes chosen by argmax over all frames
     */
    public int codesUsed(List<float[][]> frames) {
        java.util.Set<Integer> used = new java.util.HashSet<>();
        float[] w = quantizer.value().toArray();
        for (float[][] u : frames) {
            for (float[] f : u) {
                int best = 0;
                double bestV = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < CODES; c++) {
                    double s = 0;
                    for (int k = 0; k < features; k++) {
                        s += f[k] * w[k * CODES + c];
                    }
                    if (s > bestV) {
                        bestV = s;
                        best = c;
                    }
                }
                used.add(best);
            }
        }
        return used.size();
    }
}
