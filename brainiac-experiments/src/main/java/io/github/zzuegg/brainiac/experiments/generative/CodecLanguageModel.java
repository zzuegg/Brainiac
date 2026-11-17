package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.CodebookPatterns;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A MusicGen-style codec language model (Copet et al., <a href="https://arxiv.org/abs/2306.05284">2306.05284</a>)
 * over the tokens of {@link ToyAudioCodec}, a residual-VQ audio codec with {@code K} code
 * streams. Clips are continuous tones cut into frames; each frame becomes {@code K} codes.
 * Two causal transformers of the same width and depth model the code streams:
 *
 * <ul>
 *   <li>the <b>delay</b> pattern ({@link CodebookPatterns#delay}): one position per step, the
 *       input is the sum of the {@code K} codebook embeddings of the previous step, and
 *       {@code K} heads predict all codebooks at once: {@code T + K - 1} positions;</li>
 *   <li>the <b>flattened</b> baseline ({@link CodebookPatterns#flatten}): every code is its
 *       own position, {@code T K} positions.</li>
 * </ul>
 *
 * <p>Scored by held-out negative log-likelihood per code, and by continuing clips from their
 * first frames (greedy): the share of continued codes that match the clip's own, the SNR of
 * the decoded continuation against the decoded true codes, and the network calls needed.
 */
@SuppressWarnings("exports")
public final class CodecLanguageModel {
    private static final int FRAME = 16;
    private static final int CODES = 16;
    private final int stages;
    private final int frames;
    private final ToyAudioCodec codec;
    private final int width = 32;
    private final Var delayEmbedding;
    private final Model delayNet;
    private final Model flatNet;
    private final SplittableRandom random;

    private CodecLanguageModel(int stages, int frames, long seed) {
        this.stages = stages;
        this.frames = frames;
        this.random = new SplittableRandom(seed);
        codec = ToyAudioCodec.of(FRAME, 8, stages, CODES, seed);
        delayEmbedding = Var.param(Tensor.normal(Shape.of(stages * (CODES + 2), width), 0, 0.3f, random));
        Model.Builder b = Model.sequential().input(frames + stages - 1, width).positionalEncoding();
        for (int i = 0; i < 2; i++) {
            b.add(TransformerBlock.of(2).causal(true).rotary(100f));
        }
        delayNet = b.dense(stages * CODES).seed(seed + 1).build();
        flatNet = TokenModels.transformer(frames * stages, stages * CODES + 1, width, 2, true, seed + 2);
    }

    /**
     * Creates the codec and both language models, untrained.
     *
     * @param stages the RVQ stages {@code K}
     * @param frames frames per clip {@code T}
     * @param seed the seed
     * @return the experiment
     */
    public static CodecLanguageModel of(int stages, int frames, long seed) {
        return new CodecLanguageModel(stages, frames, seed);
    }

    /**
     * Synthesizes clips: one or two steady partials with random period, amplitude and phase.
     *
     * @param n the number of clips
     * @param r the random source
     * @return {@code [clip][frame][sample]}
     */
    public float[][][] clips(int n, SplittableRandom r) {
        float[][][] out = new float[n][frames][FRAME];
        for (int i = 0; i < n; i++) {
            int partials = 1 + r.nextInt(2);
            for (int p = 0; p < partials; p++) {
                double period = 6 + r.nextDouble() * 24;
                double amp = 0.3 + 0.4 * r.nextDouble();
                double phase = r.nextDouble() * 2 * Math.PI;
                for (int s = 0; s < frames * FRAME; s++) {
                    out[i][s / FRAME][s % FRAME] += (float) (amp * Math.sin(2 * Math.PI * s / period + phase));
                }
            }
        }
        return out;
    }

    private float[][] allFrames(float[][][] clips) {
        List<float[]> f = new ArrayList<>();
        for (float[][] c : clips) {
            f.addAll(List.of(c));
        }
        return f.toArray(float[][]::new);
    }

    /**
     * Trains the codec on frames of fresh clips.
     *
     * @param steps training steps of 16 clips
     * @return the SNR in dB of a round trip through all stages, on fresh clips
     */
    public double trainCodec(int steps) {
        codec.restart(allFrames(clips(64, random)));
        for (int s = 0; s < steps; s++) {
            codec.trainStep(allFrames(clips(16, random)));
            if (s % 100 == 99 && s < steps - 100) {
                codec.restart(allFrames(clips(64, random)));
            }
        }
        return codec.snr(allFrames(clips(64, random)), stages);
    }

    /**
     * Encodes clips to code streams.
     *
     * @param clips clips
     * @return {@code [clip][K][T]}
     */
    public int[][][] encode(float[][][] clips) {
        int[][][] out = new int[clips.length][][];
        for (int i = 0; i < clips.length; i++) {
            out[i] = codec.encode(clips[i], stages);
        }
        return out;
    }

    private Var delayLogits(int[][][] delayed, boolean training) {
        int b = delayed.length;
        int steps = delayed[0][0].length;
        float[] ids = new float[b * steps * stages];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < steps; s++) {
                for (int k = 0; k < stages; k++) {
                    int prev = s == 0 ? CODES + 1 : delayed[i][k][s - 1];   // the start token, else the step before
                    ids[(i * steps + s) * stages + k] = k * (CODES + 2) + prev;
                }
            }
        }
        Var x = delayEmbedding.gather(0, Tensor.of(Shape.of(ids.length), ids)).reshape(b, steps, stages, width)
                .sum(2, false);
        return delayNet.forward(x, training).reshape(b * steps * stages, CODES);
    }

    private Var delayLoss(int[][][] codes, boolean training) {
        int b = codes.length;
        int[][][] delayed = new int[b][][];
        for (int i = 0; i < b; i++) {
            delayed[i] = CodebookPatterns.delay(codes[i], CODES);
        }
        int steps = delayed[0][0].length;
        float[] target = new float[b * steps * stages];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < steps; s++) {
                for (int k = 0; k < stages; k++) {
                    int t = delayed[i][k][s];
                    target[(i * steps + s) * stages + k] = t == CODES ? -1 : t;   // padding is not scored
                }
            }
        }
        return Functional.crossEntropy(delayLogits(delayed, training), Tensor.of(Shape.of(target.length), target));
    }

    private Var flatLoss(int[][][] codes) {
        int[][] flat = new int[codes.length][];
        for (int i = 0; i < codes.length; i++) {
            flat[i] = CodebookPatterns.flatten(codes[i], CODES);
        }
        return TokenModels.autoregressiveLoss(flatNet, flat, stages * CODES, stages * CODES + 1);
    }

    /**
     * Trains both language models on the codes of fresh clips.
     *
     * @param steps optimizer steps of 32 clips for each model
     */
    public void trainLanguageModels(int steps) {
        Optimizer a = Optimizer.adam(3e-3f);
        Optimizer b = Optimizer.adam(3e-3f);
        List<Var> delayParams = new ArrayList<>(delayNet.parameters());
        delayParams.add(delayEmbedding);
        for (int s = 0; s < steps; s++) {
            int[][][] codes = encode(clips(32, random));
            Var l = delayLoss(codes, true);
            l.backward();
            a.step(delayParams);
            Var f = flatLoss(codes);
            f.backward();
            b.step(flatNet.parameters());
        }
    }

    /**
     * Held-out negative log-likelihood per code.
     *
     * @param seed the evaluation seed
     * @return {@code {delay pattern, flattened}} in nats
     */
    public double[] nll(long seed) {
        int[][][] codes = encode(clips(128, new SplittableRandom(seed)));
        return new double[] {delayLoss(codes, false).value().item(), flatLoss(codes).value().item()};
    }

    private static int argmax(float[] v, int from, int to) {
        int best = from;
        for (int i = from + 1; i < to; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Continues clips greedily from their first frames with both models.
     *
     * @param prefix the frames given
     * @param seed the evaluation seed
     * @return {@code {delay code accuracy, delay SNR dB, delay calls, flat code accuracy, flat SNR dB, flat calls}}
     */
    public double[] continuation(int prefix, long seed) {
        int n = 64;
        int[][][] truth = encode(clips(n, new SplittableRandom(seed)));
        // delay pattern: fill the known corner, then one call per remaining step
        int steps = frames + stages - 1;
        int[][][] delayed = new int[n][][];
        for (int i = 0; i < n; i++) {
            delayed[i] = CodebookPatterns.delay(truth[i], CODES);
            for (int k = 0; k < stages; k++) {
                for (int t = prefix; t < frames; t++) {
                    delayed[i][k][t + k] = CODES;
                }
            }
        }
        int delayCalls = 0;
        for (int s = prefix; s < steps; s++) {
            float[] logits = delayLogits(delayed, false).value().toArray();
            delayCalls++;
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < stages; k++) {
                    int t = s - k;
                    if (t >= prefix && t < frames) {
                        int off = ((i * steps + s) * stages + k) * CODES;
                        delayed[i][k][s] = argmax(logits, off, off + CODES) - off;
                    }
                }
            }
        }
        int[][][] delayOut = new int[n][][];
        for (int i = 0; i < n; i++) {
            delayOut[i] = CodebookPatterns.undelay(delayed[i]);
        }
        // flattened: one call per code
        int len = frames * stages;
        int vocab = stages * CODES + 1;
        int[][] flat = new int[n][];
        for (int i = 0; i < n; i++) {
            flat[i] = CodebookPatterns.flatten(truth[i], CODES);
        }
        int flatCalls = 0;
        for (int pos = prefix * stages; pos < len; pos++) {
            int[][] in = new int[n][len];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < len; j++) {
                    in[i][j] = j == 0 ? vocab - 1 : j <= pos ? flat[i][j - 1] : vocab - 1;
                }
            }
            float[] logits = flatNet.forward(Var.of(TokenModels.ids(in)), false).value().toArray();
            flatCalls++;
            int q = pos % stages;
            for (int i = 0; i < n; i++) {
                int off = (i * len + pos) * vocab;
                flat[i][pos] = argmax(logits, off + q * CODES, off + (q + 1) * CODES) - off;
            }
        }
        int[][][] flatOut = new int[n][][];
        for (int i = 0; i < n; i++) {
            flatOut[i] = CodebookPatterns.unflatten(flat[i], stages, CODES);
        }
        double[] d = score(truth, delayOut, prefix);
        double[] f = score(truth, flatOut, prefix);
        return new double[] {d[0], d[1], delayCalls, f[0], f[1], flatCalls};
    }

    /** Code accuracy and decoded SNR of the continued frames. */
    private double[] score(int[][][] truth, int[][][] out, int prefix) {
        int hit = 0;
        int total = 0;
        double signal = 0;
        double noise = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int k = 0; k < stages; k++) {
                for (int t = prefix; t < frames; t++) {
                    hit += truth[i][k][t] == out[i][k][t] ? 1 : 0;
                    total++;
                }
            }
            float[][] a = codec.decode(truth[i]);
            float[][] b = codec.decode(out[i]);
            for (int t = prefix; t < frames; t++) {
                for (int s = 0; s < FRAME; s++) {
                    signal += a[t][s] * a[t][s];
                    noise += (a[t][s] - b[t][s]) * (a[t][s] - b[t][s]);
                }
            }
        }
        return new double[] {(double) hit / total, 10 * Math.log10(signal / Math.max(noise, 1e-12))};
    }
}
