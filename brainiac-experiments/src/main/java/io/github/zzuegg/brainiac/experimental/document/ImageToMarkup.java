package io.github.zzuegg.brainiac.experimental.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * OCR-free document parsing in the style of Donut (Kim et al., ECCV 2022): instead of
 * running OCR and then parsing its tokens, a vision transformer encodes the page image
 * ({@link PatchEmbedding} plus core {@link TransformerBlock}s) and an autoregressive text
 * decoder (the decoder of {@link TransformerSeq2Seq}, cross-attending to the patch tokens)
 * emits the structured output as a token sequence, for example
 * {@code <NAME> B O B <TOTAL> 4 2 <eos>}. Training runs in two phases on the same decoder:
 * first "pseudo-OCR" reading (emit all the page's text in reading order), then the parsing
 * task. There is no OCR engine, so no OCR error to propagate, but the model must learn to
 * read. {@link #pretrainReading} runs the reading phase as a separate stage (the paper's
 * SynthDoG pretraining) and {@link #checkpoint} / {@link #restore} let one pretrained model
 * seed several fine-tuning runs.
 *
 * <p>Token id 0 is end-of-sequence; targets are padded with -1; the decoder's
 * begin-of-sequence id is {@code vocab}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2111.15664">arXiv:2111.15664</a>.
 */
public final class ImageToMarkup {

    /**
     * A training batch.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param targets per page its token ids, ending with end-of-sequence
     */
    public record Batch(Tensor images, int[][] targets) {

        /**
         * Copies the targets.
         *
         * @param images the pages
         * @param targets the token ids
         */
        public Batch {
            targets = targets.clone();
        }

        @Override
        public int[][] targets() {
            return targets.clone();
        }
    }

    private final int vocab;
    private final PatchEmbedding patches;
    private final List<Layer> encoder = new ArrayList<>();
    private final TransformerSeq2Seq decoder;

    /**
     * Creates the model.
     *
     * @param height page height, a multiple of {@code patch}
     * @param width page width, a multiple of {@code patch}
     * @param patch the patch side
     * @param dim the model width
     * @param encoderLayers vision encoder blocks
     * @param decoderLayers text decoder blocks
     * @param vocab output tokens, including end-of-sequence (0)
     * @param seed the seed
     */
    public ImageToMarkup(int height, int width, int patch, int dim, int encoderLayers, int decoderLayers, int vocab,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.patches = new PatchEmbedding(1, height, width, patch, dim, false, true, r);
        for (int i = 0; i < encoderLayers; i++) {
            encoder.add(TransformerBlock.of(4).ffnSize(2 * dim).create(Shape.of(8, dim), r));
        }
        this.decoder = TransformerSeq2Seq.of(2, vocab, dim, 4, decoderLayers, r.nextLong());
    }

    /**
     * Encodes pages.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param training whether training
     * @return the patch tokens {@code [batch, patches, dim]}
     */
    public Var memory(Tensor images, boolean training) {
        Var x = patches.forward(Var.of(images), training);
        for (Layer l : encoder) {
            x = l.forward(x, training);
        }
        return x;
    }

    /**
     * Teacher-forced cross-entropy of target sequences.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param targets {@code [batch][length]} token ids, -1 padding not scored
     * @return the mean loss
     */
    public Var loss(Tensor images, int[][] targets) {
        int b = targets.length;
        int len = targets[0].length;
        int[][] in = new int[b][len];
        float[] ids = new float[b * len];
        for (int i = 0; i < b; i++) {
            in[i][0] = vocab;
            for (int t = 0; t < len; t++) {
                if (t > 0) {
                    in[i][t] = Math.max(0, targets[i][t - 1]);
                }
                ids[i * len + t] = targets[i][t];
            }
        }
        Var logits = decoder.decode(memory(images, true), in, true).reshape(b * len, vocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(b * len), ids));
    }

    /**
     * Greedy decoding until end-of-sequence or {@code length} tokens.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param length the most tokens
     * @return per page its tokens, without the end-of-sequence
     */
    public int[][] generate(Tensor images, int length) {
        return Var.noGrad(() -> {
            int b = images.shape().dim(0);
            Var mem = memory(images, false);
            int[][] out = new int[b][length];
            boolean[] done = new boolean[b];
            int[] lens = new int[b];
            for (int t = 0; t < length; t++) {
                int[][] in = new int[b][t + 1];
                for (int i = 0; i < b; i++) {
                    in[i][0] = vocab;
                    System.arraycopy(out[i], 0, in[i], 1, t);
                }
                float[] logits = decoder.decode(mem, in, false).slice(1, t, t + 1).value().toArray();
                for (int i = 0; i < b; i++) {
                    int best = 0;
                    for (int k = 1; k < vocab; k++) {
                        best = logits[i * vocab + k] > logits[i * vocab + best] ? k : best;
                    }
                    out[i][t] = best;
                    if (!done[i]) {
                        if (best == 0) {
                            done[i] = true;
                        } else {
                            lens[i]++;
                        }
                    }
                }
            }
            int[][] res = new int[b][];
            for (int i = 0; i < b; i++) {
                res[i] = java.util.Arrays.copyOf(out[i], lens[i]);
            }
            return res;
        });
    }

    /**
     * The reading-pretraining stage: trains the whole model to emit each page's full text in
     * reading order (the caller's targets), before any parsing.
     *
     * @param steps optimiser steps
     * @param batches the batch of each step (targets need not be padded)
     * @param optimizer the optimiser
     * @return the mean loss of the last tenth of the steps
     */
    public double pretrainReading(int steps, IntFunction<Batch> batches, Optimizer optimizer) {
        return fineTune(steps, batches, optimizer);
    }

    /**
     * Trains on the caller's targets (parsing, or anything else).
     *
     * @param steps optimiser steps
     * @param batches the batch of each step (targets need not be padded)
     * @param optimizer the optimiser
     * @return the mean loss of the last tenth of the steps
     */
    public double fineTune(int steps, IntFunction<Batch> batches, Optimizer optimizer) {
        double tail = 0;
        int counted = 0;
        for (int s = 0; s < steps; s++) {
            Batch b = batches.apply(s);
            Var loss = loss(b.images(), padded(b.targets()));
            if (s >= steps - Math.max(1, steps / 10)) {
                tail += loss.value().item();
                counted++;
            }
            loss.backward();
            optimizer.step(parameters());
        }
        return counted == 0 ? Double.NaN : tail / counted;
    }

    private static int[][] padded(int[][] rows) {
        int len = 0;
        for (int[] r : rows) {
            len = Math.max(len, r.length);
        }
        int[][] out = new int[rows.length][len];
        for (int i = 0; i < rows.length; i++) {
            java.util.Arrays.fill(out[i], -1);
            System.arraycopy(rows[i], 0, out[i], 0, rows[i].length);
        }
        return out;
    }

    /**
     * A copy of every trainable value, for {@link #restore}.
     *
     * @return the values in {@link #parameters()} order
     */
    public List<Tensor> checkpoint() {
        List<Tensor> out = new ArrayList<>();
        for (Var v : parameters()) {
            out.add(Tensor.of(v.value().shape(), v.value().toArray()));
        }
        return out;
    }

    /**
     * Loads values saved by {@link #checkpoint} of a model of the same configuration.
     *
     * @param values the values
     */
    public void restore(List<Tensor> values) {
        List<Var> params = parameters();
        if (params.size() != values.size()) {
            throw new IllegalArgumentException("checkpoint has " + values.size() + " values, model " + params.size());
        }
        for (int i = 0; i < params.size(); i++) {
            Tensor t = values.get(i);
            params.get(i).assign(Tensor.of(t.shape(), t.toArray()));
        }
    }

    /** @return the trainable values (patch embedding, encoder, decoder) */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(patches.parameters());
        for (Layer l : encoder) {
            p.addAll(l.parameters());
        }
        Set<Var> unused = new HashSet<>(decoder.encoderParameters());
        for (Var v : decoder.parameters()) {
            if (!unused.contains(v)) {
                p.add(v);
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "ImageToMarkup(" + encoder.size() + " encoder blocks, " + decoder + ")";
    }
}
