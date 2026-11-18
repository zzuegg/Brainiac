package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * LLaVA's two-stage recipe for a vision-to-LM bridge (Liu et al., NeurIPS 2023 and CVPR
 * 2024) around a pretrained {@link FusionDecoder}.
 *
 * <ol>
 *   <li>{@link #align}: feature alignment. Only the bridge trains, on caption-style data
 *       (for example an image and its description), the LM stays frozen.</li>
 *   <li>{@link #instruct}: the bridge and the LM train together on the task (question
 *       answering). {@link LmMode#FROZEN} keeps the LM fixed (wave 9's setting),
 *       {@link LmMode#LORA} adds {@link LoraAdapterDense} adapters and {@link LmMode#FULL}
 *       unfreezes every LM weight (the caller restores the LM afterwards if it is shared).</li>
 * </ol>
 *
 * <p>The core {@code TransformerBlock} does not expose its projections, so LoRA here is a
 * LoRA adapter on the LM's output head plus, after each block, a low-rank residual adapter
 * {@code h + s h A B} (a {@link LoraAdapterDense} over a zero frozen weight, {@code B} starting
 * at zero) applied through the decoder's {@link FusionDecoder.BlockHook}. Both start at the
 * pretrained function.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2304.08485">arXiv:2304.08485</a>,
 * <a href="https://arxiv.org/abs/2310.03744">arXiv:2310.03744</a>.
 */
public final class TwoStageBridgeTrainer {

    /** How the LM trains in stage 2. */
    public enum LmMode {
        /** The LM is frozen. */
        FROZEN,
        /** Low-rank adapters on the LM train. */
        LORA,
        /** Every LM weight trains. */
        FULL
    }

    /**
     * A bridge from patch features to the LM.
     */
    public interface Bridge {

        /**
         * The LM input embeddings for these features and text.
         *
         * @param lm the LM
         * @param features patch features {@code [batch, patches, featureDim]}
         * @param text the token ids
         * @return {@code [batch, steps, dim]}
         */
        Var embeddings(FusionDecoder lm, Var features, int[][] text);

        /**
         * An optional hook into the LM blocks.
         *
         * @param features patch features
         * @return the hook, or null
         */
        default FusionDecoder.BlockHook hook(Var features) {
            return null;
        }

        /** @return the bridge's trainable values */
        List<Var> parameters();

        /**
         * A prefix bridge: an MLP projector's output before the text (LLaVA).
         *
         * @param projector the projector
         * @return the bridge
         */
        static Bridge projector(PatchProjector projector) {
            return new Bridge() {
                @Override
                public Var embeddings(FusionDecoder lm, Var features, int[][] text) {
                    return Var.concat(1, projector.forward(features), lm.embed(text));
                }

                @Override
                public List<Var> parameters() {
                    return projector.parameters();
                }
            };
        }

        /**
         * A prefix bridge: the Q-Former's query outputs before the text (BLIP-2).
         *
         * @param qformer the Q-Former
         * @return the bridge
         */
        static Bridge queryTransformer(QueryTransformer qformer) {
            return new Bridge() {
                @Override
                public Var embeddings(FusionDecoder lm, Var features, int[][] text) {
                    return Var.concat(1, qformer.prefix(features), lm.embed(text));
                }

                @Override
                public List<Var> parameters() {
                    return qformer.parameters();
                }
            };
        }

        /**
         * Gated cross-attention (Flamingo): {@code padSlots} pad tokens before the text, the
         * image enters through the block hook.
         *
         * @param adapter the adapter
         * @param padSlots pad tokens before the text
         * @return the bridge
         */
        static Bridge gated(GatedCrossAttentionAdapter adapter, int padSlots) {
            return new Bridge() {
                @Override
                public Var embeddings(FusionDecoder lm, Var features, int[][] text) {
                    int[][] ids = new int[text.length][padSlots + text[0].length];
                    for (int i = 0; i < text.length; i++) {
                        System.arraycopy(text[i], 0, ids[i], padSlots, text[i].length);
                    }
                    return lm.embed(ids);
                }

                @Override
                public FusionDecoder.BlockHook hook(Var features) {
                    return adapter.hook(adapter.resample(features));
                }

                @Override
                public List<Var> parameters() {
                    return adapter.parameters();
                }
            };
        }
    }

    /**
     * A training batch.
     *
     * @param features patch features {@code [batch, patches, featureDim]}
     * @param text input token ids {@code [batch][steps]}
     * @param targets next-token targets over the whole LM sequence (prefix included), -1 to skip
     */
    public record Batch(Tensor features, int[][] text, int[][] targets) {

        /**
         * Copies the arrays.
         *
         * @param features the features
         * @param text the text
         * @param targets the targets
         */
        public Batch {
            text = text.clone();
            targets = targets.clone();
        }

        @Override
        public int[][] text() {
            return text.clone();
        }

        @Override
        public int[][] targets() {
            return targets.clone();
        }
    }

    private final FusionDecoder lm;
    private final Bridge bridge;
    private final LmMode mode;
    private final LoraAdapterDense head;
    private final List<LoraAdapterDense> blockAdapters = new ArrayList<>();

    /**
     * Creates the trainer.
     *
     * @param lm the pretrained LM
     * @param bridge the bridge
     * @param mode how the LM trains in stage 2
     * @param rank the LoRA rank (used with {@link LmMode#LORA})
     * @param seed the seed of the adapters
     */
    public TwoStageBridgeTrainer(FusionDecoder lm, Bridge bridge, LmMode mode, int rank, long seed) {
        this.lm = lm;
        this.bridge = bridge;
        this.mode = mode;
        if (mode == LmMode.LORA) {
            SplittableRandom r = new SplittableRandom(seed);
            List<Var> lmParams = lm.parameters();
            Var w = lmParams.get(lmParams.size() - 2);
            Var b = lmParams.get(lmParams.size() - 1);
            this.head = LoraAdapterDense.lora(dense(lm.dim(), lm.vocab(), w.value(), b.value(), r), rank,
                    2f * rank, r);
            Tensor zeroW = Tensor.zeros(Shape.of(lm.dim(), lm.dim()));
            Tensor zeroB = Tensor.zeros(Shape.of(lm.dim()));
            for (int i = 0; i < lm.layers(); i++) {
                blockAdapters.add(LoraAdapterDense.lora(dense(lm.dim(), lm.dim(), zeroW, zeroB, r), rank, 2f * rank, r));
            }
        } else {
            this.head = null;
        }
    }

    private static Dense dense(int in, int out, Tensor weight, Tensor bias, SplittableRandom r) {
        Dense d = new Dense(in, out, Activation.IDENTITY, Init.LECUN, r);
        d.weight().assign(Tensor.of(weight.shape(), weight.toArray()));
        d.bias().assign(Tensor.of(bias.shape(), bias.toArray()));
        return d;
    }

    private FusionDecoder.BlockHook hook(FusionDecoder.BlockHook bridgeHook, boolean adapters) {
        if (!adapters || blockAdapters.isEmpty()) {
            return bridgeHook;
        }
        return (block, x) -> {
            Var h = bridgeHook == null ? x : bridgeHook.after(block, x);
            return blockAdapters.get(block).forward(h).add(h);
        };
    }

    private Var head(Var hidden, boolean adapters) {
        return adapters && head != null ? head.forward(hidden) : lm.project(hidden);
    }

    /**
     * Logits with the image.
     *
     * @param features patch features
     * @param text token ids
     * @param training whether training
     * @return {@code [batch, steps, vocab]}
     */
    public Var logits(Tensor features, int[][] text, boolean training) {
        return logits(features, text, training, true);
    }

    private Var logits(Tensor features, int[][] text, boolean training, boolean adapters) {
        Var f = Var.of(features);
        Var x = bridge.embeddings(lm, f, text);
        return head(lm.hidden(x, null, hook(bridge.hook(f), adapters), training), adapters);
    }

    /**
     * Logits of text alone (the LM's text ability, with any stage-2 adapters).
     *
     * @param text token ids
     * @param training whether training
     * @return {@code [batch, steps, vocab]}
     */
    public Var textLogits(int[][] text, boolean training) {
        return head(lm.hidden(lm.embed(text), null, hook(null, true), training), true);
    }

    /**
     * Stage 1: trains only the bridge, the LM frozen and its adapters (if any) unused.
     *
     * @param steps optimiser steps
     * @param batches the batch of each step
     * @param learningRate Adam's learning rate
     * @return the mean loss of the last tenth of the steps
     */
    public double align(int steps, IntFunction<Batch> batches, float learningRate) {
        return train(steps, batches, learningRate, bridge.parameters(), false);
    }

    /**
     * Stage 2: trains the bridge and the LM as set by the {@link LmMode}.
     *
     * @param steps optimiser steps
     * @param batches the batch of each step
     * @param learningRate Adam's learning rate
     * @return the mean loss of the last tenth of the steps
     */
    public double instruct(int steps, IntFunction<Batch> batches, float learningRate) {
        return train(steps, batches, learningRate, stageTwoParameters(), true);
    }

    /** @return the values stage 2 trains */
    public List<Var> stageTwoParameters() {
        List<Var> p = new ArrayList<>(bridge.parameters());
        switch (mode) {
            case LORA -> {
                p.addAll(head.parameters());
                for (LoraAdapterDense a : blockAdapters) {
                    p.addAll(a.parameters());
                }
            }
            case FULL -> p.addAll(lm.parameters());
            default -> {
            }
        }
        return p;
    }

    /** @return the number of values stage 2 trains */
    public long trainableCount() {
        long n = 0;
        for (Var v : stageTwoParameters()) {
            n += v.value().shape().size();
        }
        return n;
    }

    private double train(int steps, IntFunction<Batch> batches, float learningRate, List<Var> params,
            boolean adapters) {
        Optimizer opt = Optimizer.adam(learningRate);
        double tail = 0;
        int counted = 0;
        boolean lmTrains = adapters && mode == LmMode.FULL;
        for (int s = 0; s < steps; s++) {
            Batch b = batches.apply(s);
            Var loss = FusionDecoder.loss(logits(b.features(), b.text(), true, adapters), b.targets());
            if (s >= steps - Math.max(1, steps / 10)) {
                tail += loss.value().item();
                counted++;
            }
            loss.backward();
            opt.step(params);
            if (!lmTrains) {
                lm.zeroGrad();
            }
        }
        return counted == 0 ? Double.NaN : tail / counted;
    }

    @Override
    public String toString() {
        return "TwoStageBridgeTrainer(" + mode + ")";
    }
}
