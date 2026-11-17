package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SpikeGptBlock;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * SpikeGPT at toy scale: a two-block RWKV language model whose channel mixes are spiking
 * ({@link SpikeGptBlock} with LIF neurons and surrogate gradients) against the same model with
 * RWKV's dense squared-ReLU channel mix, both trained on the {@link CountingTasks} language
 * for the same number of steps. Reported: held-out perplexity and the share of spiking
 * neurons that fire per token.
 */
@SuppressWarnings("exports")
public final class SpikingLanguageModel {
    /** Sequence length. */
    public static final int LENGTH = 12;
    /** The counting modulus. */
    public static final int VOCAB = 16;
    private static final int[] STRIDES = {1, 2, 3};

    /**
     * The outcome.
     *
     * @param perplexity the held-out next-token perplexity
     * @param spikeRate the mean share of channel-mix neurons firing per token, NaN when not spiking
     */
    public record Result(double perplexity, double spikeRate) {
    }

    private SpikingLanguageModel() {
    }

    private static Tensor ids(int[][] seqs, int from, int to) {
        float[] f = new float[seqs.length * (to - from)];
        for (int i = 0; i < seqs.length; i++) {
            for (int j = from; j < to; j++) {
                f[i * (to - from) + j - from] = seqs[i][j];
            }
        }
        return Tensor.of(Shape.of(seqs.length, to - from), f);
    }

    private static Var loss(Model model, int[][] seqs, boolean training) {
        int n = seqs.length;
        Var logits = model.forward(Var.of(ids(seqs, 0, LENGTH - 1)), training);
        return Functional.crossEntropy(logits.reshape(n * (LENGTH - 1), VOCAB),
                ids(seqs, 1, LENGTH).reshape(n * (LENGTH - 1)));
    }

    /**
     * Trains a model and evaluates it.
     *
     * @param spiking whether the channel mixes spike
     * @param steps optimizer steps of 16 sequences
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(boolean spiking, int steps, long seed) {
        Model model = Model.sequential().input(LENGTH - 1).embedding(VOCAB, 32)
                .add(SpikeGptBlock.of(2, 8, 64, spiking)).add(SpikeGptBlock.of(2, 8, 64, spiking)).dense(VOCAB)
                .seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Var l = loss(model, CountingTasks.sequences(r, 16, LENGTH, VOCAB, STRIDES), true);
            l.backward();
            opt.step(model.parameters());
        }
        double nll = loss(model, CountingTasks.sequences(new SplittableRandom(seed + 2), 64, LENGTH, VOCAB, STRIDES),
                false).value().item();
        double rate = 0;
        int blocks = 0;
        for (Layer layer : model.layers()) {
            if (layer instanceof SpikeGptBlock b) {
                rate += b.spikeRate();
                blocks++;
            }
        }
        return new Result(Math.exp(nll), rate / blocks);
    }
}
