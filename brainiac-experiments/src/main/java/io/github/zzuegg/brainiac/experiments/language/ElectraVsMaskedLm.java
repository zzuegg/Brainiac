package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.ElectraPretraining;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * ELECTRA against masked-LM pre-training on the {@link CountingTasks} language, at an equal
 * compute budget. Both pre-train the same bidirectional encoder (two transformer blocks,
 * width 32): the masked LM with a vocabulary head on 15% masked positions, ELECTRA
 * ({@link ElectraPretraining}) as the replaced-token discriminator next to a generator a
 * quarter as wide. The masked LM gets extra steps to match ELECTRA's generator cost (counted in
 * parameters times tokens). The encoders are then frozen and probed: a linear classifier
 * on the features of the middle position reads the sequence's stride from 100 labeled
 * sequences.
 */
@SuppressWarnings("exports")
public final class ElectraVsMaskedLm {
    /** Sequence length. */
    public static final int LENGTH = 10;
    /** The counting modulus. */
    public static final int VOCAB = 16;
    private static final int MASK = VOCAB;
    private static final int[] STRIDES = {1, 2, 3, 4, 5};

    private ElectraVsMaskedLm() {
    }

    /**
     * A bidirectional encoder with an output layer of the given size per position.
     *
     * @param width the transformer width
     * @param outputs the outputs per position
     * @param seed the seed
     * @return the model
     */
    static Model encoder(int width, int outputs, long seed) {
        return Model.sequential().input(LENGTH).embedding(VOCAB + 1, width).positionalEncoding()
                .add(TransformerBlock.of(2).rotary(100f)).add(TransformerBlock.of(2).rotary(100f)).dense(outputs)
                .seed(seed).build();
    }

    private static int[][] batch(SplittableRandom r, int n) {
        return CountingTasks.sequences(r, n, LENGTH, VOCAB, STRIDES);
    }

    /**
     * Pre-trains with ELECTRA.
     *
     * @param steps optimizer steps of 32 sequences
     * @param seed the seed
     * @return {@code {discriminator, generator}}
     */
    public static Model[] electra(int steps, long seed) {
        Model generator = encoder(8, VOCAB + 1, seed + 10);
        Model discriminator = encoder(32, 1, seed);
        ElectraPretraining e = ElectraPretraining.of(generator, discriminator, VOCAB + 1, MASK, 3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            e.step(batch(r, 32), r);
        }
        return new Model[] {discriminator, generator};
    }

    /**
     * Pre-trains a masked language model.
     *
     * @param steps optimizer steps of 32 sequences
     * @param seed the seed
     * @return the model
     */
    public static Model maskedLm(int steps, long seed) {
        Model m = encoder(32, VOCAB + 1, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][] t = batch(r, 32);
            Var loss = ElectraPretraining.maskedLmLoss(m, t, ElectraPretraining.maskPositions(32, LENGTH, 0.15f, r),
                    MASK)[0];
            loss.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * The masked-LM steps that cost as much as some ELECTRA steps.
     *
     * @param electraSteps ELECTRA steps
     * @param electra the discriminator and generator
     * @return the matched step count
     */
    public static int matchedSteps(int electraSteps, Model[] electra) {
        double ratio = (double) (electra[0].parameterCount() + electra[1].parameterCount())
                / electra[0].parameterCount();
        return (int) Math.round(electraSteps * ratio);
    }

    private static Tensor features(Model encoder, int[][] seqs) {
        int last = encoder.graph().size() - 2;   // the step before the output layer
        Var[] captured = new Var[1];
        float[] f = new float[seqs.length * LENGTH];
        for (int i = 0; i < seqs.length; i++) {
            for (int j = 0; j < LENGTH; j++) {
                f[i * LENGTH + j] = seqs[i][j];
            }
        }
        encoder.forward(Var.of(Tensor.of(Shape.of(seqs.length, LENGTH), f)), false, (step, out) -> {
            if (step.index() == last) {
                captured[0] = out;
            }
            return out;
        });
        return captured[0].value().slice(1, LENGTH / 2, LENGTH / 2 + 1).reshape(seqs.length, captured[0].shape().dim(2));
    }

    /**
     * Freezes an encoder and trains a linear stride probe on its mean-pooled features.
     *
     * @param encoder the pre-trained encoder
     * @param seed the seed
     * @return the probe's accuracy on 500 fresh sequences
     */
    public static double probe(Model encoder, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 100;
        int[][] train = new int[n][];
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            int k = i % STRIDES.length;
            train[i] = CountingTasks.sequences(r, 1, LENGTH, VOCAB, STRIDES[k])[0];
            labels[i] = k;
        }
        Tensor x = features(encoder, train);
        int width = x.shape().dim(1);
        Model probe = Model.sequential().input(width).dense(STRIDES.length).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor mean = x.mean(0, true);
        Tensor std = x.sub(mean).square().mean(0, true).add(1e-6f).sqrt();
        Tensor xs = x.sub(mean).div(std);
        for (int s = 0; s < 300; s++) {
            Var loss = Functional.crossEntropy(probe.forward(Var.of(xs), true), Tensor.of(Shape.of(n), labels));
            loss.backward();
            opt.step(probe.parameters());
        }
        int test = 500;
        int[][] seqs = new int[test][];
        int[] truth = new int[test];
        for (int i = 0; i < test; i++) {
            truth[i] = r.nextInt(STRIDES.length);
            seqs[i] = CountingTasks.sequences(r, 1, LENGTH, VOCAB, STRIDES[truth[i]])[0];
        }
        float[] out = probe.forward(Var.of(features(encoder, seqs).sub(mean).div(std))).value().toArray();
        int hit = 0;
        for (int i = 0; i < test; i++) {
            int best = 0;
            for (int c = 1; c < STRIDES.length; c++) {
                if (out[i * STRIDES.length + c] > out[i * STRIDES.length + best]) {
                    best = c;
                }
            }
            hit += best == truth[i] ? 1 : 0;
        }
        return (double) hit / test;
    }
}
