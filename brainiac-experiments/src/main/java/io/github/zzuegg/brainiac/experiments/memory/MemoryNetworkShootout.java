package io.github.zzuegg.brainiac.experiments.memory;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DifferentiableNeuralComputer;
import io.github.zzuegg.brainiac.experimental.nn.MemoryNetwork;
import io.github.zzuegg.brainiac.experimental.nn.NeuralTuringMachine;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.SequencePool;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.9: memory networks on {@link BabiLite} stories with one, two and three
 * supporting facts, trained on all three tasks together and scored per task. Compared:
 * {@link MemoryNetwork} end-to-end (MemN2N) with 1, 2 and 3 hops and key-value (3 hops), against
 * a two-block transformer over the flattened story and question, and the experimental
 * {@link NeuralTuringMachine} and {@link DifferentiableNeuralComputer} reading one
 * bag-of-words sentence per step.
 */
public final class MemoryNetworkShootout {

    /** The compared models. */
    public enum Architecture {
        /** MemN2N, one hop */
        MEMN2N_1_HOP,
        /** MemN2N, two hops */
        MEMN2N_2_HOPS,
        /** MemN2N, three hops */
        MEMN2N_3_HOPS,
        /** key-value memory network, three hops */
        KV_MEMNN,
        /** transformer over the flat token sequence */
        TRANSFORMER,
        /** neural Turing machine over sentence vectors */
        NTM,
        /** differentiable neural computer over sentence vectors */
        DNC
    }

    private interface QaModel {
        Var logits(BabiLite.Batch b, boolean training);

        List<Var> parameters();
    }

    private MemoryNetworkShootout() {
    }

    private static Tensor flat(BabiLite.Batch b) {
        int n = b.answers().length;
        int len = BabiLite.SENTENCES * BabiLite.WORDS + BabiLite.WORDS;
        float[] f = new float[n * len];
        for (int i = 0; i < n; i++) {
            int k = 0;
            for (int[] s : b.stories()[i]) {
                for (int w : s) {
                    f[i * len + k++] = w;
                }
            }
            for (int w : b.questions()[i]) {
                f[i * len + k++] = w;
            }
        }
        return Tensor.of(Shape.of(n, len), f);
    }

    /** One bag-of-words vector per sentence, then the question with a flag feature. */
    private static Tensor bags(BabiLite.Batch b) {
        int n = b.answers().length;
        int steps = BabiLite.SENTENCES + 1;
        int f = BabiLite.VOCAB + 1;
        float[] v = new float[n * steps * f];
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < steps; s++) {
                int[] words = s < BabiLite.SENTENCES ? b.stories()[i][s] : b.questions()[i];
                for (int w : words) {
                    if (w != 0) {
                        v[(i * steps + s) * f + w] += 1;
                    }
                }
                if (s == BabiLite.SENTENCES) {
                    v[(i * steps + s) * f + BabiLite.VOCAB] = 1;
                }
            }
        }
        return Tensor.of(Shape.of(n, steps, f), v);
    }

    private static QaModel create(Architecture a, long seed) {
        switch (a) {
            case MEMN2N_1_HOP, MEMN2N_2_HOPS, MEMN2N_3_HOPS, KV_MEMNN -> {
                int hops = a == Architecture.MEMN2N_1_HOP ? 1 : a == Architecture.MEMN2N_2_HOPS ? 2 : 3;
                MemoryNetwork m = MemoryNetwork.of(BabiLite.VOCAB, 32, hops, BabiLite.SENTENCES, BabiLite.WORDS,
                        BabiLite.ANSWERS, a == Architecture.KV_MEMNN ? MemoryNetwork.Kind.KEY_VALUE
                                : MemoryNetwork.Kind.END_TO_END, seed);
                return new QaModel() {
                    @Override
                    public Var logits(BabiLite.Batch b, boolean training) {
                        return m.logits(b.stories(), b.questions(), b.values());
                    }

                    @Override
                    public List<Var> parameters() {
                        return m.parameters();
                    }
                };
            }
            case TRANSFORMER -> {
                int len = BabiLite.SENTENCES * BabiLite.WORDS + BabiLite.WORDS;
                Model m = Model.sequential().input(len).embedding(BabiLite.VOCAB, 16).positionalEncoding()
                        .add(TransformerBlock.of(2).ffnSize(32)).add(TransformerBlock.of(2).ffnSize(32))
                        .sequencePool(SequencePool.Kind.MEAN).dense(BabiLite.ANSWERS, Activation.IDENTITY)
                        .seed(seed).build();
                return model(m, true);
            }
            default -> {
                Model.Builder mb = Model.sequential().input(BabiLite.SENTENCES + 1, BabiLite.VOCAB + 1).seed(seed);
                if (a == Architecture.NTM) {
                    mb.add(NeuralTuringMachine.spec(32, 12, 16, 64));
                } else {
                    mb.add(DifferentiableNeuralComputer.spec(32, 12, 16, 2, 64));
                }
                Model m = mb.sequencePool(SequencePool.Kind.LAST).dense(BabiLite.ANSWERS, Activation.IDENTITY)
                        .build();
                return model(m, false);
            }
        }
    }

    private static QaModel model(Model m, boolean tokens) {
        return new QaModel() {
            @Override
            public Var logits(BabiLite.Batch b, boolean training) {
                return m.forward(Var.of(tokens ? flat(b) : bags(b)), training);
            }

            @Override
            public List<Var> parameters() {
                return m.parameters();
            }
        };
    }

    private static Tensor answers(BabiLite.Batch b) {
        float[] f = new float[b.answers().length];
        for (int i = 0; i < f.length; i++) {
            f[i] = b.answers()[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /**
     * Trains on all three tasks and scores each.
     *
     * @param a the model
     * @param steps Adam steps of 32 questions
     * @param seed the seed
     * @return accuracy on 300 fresh questions of task 1, 2 and 3 (chance 0.25)
     */
    public static double[] run(Architecture a, int steps, long seed) {
        QaModel m = create(a, seed);
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            BabiLite.Batch b = BabiLite.generate(32, r, 1, 2, 3);
            Functional.crossEntropy(m.logits(b, true), answers(b)).backward();
            opt.step(m.parameters());
        }
        double[] acc = new double[3];
        for (int task = 1; task <= 3; task++) {
            BabiLite.Batch b = BabiLite.generate(300, new SplittableRandom(seed * 31 + task), task);
            acc[task - 1] = m.logits(b, false).value().argmax(1).equalTo(answers(b)).mean().item();
        }
        return acc;
    }
}
