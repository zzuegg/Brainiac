package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MIMO (Havasi et al., ICLR 2021, <a href="https://arxiv.org/abs/2010.06610">arXiv:2010.06610</a>):
 * an ensemble for the price of one forward pass. During training the network takes {@code M}
 * independently drawn inputs concatenated, and has {@code M} output heads, each trained on its
 * own input's label, so it learns {@code M} independent subnetworks. At test time the same
 * input is repeated {@code M} times and the heads' predictions are averaged.
 *
 * <p>Two of the paper's knobs: the input repetition ratio {@code ρ} (the share of rows where all
 * members see the same input, which trades member independence for capacity) and batch
 * repetition (each batch row repeated {@code b} times before shuffling).
 * {@link #forward} returns the mean of the heads' logits, so the network drops into code that
 * expects a {@link Layer} from inputs to logits.
 */
public final class MimoNetwork implements Layer {

    private final int members;
    private final int inputs;
    private final int classes;
    private final FeatureClassifier net;

    /**
     * Creates the network.
     *
     * @param inputs input features of one member
     * @param hidden the hidden widths
     * @param classes the number of classes
     * @param members the ensemble size {@code M}
     * @param seed the seed
     */
    public MimoNetwork(int inputs, int[] hidden, int classes, int members, long seed) {
        this.members = members;
        this.inputs = inputs;
        this.classes = classes;
        this.net = FeatureClassifier.mlp(inputs * members, hidden, classes * members, seed);
    }

    /** @return the ensemble size */
    public int size() {
        return members;
    }

    /**
     * The training loss of a batch: {@code M} shuffled copies of the batch (row {@code i} of copy
     * {@code m} equal to copy 0 with probability {@code ρ}), concatenated, each head scored
     * against its copy's labels; summed over heads.
     *
     * @param x the batch {@code [n, inputs]}
     * @param ids the class ids {@code [n]}
     * @param repetitionRatio {@code ρ}
     * @param batchRepetition {@code b}, at least 1
     * @param random the shuffling source
     * @return the scalar loss
     */
    public Var loss(Tensor x, Tensor ids, double repetitionRatio, int batchRepetition, SplittableRandom random) {
        int n0 = x.shape().dim(0);
        int n = n0 * batchRepetition;
        float[] xv = x.toArray();
        float[] yv = ids.toArray();
        float[] stacked = new float[n * members * inputs];
        float[][] labels = new float[members][n];
        int[] base = new int[n];
        for (int i = 0; i < n; i++) {
            base[i] = random.nextInt(n0);
        }
        for (int m = 0; m < members; m++) {
            for (int i = 0; i < n; i++) {
                int row = m == 0 || random.nextDouble() < repetitionRatio ? base[i] : random.nextInt(n0);
                System.arraycopy(xv, row * inputs, stacked, (i * members + m) * inputs, inputs);
                labels[m][i] = yv[row];
            }
        }
        Var z = net.logits(Var.of(Tensor.of(Shape.of(n, members * inputs), stacked)), true)
                .reshape(n, members, classes);
        Var loss = null;
        for (int m = 0; m < members; m++) {
            Var head = z.slice(1, m, m + 1).reshape(n, classes);
            Var l = Functional.crossEntropy(head, Tensor.of(Shape.of(n), labels[m]));
            loss = loss == null ? l : loss.add(l);
        }
        return loss;
    }

    /**
     * Every head's logits for repeated inputs.
     *
     * @param input {@code [n, inputs]}
     * @param training the training flag
     * @return {@code [n, M, classes]}
     */
    public Var members(Var input, boolean training) {
        int n = input.shape().dim(0);
        Var rep = input.reshape(n, 1, inputs).expand(Shape.of(n, members, inputs)).reshape(n, members * inputs);
        return net.logits(rep, training).reshape(n, members, classes);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return members(input, training).mean(1, false);
    }

    @Override
    public List<Var> parameters() {
        return net.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    /** @return one audit member per head */
    public List<EnsembleAudit.Member> auditMembers() {
        List<EnsembleAudit.Member> out = new ArrayList<>();
        for (int i = 0; i < members; i++) {
            int m = i;
            out.add(x -> {
                Tensor z = Var.noGrad(() -> members(Var.of(x), false).value());
                return EnsembleAudit.softmaxRows(z.slice(1, m, m + 1).reshape(z.shape().dim(0), classes));
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "MimoNetwork(M=" + members + ")";
    }
}
