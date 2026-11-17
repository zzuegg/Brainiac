package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A reward model as in RLHF (Ouyang et al. 2022, <a href="https://arxiv.org/abs/2203.02155">arXiv:2203.02155</a>):
 * a {@link HookedDecoder} backbone whose last hidden state at the final token goes through a
 * scalar head, trained on preference pairs with the Bradley-Terry loss so that
 * {@code sigmoid(r(chosen) - r(rejected))} is the probability the chosen answer is
 * preferred. It scores samples for best-of-n, rejection sampling and RL.
 */
public final class RewardModel {

    private final HookedDecoder backbone;
    private final Var weight;
    private final Var bias;

    /**
     * @param backbone the decoder (trained along with the head)
     */
    public RewardModel(HookedDecoder backbone) {
        this.backbone = backbone;
        this.weight = Var.param(Tensor.zeros(Shape.of(backbone.settings().dim(), 1)));
        this.bias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    /** @return the backbone and head parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(backbone.parameters());
        p.add(weight);
        p.add(bias);
        return p;
    }

    /**
     * @param sequences {@code [batch][steps]}, the same length
     * @return the rewards {@code [batch]}
     */
    public Var score(int[][] sequences) {
        int batch = sequences.length;
        int steps = sequences[0].length;
        Var h = backbone.forward(sequences).hidden().get(backbone.layers());
        Var last = h.slice(1, steps - 1, steps).reshape(batch, backbone.settings().dim());
        return last.matmul(weight).add(bias).reshape(batch);
    }

    /**
     * @param chosen the preferred sequences
     * @param rejected the others, the same length
     * @return the Bradley-Terry loss
     */
    public Var loss(int[][] chosen, int[][] rejected) {
        return PreferenceLosses.bradleyTerry(score(chosen), score(rejected));
    }

    /**
     * @param chosen the preferred sequences
     * @param rejected the others
     * @return the share of pairs ranked correctly
     */
    public double accuracy(int[][] chosen, int[][] rejected) {
        float[] c = score(chosen).value().toArray();
        float[] r = score(rejected).value().toArray();
        int ok = 0;
        for (int i = 0; i < c.length; i++) {
            ok += c[i] > r[i] ? 1 : 0;
        }
        return (double) ok / c.length;
    }

    /** Clears the gradients of every parameter. */
    public void zeroGrad() {
        for (Var p : parameters()) {
            p.zeroGrad();
        }
    }
}
