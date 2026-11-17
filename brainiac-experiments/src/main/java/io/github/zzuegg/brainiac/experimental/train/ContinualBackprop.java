package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Continual Backprop (Dohare et al. 2024, "Loss of plasticity in deep continual learning",
 * Nature 632, <a href="https://arxiv.org/abs/2306.13812">arXiv:2306.13812</a>): networks
 * trained for a long time on changing tasks slowly lose the ability to learn (units die or
 * saturate, weights grow). Continual Backprop keeps injecting fresh randomness: every
 * hidden unit has a running <em>contribution utility</em>, {@code |h_j| * sum_k |w_jk|}
 * (its activity times its outgoing weights), and among units older than a maturity
 * threshold a small fraction {@code replacementRate} per step of the least useful ones is
 * reinitialised: new random incoming weights, zero bias and zero outgoing weights (so the
 * network's function does not jump).
 *
 * <p>Works on a chain of core {@link Dense} layers (such as {@code model.layers()} of an
 * MLP); call {@link #update(Tensor)} with the batch after each optimizer step. Adam moments
 * of a replaced unit are not reset (the core optimizers keep their state private).
 */
public final class ContinualBackprop {

    private final List<Dense> layers;
    private final double replacementRate;
    private final int maturity;
    private final double decay;
    private final SplittableRandom random;
    private final double[][] utility;
    private final int[][] age;
    private final double[] pending;
    private int replaced;

    /**
     * @param layers the dense layers in order; units of all but the last are managed
     * @param replacementRate the fraction of mature units replaced per step, for example {@code 1e-4}
     * @param maturity steps before a unit can be replaced, for example 100
     * @param seed the seed for new weights
     */
    public ContinualBackprop(List<Dense> layers, double replacementRate, int maturity, long seed) {
        this.layers = List.copyOf(layers);
        this.replacementRate = replacementRate;
        this.maturity = maturity;
        this.decay = 0.99;
        this.random = new SplittableRandom(seed);
        utility = new double[layers.size() - 1][];
        age = new int[layers.size() - 1][];
        pending = new double[layers.size() - 1];
        for (int l = 0; l < utility.length; l++) {
            int units = layers.get(l).weight().shape().dim(1);
            utility[l] = new double[units];
            age[l] = new int[units];
        }
    }

    /** @return how many units have been replaced so far */
    public int replaced() {
        return replaced;
    }

    /**
     * @param layer a managed layer
     * @return its units' current utilities
     */
    public double[] utility(int layer) {
        return utility[layer].clone();
    }

    /**
     * Updates utilities from a batch and replaces low-utility mature units.
     *
     * @param inputs the batch the model was just trained on
     */
    public void update(Tensor inputs) {
        Tensor h = inputs;
        for (int l = 0; l < utility.length; l++) {
            h = layers.get(l).forward(Var.of(h)).value();
            float[] act = h.abs().mean(0, false).toArray();
            float[] out = layers.get(l + 1).weight().value().abs().sum(1, false).toArray();
            int eligible = 0;
            for (int j = 0; j < act.length; j++) {
                utility[l][j] = decay * utility[l][j] + (1 - decay) * act[j] * out[j];
                age[l][j]++;
                if (age[l][j] > maturity) {
                    eligible++;
                }
            }
            pending[l] += replacementRate * eligible;
            while (pending[l] >= 1) {
                pending[l] -= 1;
                int worst = -1;
                for (int j = 0; j < act.length; j++) {
                    if (age[l][j] > maturity && (worst < 0 || utility[l][j] < utility[l][worst])) {
                        worst = j;
                    }
                }
                if (worst < 0) {
                    break;
                }
                reinitialise(l, worst);
            }
        }
    }

    /**
     * Gives unit {@code j} of layer {@code l} fresh incoming weights, zero bias and zero
     * outgoing weights.
     *
     * @param l the layer
     * @param j the unit
     */
    public void reinitialise(int l, int j) {
        Var w = layers.get(l).weight();
        int in = w.shape().dim(0);
        int units = w.shape().dim(1);
        float[] wv = w.value().toArray();
        float bound = (float) Math.sqrt(6.0 / (in + units));
        for (int i = 0; i < in; i++) {
            wv[i * units + j] = (float) ((random.nextDouble() * 2 - 1) * bound);
        }
        w.assign(Tensor.of(w.shape(), wv));
        Var b = layers.get(l).bias();
        if (b != null) {
            float[] bv = b.value().toArray();
            bv[j] = 0;
            b.assign(Tensor.of(b.shape(), bv));
        }
        Var next = layers.get(l + 1).weight();
        int outUnits = next.shape().dim(1);
        float[] nv = next.value().toArray();
        for (int k = 0; k < outUnits; k++) {
            nv[j * outUnits + k] = 0;
        }
        next.assign(Tensor.of(next.shape(), nv));
        utility[l][j] = 0;
        age[l][j] = 0;
        replaced++;
    }
}
