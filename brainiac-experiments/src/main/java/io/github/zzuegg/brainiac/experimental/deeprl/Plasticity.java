package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Tools against the loss of plasticity in deep RL: networks trained on non-stationary targets
 * gradually lose the ability to fit new ones. Diagnostics and interventions for models whose
 * hidden layers are activated {@link Dense} layers (the RL MLPs of this module):
 *
 * <ul>
 *   <li>{@link #dormantFraction}: the share of dormant neurons, whose activation normalised by
 *       the layer's mean activation is at most {@code tau} (Sokar et al., ICML 2023,
 *       <a href="https://arxiv.org/abs/2302.12902">arXiv:2302.12902</a>).</li>
 *   <li>{@link #redo}: ReDo, recycling dormant neurons: their incoming weights are
 *       re-initialised and their outgoing weights zeroed, so the network's function is unchanged
 *       while the neurons can learn again (optimizer moments of the touched weights are reset
 *       too).</li>
 *   <li>{@link #featureRank}: the effective rank of a layer's features (the number of singular
 *       values covering 99% of their sum; Kumar et al., 2021).</li>
 *   <li>{@link #weightNorm}: the global parameter norm, which grows as plasticity fades.</li>
 *   <li>{@link #resetLast}: periodic resets of the last layers (Nikishin et al., ICML 2022,
 *       <a href="https://arxiv.org/abs/2205.07802">arXiv:2205.07802</a>), keeping the replay
 *       buffer; plasticity injection (Nikishin et al., NeurIPS 2023) is
 *       {@link SelfPredictive#injectPlasticity()}, and shrink-and-perturb is
 *       {@link SelfPredictive#shrinkPerturb}.</li>
 * </ul>
 */
public final class Plasticity {

    private Plasticity() {
    }

    /** The activated Dense layers of a model, in order. */
    private static List<Dense> hidden(Model model) {
        List<Dense> out = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Dense d && d.activation() != Activation.IDENTITY) {
                out.add(d);
            }
        }
        return out;
    }

    /** Every Dense layer of a model, in order. */
    private static List<Dense> dense(Model model) {
        List<Dense> out = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Dense d) {
                out.add(d);
            }
        }
        return out;
    }

    /**
     * The activations of every activated Dense layer on a batch.
     *
     * @param model the model
     * @param batch the inputs
     * @return one {@code [batch, units]} tensor per hidden layer
     */
    public static List<Tensor> activations(Model model, Tensor batch) {
        List<Tensor> out = new ArrayList<>();
        Var.noGrad(() -> model.forward(Var.of(batch), false, (step, v) -> {
            if (step.layer() instanceof Dense d && d.activation() != Activation.IDENTITY) {
                out.add(v.value());
            }
            return v;
        }));
        return out;
    }

    /** Per-neuron scores: mean |activation| over the batch divided by the layer's mean of that. */
    private static float[] scores(Tensor act) {
        float[] m = act.abs().mean(0, false).toArray();
        double mean = 0;
        for (float v : m) {
            mean += v;
        }
        mean /= m.length;
        float[] s = new float[m.length];
        for (int i = 0; i < m.length; i++) {
            s[i] = (float) (m[i] / (mean + 1e-9));
        }
        return s;
    }

    /**
     * @param model the model
     * @param batch inputs to measure on
     * @param tau the dormancy threshold (0 counts only exactly dead neurons; ReDo uses 0 to 0.1)
     * @return the share of hidden neurons that are dormant
     */
    public static double dormantFraction(Model model, Tensor batch, float tau) {
        int dormant = 0;
        int total = 0;
        for (Tensor a : activations(model, batch)) {
            for (float s : scores(a)) {
                total++;
                if (s <= tau) {
                    dormant++;
                }
            }
        }
        return total == 0 ? 0 : (double) dormant / total;
    }

    /**
     * ReDo: recycles the dormant neurons of every hidden Dense layer.
     *
     * @param model the model, changed in place
     * @param batch inputs to measure on
     * @param tau the dormancy threshold
     * @param optimizer whose moments for the touched weights are reset; may be null
     * @param seed the seed of the fresh incoming weights
     * @return how many neurons were recycled
     */
    public static int redo(Model model, Tensor batch, float tau, Optimizer optimizer, long seed) {
        List<Tensor> acts = activations(model, batch);
        List<Dense> all = dense(model);
        List<Dense> hid = hidden(model);
        SplittableRandom random = new SplittableRandom(seed);
        int recycled = 0;
        for (int l = 0; l < hid.size() && l < acts.size(); l++) {
            Dense layer = hid.get(l);
            int index = all.indexOf(layer);
            Dense next = index + 1 < all.size() ? all.get(index + 1) : null;
            float[] s = scores(acts.get(l));
            Tensor w = layer.weight().value();
            int in = w.shape().dim(0);
            int units = w.shape().dim(1);
            float[] wv = w.toArray();
            float[] bv = layer.bias().value().toArray();
            float[] keepIn = new float[in * units];
            java.util.Arrays.fill(keepIn, 1);
            float[] keepBias = new float[units];
            java.util.Arrays.fill(keepBias, 1);
            float[] nv = next == null ? null : next.weight().value().toArray();
            int outs = next == null ? 0 : next.weight().value().shape().dim(1);
            float[] keepOut = next == null ? null : new float[nv.length];
            if (keepOut != null) {
                java.util.Arrays.fill(keepOut, 1);
            }
            double limit = Math.sqrt(6.0 / in);   // He-uniform for ReLU
            boolean any = false;
            for (int i = 0; i < units; i++) {
                if (s[i] > tau) {
                    continue;
                }
                any = true;
                recycled++;
                for (int r = 0; r < in; r++) {
                    wv[r * units + i] = (float) random.nextDouble(-limit, limit);
                    keepIn[r * units + i] = 0;
                }
                bv[i] = 0;
                keepBias[i] = 0;
                if (next != null) {
                    for (int c = 0; c < outs; c++) {
                        nv[i * outs + c] = 0;
                        keepOut[i * outs + c] = 0;
                    }
                }
            }
            if (!any) {
                continue;
            }
            layer.weight().assign(Tensor.of(w.shape(), wv).to(w.device()));
            layer.bias().assign(Tensor.of(layer.bias().value().shape(), bv).to(w.device()));
            if (next != null) {
                Tensor nw = next.weight().value();
                next.weight().assign(Tensor.of(nw.shape(), nv).to(nw.device()));
            }
            if (optimizer != null) {
                optimizer.resetState(layer.weight(), Tensor.of(w.shape(), keepIn));
                optimizer.resetState(layer.bias(), Tensor.of(layer.bias().value().shape(), keepBias));
                if (next != null) {
                    optimizer.resetState(next.weight(), Tensor.of(next.weight().value().shape(), keepOut));
                }
            }
        }
        return recycled;
    }

    /**
     * The effective rank of the last hidden layer's features: the smallest {@code k} whose top
     * singular values cover 99% of their sum.
     *
     * @param model the model
     * @param batch inputs to measure on
     * @return the effective rank, 0 if the model has no hidden layer
     */
    public static int featureRank(Model model, Tensor batch) {
        List<Tensor> acts = activations(model, batch);
        if (acts.isEmpty()) {
            return 0;
        }
        return effectiveRank(acts.get(acts.size() - 1), 0.01);
    }

    /**
     * @param features {@code [rows, features]}
     * @param delta the uncovered share of the singular-value sum
     * @return the effective rank
     */
    public static int effectiveRank(Tensor features, double delta) {
        float[] s = Linalg.svd(features)[1].toArray();
        double total = 0;
        for (float v : s) {
            total += v;
        }
        if (total <= 0) {
            return 0;
        }
        double acc = 0;
        for (int k = 0; k < s.length; k++) {
            acc += s[k];
            if (acc >= (1 - delta) * total) {
                return k + 1;
            }
        }
        return s.length;
    }

    /**
     * @param model the model
     * @return the ℓ2 norm of all its parameters
     */
    public static double weightNorm(Model model) {
        double sum = 0;
        for (Var p : model.parameters()) {
            sum += p.value().square().sum().item();
        }
        return Math.sqrt(sum);
    }

    /**
     * Re-initialises the last {@code layers} Dense layers (He-uniform weights, zero biases),
     * the reset of Nikishin et al. (2022).
     *
     * @param model the model, changed in place
     * @param layers how many of the last Dense layers to reset
     * @param seed the seed
     */
    public static void resetLast(Model model, int layers, long seed) {
        List<Dense> all = dense(model);
        SplittableRandom random = new SplittableRandom(seed);
        for (int l = Math.max(0, all.size() - layers); l < all.size(); l++) {
            Dense d = all.get(l);
            Shape ws = d.weight().value().shape();
            double limit = Math.sqrt(6.0 / ws.dim(0));
            d.weight().assign(Tensor.uniform(ws, (float) -limit, (float) limit, random).to(d.weight().value().device()));
            d.bias().assign(Tensor.zeros(d.bias().value().shape()).to(d.bias().value().device()));
        }
    }
}
