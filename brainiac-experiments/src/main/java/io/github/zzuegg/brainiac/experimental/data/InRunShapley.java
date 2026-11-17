package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * In-Run Data Shapley (Wang et al. 2024, "Data Shapley in One Training Run",
 * <a href="https://arxiv.org/abs/2406.11011">arXiv:2406.11011</a>, ICLR 2025 outstanding
 * paper honourable mention): the value of each training example to <em>this</em> training
 * run, accumulated step by step at almost no extra cost. A step on batch {@code B} with
 * learning rate {@code lr} moves the weights by {@code d = -(lr / |B|) sum_j g_j}; to second
 * order the validation loss changes by {@code g_val . d + d^T H_val d / 2}. Treating that
 * change as a cooperative game among the batch's examples, the Shapley value of example
 * {@code i} has the closed form
 *
 * <pre>
 *   phi_i = (lr / |B|) g_i . g_val  -  (lr^2 / 2|B|^2) g_i . H_val (sum_j g_j)
 * </pre>
 * (the first-order term is its own contribution, the second-order term splits each pair
 * interaction equally). Summed over the steps an example took part in, it is that example's
 * contribution to lowering the validation loss; negative values flag harmful data such as
 * mislabelled examples.
 *
 * <p>Per-example gradients are never formed. For a dense layer with inputs {@code A}
 * ({@code [B, in]}) and output gradients {@code D} ({@code [B, out]}) the per-example weight
 * gradient is {@code a_i^T d_i}, so its dot product with any fixed matrix {@code U} is
 * {@code (a_i U) . d_i}: one {@code [B, in] x [in, out]} product per layer (the paper's
 * "ghost dot product"). {@link #accumulate} takes those per-layer tensors for any network
 * of dense layers; {@link #record} computes them for a {@link SimpleMlp} classifier,
 * including the Hessian-vector product through {@link Var#gradients(Var, List, boolean)}
 * with a graph.
 */
public final class InRunShapley {

    private final double[] values;
    private final int[] counts;

    /**
     * @param examples the size of the training set
     */
    public InRunShapley(int examples) {
        this.values = new double[examples];
        this.counts = new int[examples];
    }

    /**
     * Adds {@code scale * <g_i, U>} for every example of a batch, from the dense layers'
     * inputs and output gradients.
     *
     * @param ids the batch's example indices
     * @param inputs per layer, the inputs {@code [B, in]}
     * @param deltas per layer, the gradient of each example's own loss with respect to the
     *     pre-activation {@code [B, out]}
     * @param weightDirections per layer, the direction {@code U} for the weight {@code [in, out]}
     * @param biasDirections per layer, the direction for the bias {@code [out]}, or {@code null}
     * @param scale the factor on the dot products
     */
    public void accumulate(int[] ids, List<Tensor> inputs, List<Tensor> deltas, List<Tensor> weightDirections,
            List<Tensor> biasDirections, double scale) {
        Tensor total = null;
        for (int l = 0; l < inputs.size(); l++) {
            Tensor d = deltas.get(l);
            Tensor dots = inputs.get(l).matmul(weightDirections.get(l)).mul(d).sum(1, false);
            if (biasDirections != null && biasDirections.get(l) != null) {
                dots = dots.add(d.mul(biasDirections.get(l)).sum(1, false));
            }
            total = total == null ? dots : total.add(dots);
        }
        float[] v = total.toArray();
        for (int i = 0; i < ids.length; i++) {
            values[ids[i]] += scale * v[i];
        }
    }

    /**
     * Scores one training step of a {@link SimpleMlp} classifier trained with cross-entropy;
     * call it before the optimizer step, with the same batch.
     *
     * @param net the network
     * @param x the batch inputs
     * @param y the batch class ids
     * @param ids the batch's example indices
     * @param validX validation inputs
     * @param validY validation class ids
     * @param learningRate the step's learning rate
     * @param secondOrder whether to add the second-order (interaction) term
     */
    public void record(SimpleMlp net, Tensor x, Tensor y, int[] ids, Tensor validX, Tensor validY, float learningRate,
            boolean secondOrder) {
        int depth = net.depth();
        int batch = ids.length;
        // the batch's pre-activations, with each example's loss summed (not averaged)
        List<Var> pre = new ArrayList<>();
        List<Tensor> inputs = new ArrayList<>();
        Var h = Var.of(x);
        for (int l = 0; l < depth; l++) {
            inputs.add(h.value());
            Var z = net.linear(l, h);
            pre.add(z);
            h = l < depth - 1 ? net.activation().apply(z) : z;
        }
        Var batchLoss = Functional.crossEntropy(h, y).mul(batch);
        List<Tensor> deltas = new ArrayList<>();
        for (Var g : Var.gradients(batchLoss, pre, false)) {
            deltas.add(g.value());
        }
        List<Var> params = net.parameters();
        Var validLoss = Functional.crossEntropy(net.forward(Var.of(validX)), validY);
        List<Var> gv = Var.gradients(validLoss, params, secondOrder);
        List<Tensor> wDir = new ArrayList<>();
        List<Tensor> bDir = new ArrayList<>();
        for (int l = 0; l < depth; l++) {
            wDir.add(gv.get(2 * l).value());
            bDir.add(gv.get(2 * l + 1).value());
        }
        accumulate(ids, inputs, deltas, wDir, bDir, learningRate / (double) batch);
        if (secondOrder) {
            List<Var> gb = Var.gradients(batchLoss, params, false);   // sum_j g_j
            Var dot = null;
            for (int p = 0; p < params.size(); p++) {
                Var term = gv.get(p).mul(gb.get(p).value()).sum();
                dot = dot == null ? term : dot.add(term);
            }
            List<Var> hv = Var.gradients(dot, params, false);   // H_val sum_j g_j
            List<Tensor> wh = new ArrayList<>();
            List<Tensor> bh = new ArrayList<>();
            for (int l = 0; l < depth; l++) {
                wh.add(hv.get(2 * l).value());
                bh.add(hv.get(2 * l + 1).value());
            }
            accumulate(ids, inputs, deltas, wh, bh,
                    -learningRate * (double) learningRate / (2.0 * batch * batch));
        }
        for (int i : ids) {
            counts[i]++;
        }
    }

    /** @return the accumulated value of each example, higher is more helpful; a copy */
    public double[] scores() {
        return values.clone();
    }

    /** @return how many recorded steps each example took part in; a copy */
    public int[] counts() {
        return counts.clone();
    }
}
