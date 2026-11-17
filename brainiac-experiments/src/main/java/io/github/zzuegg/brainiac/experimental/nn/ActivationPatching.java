package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Activation patching (causal tracing): finds where a model computes something by running
 * it on a <em>corrupted</em> input while copying in activations from a <em>clean</em> run,
 * one layer (or one layer at one position, or a few units) at a time. Where restoring the
 * clean activation restores the clean output, that activation carries the information.
 *
 * <p>The effect is reported as the restored share of the metric:
 * {@code (m(patched) - m(corrupted)) / (m(clean) - m(corrupted))}, 1 meaning fully
 * restored.
 *
 * <p>Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (causal tracing),
 * <a href="https://arxiv.org/abs/2202.05262">arXiv:2202.05262</a>; Heimersheim and Nanda 2024,
 * "How to use and interpret activation patching", <a href="https://arxiv.org/abs/2404.15255">arXiv:2404.15255</a>.
 */
public final class ActivationPatching {

    private ActivationPatching() {
    }

    /**
     * Runs the corrupted input with part of one layer's output taken from the clean run.
     *
     * @param model a sequential model
     * @param layer the layer to patch
     * @param clean the clean input
     * @param corrupted the corrupted input (same shape)
     * @param position for sequence outputs {@code [batch, time, features]}, the time step to
     *     patch, or -1 for all
     * @param units the features to patch, or null for all
     * @return the patched run's output
     */
    public static Tensor patch(Model model, int layer, Tensor clean, Tensor corrupted, int position, int[] units) {
        List<Var> cleanRun = LayerHooks.trace(model, Var.of(clean));
        List<Var> corruptRun = LayerHooks.trace(model, Var.of(corrupted));
        Tensor c = cleanRun.get(layer).value();
        Tensor x = corruptRun.get(layer).value();
        Tensor mask = mask(x, position, units);
        Tensor patched = Tensor.where(mask, c, x);
        return LayerHooks.resume(model, layer, Var.of(patched)).value();
    }

    private static Tensor mask(Tensor like, int position, int[] units) {
        int[] dims = like.shape().toArray();
        int features = dims[dims.length - 1];
        boolean[] unit = new boolean[features];
        if (units == null) {
            java.util.Arrays.fill(unit, true);
        } else {
            for (int u : units) {
                unit[u] = true;
            }
        }
        float[] m = new float[Math.toIntExact(like.size())];
        int time = dims.length == 3 ? dims[1] : 1;
        for (int i = 0; i < m.length; i++) {
            int f = i % features;
            int t = (i / features) % time;
            m[i] = unit[f] && (position < 0 || dims.length != 3 || t == position) ? 1 : 0;
        }
        return Tensor.of(like.shape(), m, like.device()).greater(Tensor.scalar(0.5f).to(like.device()));
    }

    /**
     * The restored share of a metric for patching each layer's whole output.
     *
     * @param model a sequential model
     * @param clean the clean input
     * @param corrupted the corrupted input
     * @param metric reads the quantity of interest from a model output (such as a logit)
     * @return one restoration share per layer
     */
    public static double[] traceLayers(Model model, Tensor clean, Tensor corrupted, ToDoubleFunction<Tensor> metric) {
        double mc = metric.applyAsDouble(model.forward(Var.of(clean)).value());
        double mx = metric.applyAsDouble(model.forward(Var.of(corrupted)).value());
        int layers = model.layers().size();
        double[] out = new double[layers];
        for (int l = 0; l < layers; l++) {
            out[l] = share(metric.applyAsDouble(patch(model, l, clean, corrupted, -1, null)), mc, mx);
        }
        return out;
    }

    /**
     * The restored share for patching each layer at each time step, for models whose layers
     * output sequences {@code [batch, time, features]} (layers with other shapes get NaN).
     *
     * @param model a sequential model
     * @param clean the clean input
     * @param corrupted the corrupted input
     * @param metric reads the quantity of interest from a model output
     * @return {@code [layer][time]} restoration shares
     */
    public static double[][] tracePositions(Model model, Tensor clean, Tensor corrupted,
            ToDoubleFunction<Tensor> metric) {
        double mc = metric.applyAsDouble(model.forward(Var.of(clean)).value());
        double mx = metric.applyAsDouble(model.forward(Var.of(corrupted)).value());
        List<Var> run = LayerHooks.trace(model, Var.of(clean));
        double[][] out = new double[run.size()][];
        for (int l = 0; l < run.size(); l++) {
            Tensor a = run.get(l).value();
            if (a.rank() != 3) {
                out[l] = new double[] {Double.NaN};
                continue;
            }
            out[l] = new double[a.shape().dim(1)];
            for (int t = 0; t < out[l].length; t++) {
                out[l][t] = share(metric.applyAsDouble(patch(model, l, clean, corrupted, t, null)), mc, mx);
            }
        }
        return out;
    }

    private static double share(double patched, double clean, double corrupted) {
        double gap = clean - corrupted;
        return Math.abs(gap) < 1e-12 ? 0 : (patched - corrupted) / gap;
    }

    /**
     * A metric reading one output value of the first sample.
     *
     * @param index the output index (such as the logit or probability of a class)
     * @return the metric
     */
    public static ToDoubleFunction<Tensor> output(int index) {
        return t -> t.to(DType.FLOAT32).toArray()[index];
    }
}
