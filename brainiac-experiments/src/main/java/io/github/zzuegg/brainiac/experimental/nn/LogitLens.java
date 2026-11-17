package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * The logit lens and the tuned lens: read a residual model's intermediate states as if they
 * were final, by sending each layer's output straight through the model's readout (its last
 * layers, such as final norm and unembedding). Watching the prediction form layer by layer
 * shows where a model "decides". The tuned lens first passes each state through a small
 * learned affine translator (initialised to the identity, trained to match the final
 * distribution), which corrects for layers using the residual stream differently and gives
 * far more faithful early readings.
 *
 * <p>nostalgebraist 2020, "interpreting GPT: the logit lens"; Belrose et al. 2023,
 * "Eliciting Latent Predictions from Transformers with the Tuned Lens",
 * <a href="https://arxiv.org/abs/2303.08112">arXiv:2303.08112</a>.
 *
 * <p>Works on sequential models whose layers before {@code readoutStart} all output the
 * residual width and whose readout outputs class probabilities.
 */
public final class LogitLens {

    private final Model model;
    private final int readoutStart;
    private final List<Var[]> translators = new ArrayList<>();

    private LogitLens(Model model, int readoutStart) {
        this.model = model;
        this.readoutStart = readoutStart;
    }

    /**
     * Sets up a lens.
     *
     * @param model a sequential residual model
     * @param readoutStart the index of the first readout layer
     * @return the lens
     */
    public static LogitLens of(Model model, int readoutStart) {
        if (readoutStart < 1 || readoutStart >= model.layers().size()) {
            throw new IllegalArgumentException("readout must start after the first layer and before the end");
        }
        return new LogitLens(model, readoutStart);
    }

    /**
     * The logit-lens predictions: each layer before the readout decoded by the readout.
     *
     * @param input a batch
     * @return per layer, the decoded probabilities {@code [batch, classes]}
     */
    public List<Tensor> decode(Tensor input) {
        return decode(input, false);
    }

    /**
     * The tuned-lens predictions ({@link #tune} first).
     *
     * @param input a batch
     * @return per layer, the decoded probabilities
     */
    public List<Tensor> decodeTuned(Tensor input) {
        if (translators.isEmpty()) {
            throw new IllegalStateException("call tune() first");
        }
        return decode(input, true);
    }

    private List<Tensor> decode(Tensor input, boolean tuned) {
        List<Var> states = LayerHooks.trace(model, Var.of(input));
        List<Tensor> out = new ArrayList<>();
        for (int l = 0; l < readoutStart; l++) {
            Var h = states.get(l);
            if (tuned) {
                h = translate(l, h);
            }
            out.add(LayerHooks.resume(model, readoutStart - 1, h).value());
        }
        return out;
    }

    private Var translate(int layer, Var h) {
        Var[] t = translators.get(layer);
        return h.add(h.matmul(t[0]).add(t[1]));
    }

    /**
     * Trains the tuned-lens translators to minimise the KL divergence from the final
     * prediction to each layer's decoded prediction.
     *
     * @param inputs training inputs
     * @param steps optimizer steps
     * @param learningRate the Adam learning rate
     * @return the mean KL per layer after training
     */
    public double[] tune(Tensor inputs, int steps, float learningRate) {
        List<Var> states = LayerHooks.trace(model, Var.of(inputs));
        Tensor target = states.getLast().value();
        int width = states.getFirst().shape().dim(-1);
        translators.clear();
        List<Var> params = new ArrayList<>();
        for (int l = 0; l < readoutStart; l++) {
            Var a = Var.param(Tensor.zeros(Shape.of(width, width)));
            Var b = Var.param(Tensor.zeros(Shape.of(width)));
            translators.add(new Var[] {a, b});
            params.add(a);
            params.add(b);
        }
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            Var total = null;
            for (int l = 0; l < readoutStart; l++) {
                Var h = Var.of(states.get(l).value());
                Var p = LayerHooks.resume(model, readoutStart - 1, translate(l, h));
                Var kl = kl(target, p);
                total = total == null ? kl : total.add(kl);
            }
            total.backward();
            model.parameters().forEach(Var::zeroGrad);
            opt.step(params);
        }
        return divergences(inputs, true);
    }

    private static Var kl(Tensor target, Var p) {
        Tensor t = target.add(1e-7f);
        return Var.of(t).mul(Var.of(t.log()).sub(p.add(1e-7f).log())).sum(1, false).mean();
    }

    /**
     * The mean KL divergence from the final prediction to each layer's decoded prediction.
     *
     * @param inputs the inputs
     * @param tuned whether to use the tuned lens
     * @return one divergence per layer (the last layer before the readout is 0 for the
     *     plain lens)
     */
    public double[] divergences(Tensor inputs, boolean tuned) {
        List<Tensor> lens = decode(inputs, tuned);
        Tensor fin = model.forward(Var.of(inputs)).value();
        double[] out = new double[lens.size()];
        for (int l = 0; l < out.length; l++) {
            out[l] = kl(fin, Var.of(lens.get(l))).value().item();
        }
        return out;
    }

    /**
     * The probability each layer's decoded prediction gives a class, averaged over a batch.
     *
     * @param input a batch
     * @param classes the class of each sample
     * @return one mean probability per layer
     */
    public double[] targetProbability(Tensor input, int[] classes) {
        List<Tensor> lens = decode(input);
        double[] out = new double[lens.size()];
        for (int l = 0; l < out.length; l++) {
            float[] p = lens.get(l).to(DType.FLOAT32).toArray();
            int c = p.length / classes.length;
            for (int i = 0; i < classes.length; i++) {
                out[l] += p[i * c + classes[i]] / classes.length;
            }
        }
        return out;
    }
}
