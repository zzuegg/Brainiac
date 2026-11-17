package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Small tensor helpers shared by the experimental deep RL agents: batching rows,
 * one-hot actions, gradient clipping, target-network copies and Polyak averaging, the
 * Huber loss and a numerically safe softmax. They mirror the core's package-private
 * {@code Nets} helpers, which agents outside {@code io.github.zzuegg.brainiac.rl} cannot
 * reach.
 */
public final class RlTensors {

    private RlTensors() {
    }

    /**
     * @param rows the rows, each at least {@code width} long
     * @param width the values per row
     * @param device where to put the tensor
     * @return a {@code [rows, width]} tensor
     */
    public static Tensor rows(float[][] rows, int width, Device device) {
        float[] flat = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, flat, i * width, width);
        }
        return Tensor.of(Shape.of(rows.length, width), flat).to(device);
    }

    /**
     * @param values the values
     * @param device where to put the tensor
     * @return a {@code [n, 1]} column
     */
    public static Tensor column(float[] values, Device device) {
        return Tensor.of(Shape.of(values.length, 1), values).to(device);
    }

    /**
     * Runs a model on one observation without recording gradients.
     *
     * @param model the model
     * @param observation the observation
     * @return the output row
     */
    public static float[] row(Model model, float[] observation) {
        Tensor x = Tensor.of(Shape.of(1, observation.length), observation).to(model.device());
        return model.forward(Var.of(x), false).value().to(DType.FLOAT32).toArray();
    }

    /**
     * @param indices one index per row (as floats)
     * @param n the number of classes
     * @param device where to put the tensor
     * @return a {@code [rows, n]} one-hot matrix
     */
    public static Tensor oneHot(float[] indices, int n, Device device) {
        float[] h = new float[indices.length * n];
        for (int i = 0; i < indices.length; i++) {
            h[i * n + Math.round(indices[i])] = 1;
        }
        return Tensor.of(Shape.of(indices.length, n), h).to(device);
    }

    /**
     * Rescales gradients whose global norm exceeds a bound.
     *
     * @param params the parameters
     * @param maxNorm the bound; 0 or less for none
     * @return the norm before clipping
     */
    public static double clip(List<Var> params, float maxNorm) {
        double total = 0;
        for (Var p : params) {
            if (p.grad() != null) {
                total += p.grad().square().sum().item();
            }
        }
        double norm = Math.sqrt(total);
        if (maxNorm > 0 && norm > maxNorm) {
            float scale = (float) (maxNorm / norm);
            for (Var p : params) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(scale));
                }
            }
        }
        return norm;
    }

    /**
     * Clears the gradients of every parameter.
     *
     * @param params the parameters
     */
    public static void zeroGrad(List<Var> params) {
        for (Var p : params) {
            p.zeroGrad();
        }
    }

    /**
     * Copies parameter values (same layout required).
     *
     * @param from the source parameters
     * @param to the target parameters
     */
    public static void copy(List<Var> from, List<Var> to) {
        if (from.size() != to.size()) {
            throw new IllegalArgumentException("the target network must have the same layout");
        }
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
    }

    /**
     * Polyak averaging: moves every target value a fraction towards the source.
     *
     * @param from the source parameters
     * @param to the target parameters
     * @param tau the fraction, in {@code (0, 1]}
     */
    public static void blend(List<Var> from, List<Var> to, float tau) {
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(to.get(i).value().mul(1 - tau).add(from.get(i).value().mul(tau)));
        }
    }

    /**
     * Huber loss per element: quadratic within {@code delta}, linear beyond.
     *
     * @param error the error
     * @param delta the threshold
     * @return the loss per element
     */
    public static Var huber(Var error, float delta) {
        Var abs = error.abs();
        Var quadratic = abs.minimum(delta);
        Var linear = abs.sub(quadratic);
        return quadratic.square().mul(0.5f).add(linear.mul(delta));
    }

    /**
     * Log-softmax along the last axis, shifted by the (constant) row maximum for
     * numerical safety.
     *
     * @param logits the logits
     * @return the log-probabilities
     */
    public static Var logSoftmax(Var logits) {
        int axis = logits.shape().rank() - 1;
        Var shifted = logits.sub(logits.value().max(axis, true));
        return shifted.sub(shifted.exp().sum(axis, true).log());
    }

    /**
     * Softmax along the last axis.
     *
     * @param logits the logits
     * @return the probabilities
     */
    public static Var softmax(Var logits) {
        int axis = logits.shape().rank() - 1;
        Var e = logits.sub(logits.value().max(axis, true)).exp();
        return e.div(e.sum(axis, true));
    }

    /**
     * @param v the values
     * @return the index of the largest (first on ties)
     */
    public static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * @param v the values
     * @return the index of the largest (first on ties)
     */
    public static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }
}
