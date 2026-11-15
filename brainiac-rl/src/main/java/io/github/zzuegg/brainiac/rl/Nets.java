package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/** Small helpers shared by the neural agents. */
public final class Nets {

    private Nets() {
    }

    /**
     * A plain network: {@code tanh} hidden layers, a linear output. A good default for
     * policies and value functions on small control tasks.
     *
     * @param inputs values per observation
     * @param outputs output values
     * @param seed the initialization seed
     * @param hidden the hidden layer sizes
     * @return the model
     */
    public static Model mlp(int inputs, int outputs, long seed, int... hidden) {
        return mlp(inputs, outputs, Activation.TANH, seed, hidden);
    }

    /**
     * A plain network with the given hidden activation and a linear output.
     *
     * @param inputs values per observation
     * @param outputs output values
     * @param activation the hidden activation
     * @param seed the initialization seed
     * @param hidden the hidden layer sizes
     * @return the model
     */
    public static Model mlp(int inputs, int outputs, Activation activation, long seed, int... hidden) {
        Model.Builder b = Model.sequential().input(inputs);
        for (int h : hidden) {
            b.dense(h, activation);
        }
        return b.dense(outputs).seed(seed).build();
    }

    static Tensor rows(float[][] rows, int width, Device device) {
        float[] flat = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, flat, i * width, width);
        }
        return Tensor.of(Shape.of(rows.length, width), flat).to(device);
    }

    static Tensor column(float[] values, Device device) {
        return Tensor.of(Shape.of(values.length, 1), values).to(device);
    }

    static float[] row(Model model, float[] observation) {
        Tensor x = Tensor.of(Shape.of(1, observation.length), observation).to(model.device());
        return model.forward(Var.of(x), false).value().to(DType.FLOAT32).toArray();
    }

    static Tensor oneHot(float[] actions, int n, Device device) {
        float[] h = new float[actions.length * n];
        for (int i = 0; i < actions.length; i++) {
            h[i * n + Math.round(actions[i])] = 1;
        }
        return Tensor.of(Shape.of(actions.length, n), h).to(device);
    }

    static void clip(List<Var> params, float maxNorm) {
        if (maxNorm <= 0) {
            return;
        }
        double total = 0;
        for (Var p : params) {
            if (p.grad() != null) {
                total += p.grad().square().sum().item();
            }
        }
        double norm = Math.sqrt(total);
        if (norm > maxNorm) {
            float scale = (float) (maxNorm / norm);
            for (Var p : params) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(scale));
                }
            }
        }
    }

    static void copy(Model from, Model to) {
        List<Var> a = from.parameters();
        List<Var> b = to.parameters();
        if (a.size() != b.size()) {
            throw new IllegalArgumentException("the target network must have the same layout");
        }
        for (int i = 0; i < a.size(); i++) {
            b.get(i).assign(a.get(i).value());
        }
    }

    static void blend(Model from, Model to, float tau) {
        List<Var> a = from.parameters();
        List<Var> b = to.parameters();
        for (int i = 0; i < a.size(); i++) {
            b.get(i).assign(b.get(i).value().mul(1 - tau).add(a.get(i).value().mul(tau)));
        }
    }

    /**
     * Huber loss per element: quadratic within {@code delta}, linear beyond.
     */
    static Var huber(Var error, float delta) {
        Var abs = error.abs();
        Var quadratic = abs.minimum(delta);
        Var linear = abs.sub(quadratic);
        return quadratic.square().mul(0.5f).add(linear.mul(delta));
    }

    static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }
}
