package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Energy-based prediction refined by gradient descent at inference, the idea of
 * Energy-Based Transformers (Gladstone et al., <a href="https://arxiv.org/abs/2507.02092">2507.02092</a>):
 * learn a scalar energy {@code E(x, y)} that is low when {@code y} fits {@code x}, and
 * "think" by descending it, {@code y <- y - alpha * dE/dy}, starting from a guess. More steps
 * means more thinking (System-2 style test-time compute), and the energy itself verifies an
 * answer.
 *
 * <p>Training unrolls the descent and back-propagates the prediction error through it,
 * which needs the gradient of a gradient. Brainiac's autograd does not differentiate a
 * backward pass, so this block writes {@code dE/dy} out by hand as ordinary {@link Var}
 * operations for its energy
 * {@code E(x, y) = w2 . tanh(x W1x + y W1y + b1) + lambda/2 |y|^2}:
 * {@code dE/dy = ((1 - a^2) * w2) W1y^T + lambda y}, so autograd can differentiate the whole
 * unrolled optimization (second order included). The step size {@code alpha} is learned.
 *
 * <p>As a {@link Layer} it maps {@code x} to the refined {@code y} after {@code steps}
 * descents from zero.
 */
public final class EnergyRefiner implements Layer {

    private final int outputs;
    private final int steps;
    private final float lambda;
    private final Var w1x;
    private final Var w1y;
    private final Var b1;
    private final Var w2;
    private final Var logStep;

    private EnergyRefiner(int inputs, int outputs, int hidden, int steps, float lambda, SplittableRandom r) {
        this.outputs = outputs;
        this.steps = steps;
        this.lambda = lambda;
        this.w1x = gauss(r, (float) Math.sqrt(1.0 / inputs), inputs, hidden);
        this.w1y = gauss(r, (float) Math.sqrt(1.0 / outputs), outputs, hidden);
        this.b1 = gauss(r, 0.5f, 1, hidden);
        this.w2 = gauss(r, (float) Math.sqrt(1.0 / hidden), hidden, 1);
        this.logStep = Var.param(Tensor.full(Shape.of(1, 1), -1f));
    }

    private static Var gauss(SplittableRandom r, float scale, int... dims) {
        Shape s = Shape.of(dims);
        float[] v = new float[Math.toIntExact(s.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(s, v));
    }

    /**
     * @param inputs features of {@code x}
     * @param outputs features of {@code y}
     * @param hidden hidden units of the energy
     * @param steps descent steps of {@link #forward}
     * @param lambda the weight of the quadratic term that keeps the energy bounded below
     * @param seed the initialization seed
     * @return the block
     */
    public static EnergyRefiner of(int inputs, int outputs, int hidden, int steps, float lambda, long seed) {
        return new EnergyRefiner(inputs, outputs, hidden, steps, lambda, new SplittableRandom(seed));
    }

    /**
     * @param outputs features of {@code y}
     * @param hidden hidden units of the energy
     * @param steps descent steps
     * @param lambda the quadratic weight
     * @return a spec for model builders
     */
    public static LayerSpec spec(int outputs, int hidden, int steps, float lambda) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new EnergyRefiner(input.dim(input.rank() - 1), outputs, hidden, steps, lambda, random);
            }
        };
    }

    private Var preActivation(Var x, Var y) {
        return x.matmul(w1x).add(y.matmul(w1y)).add(b1);
    }

    /**
     * @param x {@code [batch, inputs]}
     * @param y {@code [batch, outputs]}
     * @return the energy of each pair, {@code [batch, 1]}
     */
    public Var energy(Var x, Var y) {
        return preActivation(x, y).tanh().matmul(w2).add(y.square().sum(1, true).mul(lambda / 2));
    }

    /**
     * The gradient of the energy with respect to {@code y}, as a differentiable expression.
     *
     * @param x {@code [batch, inputs]}
     * @param y {@code [batch, outputs]}
     * @return {@code dE/dy}, {@code [batch, outputs]}
     */
    public Var energyGradient(Var x, Var y) {
        Var a = preActivation(x, y).tanh();
        Var da = a.square().neg().add(1f).mul(w2.transpose());
        return da.matmul(w1y.transpose()).add(y.mul(lambda));
    }

    /** @return the learned step size */
    public Var stepSize() {
        return logStep.exp();
    }

    /**
     * Thinks: gradient descent on the energy from a starting guess.
     *
     * @param x the inputs
     * @param start the first guess
     * @param n descent steps
     * @return every iterate, the start excluded; differentiable with respect to the energy's
     *     parameters
     */
    public List<Var> refine(Var x, Var start, int n) {
        List<Var> out = new ArrayList<>();
        Var y = start;
        Var alpha = stepSize();
        for (int i = 0; i < n; i++) {
            y = y.sub(energyGradient(x, y).mul(alpha));
            out.add(y);
        }
        return out;
    }

    /**
     * The training loss: squared error of every iterate of an unrolled descent from
     * {@code start}, averaged.
     *
     * @param x the inputs
     * @param target the right outputs
     * @param start the first guess (EBT starts from noise)
     * @param n unrolled steps
     * @return the loss
     */
    public Var loss(Var x, Tensor target, Var start, int n) {
        Var total = null;
        for (Var y : refine(x, start, n)) {
            Var l = y.sub(target).square().mean();
            total = total == null ? l : total.add(l);
        }
        return total.div(n);
    }

    @Override
    public Var forward(Var input) {
        Var start = Var.of(Tensor.zeros(Shape.of(input.shape().dim(0), outputs)));
        List<Var> ys = refine(input, start, steps);
        return ys.get(ys.size() - 1);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1x, w1y, b1, w2, logStep);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }
}
