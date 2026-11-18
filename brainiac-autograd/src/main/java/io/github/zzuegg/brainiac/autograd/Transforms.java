package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Function transforms in the style of JAX and {@code torch.func}: Jacobian-vector products
 * (forward mode), vector-Jacobian products, Hessian-vector products, full Jacobians, and
 * mapping a function over a batch, including per-example gradients.
 *
 * <p>Forward mode is computed with the double-backward identity: the vector-Jacobian
 * product {@code u -> J^T u} is linear in {@code u}, so differentiating
 * {@code <J^T u, v>} with respect to {@code u} gives {@code J v}. It needs operations
 * whose gradients can be differentiated again, as {@link Var#gradients} with a graph does.
 *
 * <p>{@link #vmap} runs the function once per example and stacks the results, so any
 * function works, including ones that only accept a single example; it gives the
 * semantics of a vectorizing map, not its speed.
 */
public final class Transforms {

    private Transforms() {
    }

    /**
     * The output of a function and its derivative along a direction.
     *
     * @param output {@code f(primals)}
     * @param tangent {@code J v}: how the output moves when the inputs move along the tangents
     */
    public record Jvp(Tensor output, Tensor tangent) {
    }

    /**
     * The output of a function and the gradients of a weighted sum of its outputs.
     *
     * @param output {@code f(primals)}
     * @param gradients {@code J^T u}, one per input
     */
    public record Vjp(Tensor output, List<Tensor> gradients) {
    }

    private static List<Var> params(List<Tensor> values) {
        List<Var> xs = new ArrayList<>(values.size());
        for (Tensor t : values) {
            xs.add(Var.param(t));
        }
        return xs;
    }

    /**
     * Jacobian-vector product (forward-mode derivative).
     *
     * @param f the function
     * @param primals the point to differentiate at
     * @param tangents a direction per input, same shapes as the primals
     * @return the output and its directional derivative
     */
    public static Jvp jvp(Function<List<Var>, Var> f, List<Tensor> primals, List<Tensor> tangents) {
        if (primals.size() != tangents.size()) {
            throw new IllegalArgumentException(primals.size() + " primals but " + tangents.size() + " tangents");
        }
        List<Var> xs = params(primals);
        Var y = f.apply(xs);
        Var u = Var.param(Tensor.zeros(y.shape(), y.dtype(), y.value().device()));
        List<Var> g = Var.gradients(y, u, xs, true);
        Var dot = null;
        for (int i = 0; i < g.size(); i++) {
            if (!tangents.get(i).shape().equals(primals.get(i).shape())) {
                throw new IllegalArgumentException("tangent " + i + " has shape " + tangents.get(i).shape()
                        + ", its primal " + primals.get(i).shape());
            }
            Var term = g.get(i).mul(tangents.get(i)).sum();
            dot = dot == null ? term : dot.add(term);
        }
        Tensor tangent = dot.requiresGrad() ? Var.gradients(dot, List.of(u), false).get(0).value()
                : Tensor.zeros(y.shape(), y.dtype(), y.value().device());
        return new Jvp(y.value(), tangent);
    }

    /**
     * {@link #jvp(Function, List, List)} of a function of one value.
     *
     * @param f the function
     * @param primal the point
     * @param tangent the direction
     * @return the output and its directional derivative
     */
    public static Jvp jvp(Function<Var, Var> f, Tensor primal, Tensor tangent) {
        return jvp(xs -> f.apply(xs.get(0)), List.of(primal), List.of(tangent));
    }

    /**
     * Vector-Jacobian product (reverse-mode derivative).
     *
     * @param f the function
     * @param primals the point to differentiate at
     * @param cotangent weights for the output, same shape as it
     * @return the output and {@code J^T u} per input
     */
    public static Vjp vjp(Function<List<Var>, Var> f, List<Tensor> primals, Tensor cotangent) {
        List<Var> xs = params(primals);
        Var y = f.apply(xs);
        List<Var> g = Var.gradients(y, Var.of(cotangent), xs, false);
        return new Vjp(y.value(), g.stream().map(Var::value).toList());
    }

    /**
     * Hessian-vector product of a scalar function.
     *
     * @param f a function with one output element, such as a loss
     * @param primal the point
     * @param v the direction
     * @return {@code H v}, shaped like the primal
     */
    public static Tensor hvp(Function<Var, Var> f, Tensor primal, Tensor v) {
        Var x = Var.param(primal);
        Var y = f.apply(x);
        Var g = Var.gradients(y, List.of(x), true).get(0);
        Var dot = g.mul(v).sum();
        if (!dot.requiresGrad()) {
            return Tensor.zeros(primal.shape(), primal.dtype(), primal.device());
        }
        return Var.gradients(dot, List.of(x), false).get(0).value();
    }

    /**
     * The full Jacobian, one reverse pass per output element: for small functions.
     *
     * @param f the function
     * @param primal the point
     * @return {@code [output dims..., input dims...]}
     */
    public static Tensor jacobian(Function<Var, Var> f, Tensor primal) {
        Var x = Var.param(primal);
        Var y = f.apply(x);
        int outputs = Math.toIntExact(y.shape().size());
        int inputs = Math.toIntExact(primal.shape().size());
        float[] rows = new float[outputs * inputs];
        for (int o = 0; o < outputs; o++) {
            float[] seed = new float[outputs];
            seed[o] = 1f;
            Tensor s = Tensor.of(y.shape(), seed, y.value().device()).to(y.dtype());
            float[] g = Var.gradients(y, Var.of(s), List.of(x), false).get(0).value()
                    .to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
            System.arraycopy(g, 0, rows, o * inputs, inputs);
        }
        int[] dims = new int[y.shape().rank() + primal.shape().rank()];
        System.arraycopy(y.shape().toArray(), 0, dims, 0, y.shape().rank());
        System.arraycopy(primal.shape().toArray(), 0, dims, y.shape().rank(), primal.shape().rank());
        return Tensor.of(Shape.of(dims), rows, primal.device());
    }

    /**
     * Maps a function over one axis of its input: each example is passed without that
     * axis, and the results are stacked along a new leading axis. Gradients flow through.
     *
     * @param f the function of one example
     * @param batch the examples
     * @param axis the axis that holds the examples
     * @return {@code [examples, f's output dims...]}
     */
    public static Var vmap(Function<Var, Var> f, Var batch, int axis) {
        int a = axis < 0 ? axis + batch.shape().rank() : axis;
        int n = batch.shape().dim(a);
        int[] rest = new int[batch.shape().rank() - 1];
        for (int d = 0, k = 0; d < batch.shape().rank(); d++) {
            if (d != a) {
                rest[k++] = batch.shape().dim(d);
            }
        }
        Var[] outs = new Var[n];
        for (int i = 0; i < n; i++) {
            Var y = f.apply(batch.slice(a, i, i + 1).reshape(rest));
            int[] lead = new int[y.shape().rank() + 1];
            lead[0] = 1;
            System.arraycopy(y.shape().toArray(), 0, lead, 1, y.shape().rank());
            outs[i] = y.reshape(lead);
        }
        return Var.concat(0, outs);
    }

    /**
     * The gradient of a loss for each example separately, as differential privacy
     * (per-example clipping), influence functions and gradient-noise studies need.
     *
     * @param loss the loss of one example (a batch of one, leading axis kept), with one element
     * @param parameters the values to differentiate with respect to
     * @param batch the examples along axis 0
     * @return one tensor per parameter, {@code [examples, parameter dims...]}
     */
    public static List<Tensor> perExampleGradients(Function<Var, Var> loss, List<Var> parameters, Tensor batch) {
        int n = batch.shape().dim(0);
        List<List<Tensor>> rows = new ArrayList<>(parameters.size());
        for (int p = 0; p < parameters.size(); p++) {
            rows.add(new ArrayList<>(n));
        }
        for (int i = 0; i < n; i++) {
            Var x = Var.of(batch.slice(0, i, i + 1));
            List<Var> g = Var.gradients(loss.apply(x), parameters, false);
            for (int p = 0; p < parameters.size(); p++) {
                Tensor t = g.get(p).value();
                int[] lead = new int[t.shape().rank() + 1];
                lead[0] = 1;
                System.arraycopy(t.shape().toArray(), 0, lead, 1, t.shape().rank());
                rows.get(p).add(t.reshape(lead));
            }
        }
        List<Tensor> out = new ArrayList<>(parameters.size());
        for (List<Tensor> r : rows) {
            out.add(Tensor.concat(0, r.toArray(Tensor[]::new)));
        }
        return out;
    }

    /**
     * A differentially private gradient (DP-SGD, Abadi et al. 2016): each example's
     * gradient is scaled down to an L2 norm of at most {@code clipNorm} over all parameters
     * together, the clipped gradients are summed, Gaussian noise with standard deviation
     * {@code noiseMultiplier * clipNorm} is added, and the result is divided by the batch size.
     *
     * @param loss the loss of one example (a batch of one, leading axis kept), with one element
     * @param parameters the values to differentiate with respect to
     * @param batch the examples along axis 0
     * @param clipNorm the largest norm one example may contribute
     * @param noiseMultiplier the noise relative to {@code clipNorm}; 0 only clips
     * @param random the source of the noise
     * @return one averaged, noised gradient per parameter
     */
    public static List<Tensor> privateGradients(Function<Var, Var> loss, List<Var> parameters, Tensor batch,
            float clipNorm, float noiseMultiplier, java.util.random.RandomGenerator random) {
        if (!(clipNorm > 0) || noiseMultiplier < 0) {
            throw new IllegalArgumentException("clipNorm must be positive and noiseMultiplier non-negative, got "
                    + clipNorm + " and " + noiseMultiplier);
        }
        int n = batch.shape().dim(0);
        List<Tensor> per = perExampleGradients(loss, parameters, batch);
        // squared norm of each example's gradient over all parameters: [n]
        Tensor squares = null;
        for (Tensor g : per) {
            Tensor s = g.reshape(n, Math.toIntExact(g.shape().size() / Math.max(1, n))).square().sum(1, false);
            squares = squares == null ? s : squares.add(s);
        }
        float[] norms = squares.toArray();
        float[] scale = new float[n];
        for (int i = 0; i < n; i++) {
            float norm = (float) Math.sqrt(norms[i]);
            scale[i] = norm > clipNorm ? clipNorm / norm : 1f;
        }
        List<Tensor> out = new ArrayList<>(per.size());
        for (Tensor g : per) {
            int[] dims = new int[g.shape().rank()];
            java.util.Arrays.fill(dims, 1);
            dims[0] = n;
            Tensor factors = Tensor.of(Shape.of(dims), scale, g.device()).to(g.dtype());
            Tensor sum = g.mul(factors).sum(0, false);
            if (noiseMultiplier > 0) {
                sum = sum.add(Tensor.normal(sum.shape(), 0f, noiseMultiplier * clipNorm, random)
                        .to(sum.device()).to(sum.dtype()));
            }
            out.add(sum.div(n));
        }
        return out;
    }
}
