package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Forward gradients with local losses and activity perturbation (Ren, Kornblith, Liao and
 * Hinton, ICLR 2023, <a href="https://arxiv.org/abs/2210.03310">arXiv:2210.03310</a>). Plain
 * weight-perturbed forward gradients ({@link ForwardGradientLearner}) have a variance that grows
 * with the number of parameters perturbed. This learner cuts it in three ways:
 *
 * <ul>
 *   <li><b>local losses</b>: every hidden layer has its own linear classifier head and is
 *       trained only on that head's loss, so no signal crosses layers (inputs are detached);</li>
 *   <li><b>activity perturbation</b>: the random tangent perturbs each example's pre-activations
 *       {@code z_i} (a few dozen numbers) instead of the weights (thousands), and the
 *       per-example directional derivative {@code d_i = (dl_i/dz_i) . u_i} gives
 *       {@code g_i = d_i u_i}, an unbiased estimate of {@code dl_i/dz_i}; the weight gradient then
 *       follows locally as {@code h^T g / batch};</li>
 *   <li><b>groups</b>: the units of a layer are split into groups perturbed one at a time, so each
 *       estimate covers only {@code width / groups} dimensions.</li>
 * </ul>
 *
 * <p>The directional derivatives are Jacobian-vector products of the local loss
 * ({@link CurvatureProducts#jacobianVector}); the heads, one linear layer each, take their exact
 * local gradient. {@link #meanCosine()} reports how well the estimates align with the true
 * local-loss gradient of the hidden weights.
 */
public final class LocalForwardGradient implements LocalLearner {

    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final List<Var> heads = new ArrayList<>();
    private final List<Var> headBiases = new ArrayList<>();
    private final List<Var> all = new ArrayList<>();
    private final Optimizer optimizer;
    private final int groups;
    private final int directions;
    private final SplittableRandom random;
    private final int classes;
    private double cosineSum;
    private long cosineCount;

    /**
     * @param inputs the input width
     * @param classes the number of classes
     * @param hidden the hidden layer widths
     * @param groups unit groups perturbed separately per layer
     * @param directions random tangents averaged per group
     * @param optimizer the optimizer that applies the estimates
     * @param seed the seed
     */
    public LocalForwardGradient(int inputs, int classes, int[] hidden, int groups, int directions,
            Optimizer optimizer, long seed) {
        this.optimizer = optimizer;
        this.groups = groups;
        this.directions = directions;
        this.classes = classes;
        this.random = new SplittableRandom(seed);
        int in = inputs;
        for (int h : hidden) {
            float std = (float) Math.sqrt(2.0 / (in + h)) * 1.4f;
            weights.add(Var.param(Tensor.normal(Shape.of(in, h), 0, std, random)));
            biases.add(Var.param(Tensor.zeros(Shape.of(1, h))));
            heads.add(Var.param(Tensor.normal(Shape.of(h, classes), 0, (float) Math.sqrt(1.0 / h), random)));
            headBiases.add(Var.param(Tensor.zeros(Shape.of(1, classes))));
            in = h;
        }
        for (int l = 0; l < weights.size(); l++) {
            all.addAll(List.of(weights.get(l), biases.get(l), heads.get(l), headBiases.get(l)));
        }
    }

    /** @return every parameter, layer by layer */
    public List<Var> parameters() {
        return List.copyOf(all);
    }

    /**
     * @return the mean cosine between the estimated and the exact local-loss gradient of the
     *     hidden weights over all training batches so far
     */
    public double meanCosine() {
        return cosineCount == 0 ? 0 : cosineSum / cosineCount;
    }

    private Var perExampleLoss(Var z, int l, Tensor onehot) {
        Var logits = z.tanh().matmul(heads.get(l)).add(headBiases.get(l));
        return logits.logsumexp(1, true).sub(logits.mul(Var.of(onehot)).sum(1, true));
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int batch = inputs.shape().dim(0);
        float[] y = labels.toArray();
        float[] oh = new float[batch * classes];
        for (int i = 0; i < batch; i++) {
            oh[i * classes + (int) y[i]] = 1;
        }
        Tensor onehot = Tensor.of(Shape.of(batch, classes), oh);
        Tensor h = inputs;
        float total = 0;
        for (int l = 0; l < weights.size(); l++) {
            Tensor z = h.matmul(weights.get(l).value()).add(biases.get(l).value());
            Var zv = Var.param(z);
            Var losses = perExampleLoss(zv, l, onehot);
            Var mean = losses.mean();
            total += mean.value().item();
            List<Var> headGrads = Var.gradients(mean, List.of(heads.get(l), headBiases.get(l)), false);
            heads.get(l).setGrad(headGrads.get(0).value());
            headBiases.get(l).setGrad(headGrads.get(1).value());
            int width = z.shape().dim(1);
            float[] est = new float[batch * width];
            for (int g = 0; g < groups; g++) {
                int from = g * width / groups;
                int to = (g + 1) * width / groups;
                for (int k = 0; k < directions; k++) {
                    double[] u = new double[batch * width];
                    for (int i = 0; i < batch; i++) {
                        for (int j = from; j < to; j++) {
                            u[i * width + j] = random.nextGaussian();
                        }
                    }
                    float[] d = CurvatureProducts.jacobianVector(losses, List.of(zv), u).toArray();
                    for (int i = 0; i < batch; i++) {
                        for (int j = from; j < to; j++) {
                            est[i * width + j] += (float) (d[i] * u[i * width + j] / directions);
                        }
                    }
                }
            }
            Tensor gz = Tensor.of(z.shape(), est).div(batch);
            Tensor gw = h.transpose().matmul(gz);
            weights.get(l).setGrad(gw);
            biases.get(l).setGrad(gz.sum(0, true));
            // diagnostic only: the exact local gradient of this layer's weight
            Tensor exact = h.transpose().matmul(Var.gradients(mean, List.of(zv), false).get(0).value());
            cosineSum += cosine(gw.toArray(), exact.toArray());
            cosineCount++;
            h = z.tanh();
        }
        optimizer.step(all);
        return total / weights.size();
    }

    private static double cosine(float[] a, float[] b) {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        return aa == 0 || bb == 0 ? 0 : ab / Math.sqrt(aa * bb);
    }

    @Override
    public Tensor predict(Tensor inputs) {
        Tensor h = inputs;
        int last = weights.size() - 1;
        for (int l = 0; l < last; l++) {
            h = h.matmul(weights.get(l).value()).add(biases.get(l).value()).tanh();
        }
        Tensor top = h.matmul(weights.get(last).value()).add(biases.get(last).value()).tanh();
        return top.matmul(heads.get(last).value()).add(headBiases.get(last).value());
    }

    @Override
    public String toString() {
        return "LocalForwardGradient(groups=" + groups + ", directions=" + directions + ")";
    }
}
