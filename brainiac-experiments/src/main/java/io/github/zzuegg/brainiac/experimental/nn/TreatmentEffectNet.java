package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Neural estimators of heterogeneous treatment effects {@code tau(x) = E[y(1) - y(0) | x]}
 * from observational data: a shared representation {@code Phi(x)} feeds one outcome head
 * per treatment arm, each trained only on its own arm's factual outcomes.
 *
 * <ul>
 *   <li>{@link Kind#TARNET} (Shalit, Johansson, Sontag 2017, <a href="https://arxiv.org/abs/1606.03976">arXiv:1606.03976</a>):
 *       the two heads on a shared body.</li>
 *   <li>{@link Kind#DRAGONNET} (Shi, Blei, Veitch 2019, <a href="https://arxiv.org/abs/1906.02120">arXiv:1906.02120</a>):
 *       adds a propensity head on the same representation, so it keeps the information
 *       relevant to treatment (and drops what only predicts the outcome), and
 *       <em>targeted regularization</em>: a learned {@code epsilon} perturbs the outcome
 *       prediction along {@code t / g - (1 - t) / (1 - g)}, and its loss makes the fitted
 *       outcomes satisfy the efficient-influence-curve equation, so the plug-in ATE of
 *       the perturbed heads ({@link #ate}) is doubly robust.</li>
 * </ul>
 *
 * <p>{@link #forward} gives {@code [batch, 3]}: the untreated outcome, the treated outcome
 * and the propensity logit (0 for TARNet).
 */
public final class TreatmentEffectNet implements Layer {

    /** The architecture. */
    public enum Kind {
        /** Shared body, two outcome heads. */
        TARNET,
        /** Plus a propensity head and targeted regularization. */
        DRAGONNET
    }

    private final Kind kind;
    private final List<Var> body = new ArrayList<>();
    private final List<Var> head0 = new ArrayList<>();
    private final List<Var> head1 = new ArrayList<>();
    private final Var propensityW;
    private final Var propensityB;
    private final Var epsilon;

    /**
     * Creates the network: two hidden ReLU layers in the body, one hidden layer per head.
     *
     * @param kind the architecture
     * @param inputs covariates
     * @param hidden the width of every hidden layer
     * @param random the initializer
     */
    public TreatmentEffectNet(Kind kind, int inputs, int hidden, RandomGenerator random) {
        this.kind = kind;
        dense(body, inputs, hidden, random);
        dense(body, hidden, hidden, random);
        dense(head0, hidden, hidden / 2, random);
        dense(head0, hidden / 2, 1, random);
        dense(head1, hidden, hidden / 2, random);
        dense(head1, hidden / 2, 1, random);
        this.propensityW = Var.param(Tensor.normal(Shape.of(hidden, 1), 0f, (float) Math.sqrt(1.0 / hidden), random));
        this.propensityB = Var.param(Tensor.zeros(Shape.of(1)));
        this.epsilon = Var.param(Tensor.zeros(Shape.of(1)));
    }

    private static void dense(List<Var> into, int in, int out, RandomGenerator random) {
        into.add(Var.param(Tensor.normal(Shape.of(in, out), 0f, (float) Math.sqrt(2.0 / in), random)));
        into.add(Var.param(Tensor.zeros(Shape.of(out))));
    }

    private static Var mlp(List<Var> layers, Var x, boolean lastLinear) {
        Var h = x;
        for (int l = 0; l < layers.size(); l += 2) {
            h = h.matmul(layers.get(l)).add(layers.get(l + 1));
            if (!lastLinear || l + 2 < layers.size()) {
                h = h.relu();
            }
        }
        return h;
    }

    /** @return the architecture */
    public Kind kind() {
        return kind;
    }

    @Override
    public Var forward(Var input) {
        Var phi = mlp(body, input, false);
        Var y0 = mlp(head0, phi, true);
        Var y1 = mlp(head1, phi, true);
        Var g = kind == Kind.DRAGONNET ? phi.matmul(propensityW).add(propensityB) : y0.mul(0f);
        return Var.concat(1, y0, y1, g);
    }

    /**
     * The training loss: the factual outcome's squared error, plus for DragonNet
     * {@code alpha} times the propensity cross-entropy and {@code beta} times the targeted
     * regularization.
     *
     * @param output this network's output for the batch
     * @param t treatments {@code [batch]}, 0 or 1
     * @param y outcomes {@code [batch]}
     * @param alpha the propensity loss weight, such as 1
     * @param beta the targeted regularization weight, such as 1
     * @return the mean loss
     */
    public Var loss(Var output, Tensor t, Tensor y, float alpha, float beta) {
        int n = t.shape().dim(0);
        Tensor tc = t.reshape(n, 1);
        Tensor yc = y.reshape(n, 1);
        Var y0 = output.slice(1, 0, 1);
        Var y1 = output.slice(1, 1, 2);
        Var factual = y1.mul(tc).add(y0.mul(tc.neg().add(1f)));
        Var loss = factual.sub(yc).square().mean();
        if (kind == Kind.DRAGONNET) {
            Var logit = output.slice(1, 2, 3);
            Var g = logit.sigmoid().mul(0.98f).add(0.01f);
            Var bce = g.log().mul(tc).add(g.neg().add(1f).log().mul(tc.neg().add(1f))).mean().neg();
            Var h = g.pow(-1f).mul(tc).sub(g.neg().add(1f).pow(-1f).mul(tc.neg().add(1f)));
            Var perturbed = factual.add(h.mul(epsilon));
            Var targeted = perturbed.sub(yc).square().mean();
            loss = loss.add(bce.mul(alpha)).add(targeted.mul(beta));
        }
        return loss;
    }

    /**
     * The conditional average treatment effect {@code mu1(x) - mu0(x)} per row.
     *
     * @param x covariates
     * @return {@code [batch]}
     */
    public Tensor cate(Tensor x) {
        Tensor out = forward(Var.of(x)).value();
        int n = x.shape().dim(0);
        return out.slice(1, 1, 2).sub(out.slice(1, 0, 1)).reshape(n);
    }

    /**
     * The average treatment effect: the mean CATE, for DragonNet with the targeted
     * perturbation {@code epsilon (1 / g + 1 / (1 - g))} applied to it.
     *
     * @param x covariates
     * @return the estimate
     */
    public double ate(Tensor x) {
        double plain = cate(x).mean().item();
        if (kind != Kind.DRAGONNET) {
            return plain;
        }
        Tensor g = forward(Var.of(x)).value().slice(1, 2, 3).sigmoid().mul(0.98f).add(0.01f);
        double eps = epsilon.value().item();
        return plain + eps * g.pow(-1f).add(g.neg().add(1f).pow(-1f)).mean().item();
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>(body);
        out.addAll(head0);
        out.addAll(head1);
        if (kind == Kind.DRAGONNET) {
            out.add(propensityW);
            out.add(propensityB);
            out.add(epsilon);
        }
        return out;
    }

    @Override
    public int outputSize(int inputSize) {
        return 3;
    }

    @Override
    public String toString() {
        return kind + "(" + body.get(0).value().shape() + ")";
    }
}
