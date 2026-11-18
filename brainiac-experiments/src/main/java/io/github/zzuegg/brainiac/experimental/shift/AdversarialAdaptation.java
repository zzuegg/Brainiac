package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Adversarial domain adaptation with a gradient-reversed domain discriminator:
 *
 * <ul>
 *   <li>{@link Mode#DANN}: the discriminator sees the features, so the extractor learns
 *       marginally domain-invariant features (Ganin et al., JMLR 2016,
 *       <a href="https://arxiv.org/abs/1505.07818">arXiv:1505.07818</a>).</li>
 *   <li>{@link Mode#CDAN}: it sees the flattened outer product of the (detached) predicted
 *       class probabilities and the features, which aligns the class-conditional
 *       distributions; {@link Mode#CDAN_E} also weights each example by
 *       {@code 1 + exp(-entropy)}, normalised per domain, to favour confident predictions
 *       (Long et al., NeurIPS 2018, <a href="https://arxiv.org/abs/1705.10667">arXiv:1705.10667</a>).</li>
 * </ul>
 *
 * <p>Add {@link #loss} to the source classification loss and step the extractor, classifier
 * and {@link #parameters()} together; ramp {@code lambda} with {@link GradientReversal#schedule}.
 */
public final class AdversarialAdaptation {

    /** The discriminator's input. */
    public enum Mode {
        /** features */
        DANN,
        /** features times class probabilities */
        CDAN,
        /** CDAN with entropy conditioning */
        CDAN_E
    }

    private final Mode mode;
    private final Model discriminator;

    /**
     * Creates the discriminator (an MLP with one hidden layer).
     *
     * @param mode DANN or CDAN
     * @param features the feature size
     * @param classes the number of classes
     * @param hidden the discriminator's hidden width
     * @param seed the seed
     */
    public AdversarialAdaptation(Mode mode, int features, int classes, int hidden, long seed) {
        this.mode = mode;
        int in = mode == Mode.DANN ? features : features * classes;
        this.discriminator = Model.sequential().input(in).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                .dense(1, Activation.IDENTITY).seed(seed).build();
    }

    /** @return the discriminator's parameters */
    public List<Var> parameters() {
        return discriminator.parameters();
    }

    /**
     * The domain loss through gradient reversal.
     *
     * @param sourceFeatures {@code [n, d]}
     * @param sourceLogits {@code [n, classes]} (CDAN only)
     * @param targetFeatures {@code [m, d]}
     * @param targetLogits {@code [m, classes]} (CDAN only)
     * @param lambda the reversal strength
     * @return the discriminator's binary cross-entropy (source = 1, target = 0)
     */
    public Var loss(Var sourceFeatures, Var sourceLogits, Var targetFeatures, Var targetLogits, float lambda) {
        Var s = GradientReversal.apply(input(sourceFeatures, sourceLogits), lambda);
        Var t = GradientReversal.apply(input(targetFeatures, targetLogits), lambda);
        Var ds = discriminator.forward(s, true).reshape(s.shape().dim(0));
        Var dt = discriminator.forward(t, true).reshape(t.shape().dim(0));
        Var ls = ds.neg().softplus();   // -log sigmoid(d)
        Var lt = dt.softplus();         // -log (1 - sigmoid(d))
        if (mode == Mode.CDAN_E) {
            Tensor ws = entropyWeights(sourceLogits.value());
            Tensor wt = entropyWeights(targetLogits.value());
            return ls.mul(ws).sum().add(lt.mul(wt).sum()).mul(0.5f);
        }
        return ls.mean().add(lt.mean()).mul(0.5f);
    }

    /**
     * The discriminator's accuracy at telling the domains apart (0.5 = aligned).
     *
     * @param sourceFeatures source features
     * @param sourceLogits source logits
     * @param targetFeatures target features
     * @param targetLogits target logits
     * @return the share of rows classified to their domain
     */
    public double domainAccuracy(Tensor sourceFeatures, Tensor sourceLogits, Tensor targetFeatures,
            Tensor targetLogits) {
        float[] s = discriminator.forward(input(Var.of(sourceFeatures), Var.of(sourceLogits)), false).value().toArray();
        float[] t = discriminator.forward(input(Var.of(targetFeatures), Var.of(targetLogits)), false).value().toArray();
        int ok = 0;
        for (float v : s) {
            ok += v > 0 ? 1 : 0;
        }
        for (float v : t) {
            ok += v <= 0 ? 1 : 0;
        }
        return ok / (double) (s.length + t.length);
    }

    private Var input(Var features, Var logits) {
        if (mode == Mode.DANN) {
            return features;
        }
        int n = features.shape().dim(0);
        int d = features.shape().dim(1);
        int k = logits.shape().dim(1);
        Tensor p = logits.value().sub(logits.value().logsumexp(1, true)).exp();
        Tensor pe = p.reshape(n, k, 1).expand(Shape.of(n, k, d));
        return features.reshape(n, 1, d).expand(Shape.of(n, k, d)).mul(pe).reshape(n, k * d);
    }

    /** {@code 1 + exp(-H)}, normalised to sum to 1. */
    private static Tensor entropyWeights(Tensor logits) {
        Tensor logP = logits.sub(logits.logsumexp(1, true));
        Tensor h = logP.exp().mul(logP).sum(1, false).neg();
        Tensor w = h.neg().exp().add(1);
        return w.div(w.sum().item());
    }
}
