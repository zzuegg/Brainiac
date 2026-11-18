package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Adversarial training of a classifier against {@code ℓ∞} (or {@code ℓ2}) perturbations:
 *
 * <ul>
 *   <li>{@link Method#PGD_AT}: train on PGD examples, {@code min_θ E max_(||δ||<=ε) L(x + δ)}
 *       (Madry et al., ICLR 2018, <a href="https://arxiv.org/abs/1706.06083">arXiv:1706.06083</a>).</li>
 *   <li>{@link Method#TRADES}: clean cross-entropy plus {@code β KL(f(x) || f(x + δ))}, with
 *       {@code δ} found by PGD on the KL term (Zhang et al., ICML 2019,
 *       <a href="https://arxiv.org/abs/1901.08573">arXiv:1901.08573</a>).</li>
 *   <li>{@link Method#FAST_FGSM}: one FGSM step of size {@code 1.25 ε} from a uniform random
 *       start, "fast is better than free" (Wong, Rice and Kolter, ICLR 2020,
 *       <a href="https://arxiv.org/abs/2001.03994">arXiv:2001.03994</a>).</li>
 *   <li>{@link Method#NOISE} and {@link Method#STANDARD}: the baselines (random sign noise of
 *       the same norm; clean training).</li>
 * </ul>
 *
 * <p>The model is any logits function over parameters stepped by a core {@link Optimizer}.
 */
public final class AdversarialTraining {

    /** The training objective. */
    public enum Method {
        /** clean training */
        STANDARD,
        /** random sign noise of norm {@code ε} */
        NOISE,
        /** FGSM from a random start */
        FAST_FGSM,
        /** PGD adversarial training */
        PGD_AT,
        /** TRADES */
        TRADES
    }

    private final UnaryOperator<Var> logits;
    private final List<Var> parameters;
    private final Optimizer optimizer;
    private Method method = Method.PGD_AT;
    private AdversarialAttacks.Norm norm = AdversarialAttacks.Norm.LINF;
    private float epsilon = 0.1f;
    private int steps = 7;
    private float beta = 6;
    private SplittableRandom random = new SplittableRandom(1);

    private AdversarialTraining(UnaryOperator<Var> logits, List<Var> parameters, Optimizer optimizer) {
        this.logits = Objects.requireNonNull(logits);
        this.parameters = List.copyOf(parameters);
        this.optimizer = Objects.requireNonNull(optimizer);
    }

    /**
     * Adversarial training of a model.
     *
     * @param logits the model's training-mode logits
     * @param parameters its parameters
     * @param optimizer the optimizer
     * @return the settings
     */
    public static AdversarialTraining of(UnaryOperator<Var> logits, List<Var> parameters, Optimizer optimizer) {
        return new AdversarialTraining(logits, parameters, optimizer);
    }

    /**
     * @param m the objective
     * @return this
     */
    public AdversarialTraining method(Method m) {
        this.method = Objects.requireNonNull(m);
        return this;
    }

    /**
     * @param n the threat model's norm
     * @return this
     */
    public AdversarialTraining norm(AdversarialAttacks.Norm n) {
        this.norm = Objects.requireNonNull(n);
        return this;
    }

    /**
     * @param e the training budget {@code ε}
     * @return this
     */
    public AdversarialTraining epsilon(float e) {
        this.epsilon = e;
        return this;
    }

    /**
     * @param k PGD steps of the inner maximisation (PGD-AT and TRADES)
     * @return this
     */
    public AdversarialTraining steps(int k) {
        this.steps = k;
        return this;
    }

    /**
     * @param b TRADES' robustness weight
     * @return this
     */
    public AdversarialTraining beta(float b) {
        this.beta = b;
        return this;
    }

    /**
     * @param seed the seed of random starts and noise
     * @return this
     */
    public AdversarialTraining seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /**
     * One training step on a batch.
     *
     * @param x the clean inputs
     * @param ids the classes
     * @return the training loss
     */
    public float step(Tensor x, Tensor ids) {
        Tensor input = switch (method) {
            case STANDARD, TRADES -> x;
            case NOISE -> x.add(signNoise(x.shape()).mul(epsilon));
            case FAST_FGSM -> {
                Tensor start = AdversarialAttacks.project(x.add(uniform(x.shape()).mul(epsilon)), x, norm, epsilon);
                Tensor g = AdversarialAttacks.inputGradient(logits, start, ids);
                yield AdversarialAttacks.project(start.add(AdversarialAttacks.direction(g, norm).mul(1.25f * epsilon)),
                        x, norm, epsilon);
            }
            case PGD_AT -> AdversarialAttacks.pgd(logits, x, ids, norm, epsilon, 2.5f * epsilon / steps, steps, 1,
                    random);
        };
        Var loss;
        if (method == Method.TRADES) {
            Tensor adv = tradesAdversary(x);
            Var clean = logits.apply(Var.of(x));
            loss = Functional.crossEntropy(clean, ids).add(kl(clean, logits.apply(Var.of(adv))).mul(beta));
        } else {
            loss = Functional.crossEntropy(logits.apply(Var.of(input)), ids);
        }
        float value = loss.value().item();
        for (Var p : parameters) {
            p.zeroGrad();
        }
        loss.backward();
        optimizer.step(parameters);
        return value;
    }

    /** PGD on {@code KL(f(x) || f(x'))} from a small Gaussian start. */
    private Tensor tradesAdversary(Tensor x) {
        Tensor p = logits.apply(Var.of(x)).value();
        Tensor adv = x.add(gaussian(x.shape()).mul(0.001f));
        float step = 2.5f * epsilon / steps;
        for (int s = 0; s < steps; s++) {
            Var in = Var.param(adv);
            Var k = kl(Var.of(p), logits.apply(in));
            Tensor g = Var.gradients(k, List.of(in), false).get(0).value();
            adv = AdversarialAttacks.project(adv.add(AdversarialAttacks.direction(g, norm).mul(step)), x, norm, epsilon);
        }
        return adv;
    }

    /**
     * The batch-mean {@code KL(softmax(a) || softmax(b))}.
     *
     * @param a the reference logits
     * @param b the other logits
     * @return the divergence
     */
    public static Var kl(Var a, Var b) {
        Var logP = a.sub(a.logsumexp(1, true));
        Var logQ = b.sub(b.logsumexp(1, true));
        return logP.exp().mul(logP.sub(logQ)).sum(1, false).mean();
    }

    private Tensor uniform(Shape s) {
        return Tensor.uniform(s, -1, 1, random);
    }

    private Tensor gaussian(Shape s) {
        return Tensor.normal(s, 0, 1, random);
    }

    private Tensor signNoise(Shape s) {
        return Tensor.uniform(s, -1, 1, random).sign();
    }
}
