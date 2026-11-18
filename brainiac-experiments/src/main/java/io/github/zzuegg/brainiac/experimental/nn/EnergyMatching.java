package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * Energy matching (Balcerak et al., <a href="https://arxiv.org/abs/2504.10612">arXiv:2504.10612</a>):
 * one time-independent scalar potential {@code V(x)} whose negative gradient is the sampler's
 * velocity everywhere. Far from the data, {@code -grad V} is trained to follow the
 * optimal-transport flow from noise to data (flow matching on minibatch-OT pairs, with the
 * target velocity tapering to zero over the last stretch of the path so the data become
 * equilibria, as in equilibrium matching); near the data a contrastive term
 * {@code V(data) - V(negatives)}, with negatives from short Langevin chains started at the
 * data, shapes {@code V} into a Boltzmann energy, {@code p(x) ~ exp(-V(x))}. One network is
 * then both a fast transport sampler and an unnormalised density model, which flow matching
 * (a velocity field, no energy) and equilibrium matching (no density) are not.
 *
 * <p>Sampling is gradient descent on {@code V} from noise, optionally ending with Langevin
 * steps at a temperature.
 */
public final class EnergyMatching {

    /** A scalar potential network: points {@code [n, dim]} to energies {@code [n, 1]}. */
    public interface Potential extends UnaryOperator<Var> {
        /** @return the trainable parameters */
        List<Var> parameters();
    }

    private final Potential potential;
    private final float decayStart;
    private final float contrastiveWeight;
    private final int langevinSteps;
    private final float langevinStep;

    private EnergyMatching(Potential potential, float decayStart, float contrastiveWeight, int langevinSteps,
            float langevinStep) {
        this.potential = potential;
        this.decayStart = decayStart;
        this.contrastiveWeight = contrastiveWeight;
        this.langevinSteps = langevinSteps;
        this.langevinStep = langevinStep;
    }

    /**
     * Creates the trainer.
     *
     * @param potential {@code V}: points {@code [n, dim]} to energies {@code [n, 1]}, built from
     *     twice-differentiable operations
     * @param decayStart the path time after which the target velocity tapers to 0 at the data
     * @param contrastiveWeight the weight of the contrastive (Boltzmann) term, 0 for none
     * @param langevinSteps steps of each negative chain
     * @param langevinStep the Langevin step size {@code eta}: {@code x -= eta grad V + sqrt(2 eta) n}
     * @return the trainer
     */
    public static EnergyMatching of(Potential potential, float decayStart, float contrastiveWeight,
            int langevinSteps, float langevinStep) {
        return new EnergyMatching(potential, decayStart, contrastiveWeight, langevinSteps, langevinStep);
    }

    /**
     * An MLP potential with SiLU hidden layers (twice differentiable, as training needs).
     *
     * @param dim the input dimension
     * @param width the hidden width
     * @param depth the number of hidden layers, at least 1
     * @param seed the initialization seed
     * @return the potential
     */
    public static Potential mlp(int dim, int width, int depth, long seed) {
        int[] sizes = new int[depth + 2];
        sizes[0] = dim;
        java.util.Arrays.fill(sizes, 1, depth + 1, width);
        sizes[depth + 1] = 1;
        GenerativeOps.Mlp net = new GenerativeOps.Mlp(new java.util.SplittableRandom(seed), false, sizes);
        return new Potential() {
            @Override
            public Var apply(Var x) {
                return net.apply(x);
            }

            @Override
            public List<Var> parameters() {
                return net.parameters();
            }
        };
    }

    /** @return the potential */
    public Potential potential() {
        return potential;
    }

    /**
     * The energy of points.
     *
     * @param x points {@code [n, dim]}
     * @return {@code [n]}
     */
    public Tensor energy(Tensor x) {
        return potential.apply(Var.of(x)).value().reshape(x.shape().dim(0));
    }

    /**
     * {@code grad V(x)}.
     *
     * @param x points {@code [n, dim]}
     * @return the gradient, the shape of {@code x}
     */
    public Tensor gradient(Tensor x) {
        Var xv = Var.param(x);
        return Var.gradients(potential.apply(xv).sum(), List.of(xv), false).get(0).value();
    }

    /**
     * The training loss on a batch: OT flow matching of {@code -grad V} plus the contrastive term.
     *
     * @param data data {@code [n, dim]}
     * @param random the source of noise, times and Langevin noise
     * @return the scalar loss
     */
    public Var loss(Tensor data, RandomGenerator random) {
        int n = data.shape().dim(0);
        int dim = data.shape().dim(1);
        Tensor noise = Tensor.normal(data.shape(), 0, 1, random);
        float[][] paired = MinibatchOptimalTransport.pair(GenerativeOps.rows(data), GenerativeOps.rows(noise));
        Tensor z = GenerativeOps.matrix(paired);
        float[] t = new float[n];
        float[] scale = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) random.nextDouble();
            scale[i] = t[i] < decayStart ? 1 : (1 - t[i]) / (1 - decayStart);
        }
        Tensor tc = GenerativeOps.column(t);
        // noise at t = 0, data at t = 1: the path an energy sampler walks
        Tensor xt = z.mul(tc.neg().add(1f)).add(data.mul(tc));
        Tensor target = data.sub(z).mul(GenerativeOps.column(scale));
        Var xv = Var.param(xt);
        Var grad = Var.gradients(potential.apply(xv).sum(), List.of(xv), true).get(0);
        Var loss = grad.neg().sub(Var.of(target)).square().sum(1, false).mean();
        if (contrastiveWeight > 0) {
            Tensor negatives = data.add(Tensor.normal(data.shape(), 0, 0.1f, random));
            for (int k = 0; k < langevinSteps; k++) {
                negatives = negatives.sub(gradient(negatives).mul(langevinStep))
                        .add(Tensor.normal(Shape.of(n, dim), 0, (float) Math.sqrt(2 * langevinStep), random));
            }
            Var pos = potential.apply(Var.of(data)).mean();
            Var neg = potential.apply(Var.of(negatives)).mean();
            // a small square penalty keeps the energies from drifting off together
            Var reg = potential.apply(Var.of(data)).square().mean().mul(0.01f);
            loss = loss.add(pos.sub(neg).add(reg).mul(contrastiveWeight));
        }
        return loss;
    }

    /**
     * Samples by gradient descent on the potential, then Langevin steps.
     *
     * @param noise standard normal starts {@code [n, dim]}
     * @param steps gradient-descent steps along {@code -grad V} over unit path time
     * @param langevin Langevin steps at the end, 0 for none
     * @param temperature their temperature
     * @param random the Langevin noise source
     * @return the samples; {@code steps + langevin} gradient evaluations
     */
    public Tensor sample(Tensor noise, int steps, int langevin, float temperature, RandomGenerator random) {
        Tensor x = noise;
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            x = x.sub(gradient(x).mul(dt));
        }
        for (int k = 0; k < langevin; k++) {
            x = x.sub(gradient(x).mul(langevinStep))
                    .add(Tensor.normal(x.shape(), 0, (float) Math.sqrt(2 * langevinStep * temperature), random));
        }
        return x;
    }
}
