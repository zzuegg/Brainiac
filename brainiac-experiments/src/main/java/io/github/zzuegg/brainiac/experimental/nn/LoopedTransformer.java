package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A looped transformer with ticks (experiment X11): one block applied up to
 * {@code maxTicks} times, with a learned per-token halting decision, so easy tokens stop
 * early and hard tokens "think" longer with the same parameters. Before each tick a
 * sinusoidal tick embedding is added, so the block knows which iteration it is in.
 * {@link #lastTicks()} reports how many ticks every token used.
 *
 * <p>Two halting schemes:
 * <ul>
 *   <li>{@link Halting#ACT} (Graves): a halting unit emits {@code p_n = sigmoid(h w + b)} per
 *       tick; a token halts once its cumulative probability would pass {@code 1 - epsilon}, using
 *       the remainder {@code R} as its last weight. The output is {@code sum_n p_n h_n}, and the
 *       ponder cost {@code N + R} ({@link #ponderLoss()}) is added to the loss to pay for
 *       thinking time.</li>
 *   <li>{@link Halting#PONDER} (PonderNet): {@code lambda_n} is the conditional halting
 *       probability, so {@code p_n = lambda_n prod_{j<n} (1 - lambda_j)} is a proper
 *       distribution over ticks. The layer outputs the expected state {@code sum_n p_n h_n};
 *       {@link #ponderLoss()} is {@code KL(p || Geometric(prior))}. Callers wanting the exact
 *       PonderNet objective {@code sum_n p_n L(y_n)} can read {@link #tickStates()} and
 *       {@link #haltingDistribution()}. At inference, a token stops at the first tick where its
 *       cumulative halting probability passes one half.</li>
 * </ul>
 *
 * <p>The block maps a state {@code [b, t, d]} to the next state of the same shape (for
 * example a core {@code TransformerBlock}, which includes its residual connection). Halted
 * tokens keep their state; the loop ends early once every token has halted.
 *
 * <p>Papers: Graves, "Adaptive Computation Time for Recurrent Neural Networks",
 * <a href="https://arxiv.org/abs/1603.08983">arXiv:1603.08983</a>; Banino et al., "PonderNet:
 * Learning to Ponder", <a href="https://arxiv.org/abs/2107.05407">arXiv:2107.05407</a>.
 */
public final class LoopedTransformer implements Layer {

    /** How tokens decide to stop. */
    public enum Halting {
        /** Adaptive Computation Time with a remainder and a ponder cost. */
        ACT,
        /** PonderNet: a halting distribution regularized towards a geometric prior. */
        PONDER
    }

    private final Layer block;
    private final int features;
    private final int maxTicks;
    private final Halting halting;
    private final float epsilon;
    private final float prior;
    private final Var haltWeight;
    private final Var haltBias;
    private final Tensor tickEmbedding;
    private Var lastPonderLoss;
    private Var lastAuxiliary;
    private float ponderCost;
    private float[] lastTicks;
    private List<Var> lastStates = List.of();
    private List<Var> lastDistribution = List.of();

    /**
     * Creates the loop.
     *
     * @param block the shared block, {@code [b, t, features] -> [b, t, features]}
     * @param features the width
     * @param maxTicks the most iterations
     * @param halting the halting scheme
     * @param random the random source
     */
    public LoopedTransformer(Layer block, int features, int maxTicks, Halting halting, RandomGenerator random) {
        if (maxTicks < 1) {
            throw new IllegalArgumentException("maxTicks must be positive");
        }
        this.block = block;
        this.features = features;
        this.maxTicks = maxTicks;
        this.halting = halting;
        this.epsilon = 0.01f;
        this.prior = 1f / Math.max(2, maxTicks / 2f);
        this.haltWeight = MixerOps.uniform(random, 0.01f, features, 1);
        // start by halting slowly: sigmoid(-1) ~ 0.27
        this.haltBias = MixerOps.filled(-1f, 1);
        this.tickEmbedding = Functional.sinusoidal(maxTicks, features, 10000f, Tensor.zeros(Shape.of(1)).device());
    }

    /**
     * Describes the loop for {@code Model.sequential().add(...)}.
     *
     * @param block the spec of the shared block
     * @param maxTicks the most iterations
     * @param halting the halting scheme
     * @return the spec
     */
    public static LayerSpec of(LayerSpec block, int maxTicks, Halting halting) {
        return MixerOps.sameShape((in, r) -> new LoopedTransformer(block.create(in, r), in.dim(-1), maxTicks, halting, r));
    }

    /**
     * Ticks per token in the last forward pass: {@code N + R} for ACT while training (the
     * ponder cost), the expected tick count for PonderNet while training, and the whole number
     * of ticks run at inference.
     *
     * @return {@code [batch * steps]} tick counts
     */
    public float[] lastTicks() {
        return lastTicks.clone();
    }

    /** @return the mean of {@link #lastTicks()} */
    public float meanTicks() {
        float s = 0;
        for (float t : lastTicks) {
            s += t;
        }
        return s / lastTicks.length;
    }

    /** @return the ponder cost (ACT) or KL to the geometric prior (PonderNet) of the last pass */
    public Var ponderLoss() {
        return lastPonderLoss;
    }

    /**
     * Sets the weight of {@link #ponderLoss()} in this layer's {@link #auxiliaryLoss()}, which
     * the core trainer adds to the loss.
     *
     * @param weight the factor, for example {@code 0.001} for ACT's time penalty; 0 (the
     *     default) reports none
     * @return this layer
     */
    public LoopedTransformer ponderCost(float weight) {
        this.ponderCost = weight;
        return this;
    }

    /** @return {@code ponderCost * ponderLoss()} of the last training pass, or {@code null} */
    @Override
    public Var auxiliaryLoss() {
        return lastAuxiliary;
    }

    /** @return the state after each tick of the last pass, each {@code [b, t, d]} */
    public List<Var> tickStates() {
        return lastStates;
    }

    /** @return the halting probability of each tick of the last pass, each {@code [b, t]} */
    public List<Var> haltingDistribution() {
        return lastDistribution;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "LoopedTransformer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int n = b * t;
        Tensor ones = Tensor.ones(Shape.of(b, t));
        Tensor running = ones;            // 1 while a token still runs
        Var cumulative = Var.of(Tensor.zeros(Shape.of(b, t)));
        Var notHalted = Var.of(ones);     // PonderNet: prod (1 - lambda)
        Var output = Var.of(Tensor.zeros(input.shape()));
        Var ponder = Var.of(Tensor.zeros(Shape.of(b, t)));
        float[] ticks = new float[n];
        List<Var> states = new ArrayList<>();
        List<Var> dist = new ArrayList<>();
        Var h = input;
        for (int tick = 0; tick < maxTicks; tick++) {
            Var next = block.forward(h.add(tickEmbedding.slice(0, tick, tick + 1)), training);
            Var mask3 = Var.of(running.reshape(b, t, 1));
            // halted tokens keep their state
            h = next.mul(mask3).add(h.mul(mask3.neg().add(1f)));
            states.add(h);
            Var lambda = h.matmul(haltWeight).add(haltBias).sigmoid().reshape(b, t);
            boolean last = tick == maxTicks - 1;
            float[] run = running.toArray();
            Var weight;
            if (halting == Halting.ACT) {
                float[] cum = cumulative.value().toArray();
                float[] lam = lambda.value().toArray();
                float[] stop = new float[n];
                for (int i = 0; i < n; i++) {
                    stop[i] = run[i] > 0 && (last || cum[i] + lam[i] > 1 - epsilon) ? 1 : 0;
                }
                Tensor stopT = Tensor.of(Shape.of(b, t), stop);
                Tensor cont = running.sub(stopT);
                Var remainder = cumulative.neg().add(1f);
                weight = lambda.mul(cont).add(remainder.mul(stopT));
                cumulative = cumulative.add(lambda.mul(cont));
                ponder = ponder.add(Var.of(cont)).add(remainder.add(1f).mul(stopT));   // N + R at the stop
                for (int i = 0; i < n; i++) {
                    ticks[i] += run[i];
                }
                running = cont;
            } else {
                Var p = last ? notHalted : notHalted.mul(lambda);
                weight = p;
                notHalted = last ? notHalted : notHalted.mul(lambda.neg().add(1f));
                float[] cumBefore = cumulative.value().toArray();
                cumulative = cumulative.add(p);
                float[] cumAfter = cumulative.value().toArray();
                float[] keep = new float[n];
                for (int i = 0; i < n; i++) {
                    if (training) {
                        ticks[i] += (1 - cumBefore[i]);   // expected ticks = sum_n P(still running at n)
                        keep[i] = 1;
                    } else {
                        ticks[i] += run[i];
                        keep[i] = run[i] > 0 && cumAfter[i] < 0.5f ? 1 : 0;
                    }
                }
                running = training ? ones : Tensor.of(Shape.of(b, t), keep);
            }
            dist.add(weight);
            output = output.add(h.mul(weight.reshape(b, t, 1)));
            boolean anyRunning = false;
            for (float r : running.toArray()) {
                anyRunning |= r > 0;
            }
            if (!anyRunning) {
                break;
            }
        }
        if (!training && halting == Halting.PONDER) {
            output = h;   // inference: each token's state froze at its stopping tick
        }
        if (halting == Halting.ACT) {
            lastPonderLoss = ponder.mean();
        } else {
            // KL(p || geometric(prior)) over the ticks that were run
            Var kl = Var.of(Tensor.zeros(Shape.of(b, t)));
            for (int i = 0; i < dist.size(); i++) {
                double logPrior = Math.log(prior) + i * Math.log(1 - prior);
                Var p = dist.get(i);
                kl = kl.add(p.mul(p.add(1e-8f).log().sub((float) logPrior)));
            }
            lastPonderLoss = kl.mean();
        }
        lastAuxiliary = training && ponderCost != 0 ? lastPonderLoss.mul(ponderCost) : null;
        lastTicks = ticks;
        lastStates = states;
        lastDistribution = dist;
        return output;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(haltWeight, haltBias));
        p.addAll(block.parameters());
        return p;
    }

    @Override
    public List<Var> buffers() {
        return block.buffers();
    }

    @Override
    public String toString() {
        return "LoopedTransformer(" + block + ", maxTicks=" + maxTicks + ", " + halting + ")";
    }
}
