package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * AlphaZero (Silver et al., 2017, <a href="https://arxiv.org/abs/1712.01815">arXiv:1712.01815</a>):
 * learning a board game from self-play alone. One network maps a position to a move prior
 * and a value; {@link Mcts} guided by it plays both sides; the network then learns to
 * predict the search's visit distribution (cross-entropy) and the final result (squared
 * error). Search improves the policy, learning distils the improvement, and the loop
 * repeats.
 *
 * <pre>{@code
 * AlphaZero az = AlphaZero.of(NInARow.ticTacToe(), Nets.mlp(18, 10, Activation.RELU, 1, 64, 64))
 *         .simulations(32);
 * az.train(10, 20);                                // 10 iterations of 20 self-play games
 * TwoPlayerGame.Player p = az.player(32);
 * }</pre>
 */
public final class AlphaZero {

    private record Sample(float[] input, float[] policy, float outcome) {
    }

    private final TwoPlayerGame start;
    private final Model network;
    private final int actions;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int simulations = 32;
    private int temperatureMoves = 4;
    private int epochs = 4;
    private int batchSize = 64;
    private int window = 4000;
    private float noiseAlpha = 0.5f;
    private float noiseFraction = 0.25f;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> replay = new ArrayList<>();
    private double lastLoss = Double.NaN;

    private AlphaZero(TwoPlayerGame start, Model network) {
        this.start = start;
        this.network = network;
        this.actions = start.actions();
        if (network.outputSize() != actions + 1) {
            throw new IllegalArgumentException("the network must output " + (actions + 1)
                    + " values: move logits then the value");
        }
    }

    /**
     * @param start the opening position
     * @param network position encoding to {@code actions} move logits followed by one value
     *     (squashed with tanh)
     * @return the trainer
     */
    public static AlphaZero of(TwoPlayerGame start, Model network) {
        return new AlphaZero(Objects.requireNonNull(start), Objects.requireNonNull(network));
    }

    /** @param n MCTS simulations per self-play move @return this trainer */
    public AlphaZero simulations(int n) {
        this.simulations = n;
        return this;
    }

    /** @param moves opening moves sampled from visit counts rather than played greedily @return this trainer */
    public AlphaZero temperatureMoves(int moves) {
        this.temperatureMoves = moves;
        return this;
    }

    /** @param n training passes over the replay window per iteration @return this trainer */
    public AlphaZero epochs(int n) {
        this.epochs = n;
        return this;
    }

    /** @param o the optimizer @return this trainer */
    public AlphaZero optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param alpha Dirichlet concentration of the root noise
     * @param fraction its weight
     * @return this trainer
     */
    public AlphaZero rootNoise(float alpha, float fraction) {
        this.noiseAlpha = alpha;
        this.noiseFraction = fraction;
        return this;
    }

    /** @param seed the seed @return this trainer */
    public AlphaZero seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the network */
    public Model network() {
        return network;
    }

    /** @return the loss of the last training batch */
    public double lastLoss() {
        return lastLoss;
    }

    /**
     * The network's judgement of a position.
     *
     * @param position a live position
     * @return masked, normalized prior and tanh value
     */
    public Mcts.Evaluation evaluate(TwoPlayerGame position) {
        float[] x = position.encode();
        float[] out = network.forward(Var.of(Tensor.of(Shape.of(1, x.length), x)), false).value()
                .to(DType.FLOAT32).toArray();
        boolean[] legal = position.legal();
        float max = Float.NEGATIVE_INFINITY;
        for (int a = 0; a < actions; a++) {
            if (legal[a]) {
                max = Math.max(max, out[a]);
            }
        }
        float[] p = new float[actions];
        float sum = 0;
        for (int a = 0; a < actions; a++) {
            if (legal[a]) {
                p[a] = (float) Math.exp(out[a] - max);
                sum += p[a];
            }
        }
        for (int a = 0; a < actions; a++) {
            p[a] /= sum;
        }
        return new Mcts.Evaluation(p, (float) Math.tanh(out[actions]), false);
    }

    private Mcts<TwoPlayerGame> search(int sims, boolean noise) {
        Mcts<TwoPlayerGame> m = Mcts.of(TwoPlayerGame.searchModel(actions, this::evaluate)).simulations(sims);
        return noise ? m.rootNoise(noiseAlpha, noiseFraction) : m;
    }

    /**
     * Plays one self-play game and stores its positions.
     *
     * @return the winner
     */
    public int selfPlay() {
        Mcts<TwoPlayerGame> mcts = search(simulations, true);
        List<float[]> inputs = new ArrayList<>();
        List<float[]> policies = new ArrayList<>();
        List<Integer> movers = new ArrayList<>();
        TwoPlayerGame g = start;
        int move = 0;
        while (!g.over()) {
            Mcts.Result r = mcts.search(g, random);
            inputs.add(g.encode());
            policies.add(r.policy());
            movers.add(g.toMove());
            g = g.play(r.sample(move < temperatureMoves ? 1 : 0, random));
            move++;
        }
        for (int i = 0; i < inputs.size(); i++) {
            replay.add(new Sample(inputs.get(i), policies.get(i), g.winner() * movers.get(i)));
        }
        while (replay.size() > window) {
            replay.removeFirst();
        }
        return g.winner();
    }

    /**
     * Runs the AlphaZero loop.
     *
     * @param iterations rounds of self-play followed by training
     * @param gamesPerIteration self-play games per round
     */
    public void train(int iterations, int gamesPerIteration) {
        for (int it = 0; it < iterations; it++) {
            for (int g = 0; g < gamesPerIteration; g++) {
                selfPlay();
            }
            learn();
        }
    }

    private void learn() {
        int n = replay.size();
        int in = replay.getFirst().input().length;
        int steps = Math.max(1, epochs * n / batchSize);
        for (int s = 0; s < steps; s++) {
            int b = Math.min(batchSize, n);
            float[] x = new float[b * in];
            float[] pi = new float[b * actions];
            float[] z = new float[b];
            for (int i = 0; i < b; i++) {
                Sample sm = replay.get(random.nextInt(n));
                System.arraycopy(sm.input(), 0, x, i * in, in);
                System.arraycopy(sm.policy(), 0, pi, i * actions, actions);
                z[i] = sm.outcome();
            }
            Var out = network.forward(Var.of(Tensor.of(Shape.of(b, in), x)), true);
            Var logits = out.slice(1, 0, actions);
            Var value = out.slice(1, actions, actions + 1).tanh();
            Var policyLoss = RlTensors.logSoftmax(logits).mul(Tensor.of(Shape.of(b, actions), pi)).sum(1, true)
                    .neg().mean();
            Var valueLoss = value.sub(Tensor.of(Shape.of(b, 1), z)).square().mean();
            Var loss = policyLoss.add(valueLoss);
            loss.backward();
            optimizer.step(network.parameters());
            lastLoss = loss.value().item();
        }
    }

    /**
     * @param sims MCTS simulations per move; 0 to play the network's prior greedily
     * @return a player using the trained network
     */
    public TwoPlayerGame.Player player(int sims) {
        if (sims <= 0) {
            return g -> RlTensors.argmax(evaluate(g).priors());
        }
        Mcts<TwoPlayerGame> m = search(sims, false);
        SplittableRandom r = new SplittableRandom(7);
        return g -> m.search(g, r).best();
    }
}
