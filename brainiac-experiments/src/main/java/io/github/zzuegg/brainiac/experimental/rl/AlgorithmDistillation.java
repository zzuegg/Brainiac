package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Bandit;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Algorithm distillation (Laskin et al., 2022, <a href="https://arxiv.org/abs/2210.14215">arXiv:2210.14215</a>):
 * in-context reinforcement learning on multi-armed {@link Bandit} tasks. A causal
 * transformer is trained, by plain next-action prediction, on whole <em>learning
 * histories</em> of a source RL algorithm, each on a different task. Because a history
 * shows the policy getting better, predicting the next action well means imitating the
 * improvement operator itself: on a new task, the frozen model explores and then
 * exploits in context, with no weight updates.
 *
 * <p>Token {@code t} embeds the previous pull (one-hot arm and reward) plus a learned
 * position; the output at token {@code t} is the distribution of the next arm. The same
 * model trained on expert histories only (always the best arm) is the expert-distillation
 * baseline: it never saw exploration, so it cannot find the good arm on its own. The
 * transformer is built like {@link DecisionTransformer}'s (core causal
 * {@link TransformerBlock}s), whose internals are private.
 */
public final class AlgorithmDistillation implements Agent {

    /** A learning history: the arms pulled and the rewards they paid. @param arms arms @param rewards rewards */
    public record History(int[] arms, float[] rewards) {
    }

    /** A source policy for collecting histories: picks the next arm from the history so far. */
    @FunctionalInterface
    public interface Source {

        /**
         * @param arms the arms pulled so far (only the first {@code t} entries are valid)
         * @param rewards their rewards
         * @param t pulls so far
         * @param task the task (an expert may peek; a learner must not)
         * @param random the random source
         * @return the next arm
         */
        int next(int[] arms, float[] rewards, int t, Bandit task, SplittableRandom random);
    }

    private final int arms;
    private final int horizon;
    private final int width;
    private final Var embed;
    private final Var position;
    private final Var head;
    private final List<Layer> blocks = new ArrayList<>();
    private final Optimizer optimizer = Optimizer.adamW(1e-3f, 1e-4f);
    private int batchSize = 16;
    private SplittableRandom random;
    private final List<Integer> pastArms = new ArrayList<>();
    private final List<Float> pastRewards = new ArrayList<>();
    private double lastLoss = Double.NaN;

    private AlgorithmDistillation(int arms, int horizon, int width, int layers, long seed) {
        this.arms = arms;
        this.horizon = horizon;
        this.width = width;
        this.random = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        embed = Var.param(Tensor.normal(Shape.of(arms + 1, width), 0, 0.1f, r));
        position = Var.param(Tensor.normal(Shape.of(horizon, width), 0, 0.1f, r));
        head = Var.param(Tensor.normal(Shape.of(width, arms), 0, 0.1f, r));
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(2).causal(true).ffnSize(2 * width).create(Shape.of(horizon, width), r));
        }
    }

    /**
     * @param arms arms per bandit
     * @param horizon pulls per history (the context length)
     * @param width the model width
     * @param layers transformer blocks
     * @param seed the seed
     * @return the model
     */
    public static AlgorithmDistillation of(int arms, int horizon, int width, int layers, long seed) {
        return new AlgorithmDistillation(arms, horizon, width, layers, seed);
    }

    /** @param size histories per gradient step @return this model */
    public AlgorithmDistillation batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, position, head));
        for (Layer b : blocks) {
            p.addAll(b.parameters());
        }
        return p;
    }

    /**
     * Runs a source on fresh tasks and records its histories.
     *
     * @param tasks draws a task
     * @param source the policy
     * @param count histories
     * @param horizon pulls per history
     * @param seed the seed
     * @return the histories
     */
    public static List<History> collect(Function<SplittableRandom, Bandit> tasks, Source source, int count,
            int horizon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<History> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            Bandit task = tasks.apply(r);
            int[] a = new int[horizon];
            float[] rw = new float[horizon];
            for (int t = 0; t < horizon; t++) {
                a[t] = source.next(a, rw, t, task, r);
                task.reset(r.split());
                rw[t] = task.step(new float[] {a[t]}).reward();
            }
            out.add(new History(a, rw));
        }
        return out;
    }

    /** Next-arm logits at every position, {@code [b, horizon, arms]}. */
    private Var logits(float[] tokens, int b, boolean training) {
        Var x = Var.of(Tensor.of(Shape.of(b, horizon, arms + 1), tokens)).matmul(embed).add(position);
        for (Layer block : blocks) {
            x = block.forward(x, training);
        }
        return x.matmul(head);
    }

    /**
     * Trains next-arm prediction on histories.
     *
     * @param data the histories (each {@code horizon} pulls long)
     * @param steps gradient steps
     * @return this model
     */
    public AlgorithmDistillation train(List<History> data, int steps) {
        int w = arms + 1;
        for (int it = 0; it < steps; it++) {
            float[] tokens = new float[batchSize * horizon * w];
            float[] target = new float[batchSize * horizon * arms];
            for (int i = 0; i < batchSize; i++) {
                History h = data.get(random.nextInt(data.size()));
                for (int t = 0; t < horizon; t++) {
                    int row = i * horizon + t;
                    if (t > 0) {
                        tokens[row * w + h.arms()[t - 1]] = 1;
                        tokens[row * w + arms] = h.rewards()[t - 1];
                    }
                    target[row * arms + h.arms()[t]] = 1;
                }
            }
            Var loss = RlTensors.logSoftmax(logits(tokens, batchSize, true))
                    .mul(Tensor.of(Shape.of(batchSize, horizon, arms), target)).sum().neg()
                    .div(batchSize * horizon);
            loss.backward();
            RlTensors.clip(parameters(), 1f);
            optimizer.step(parameters());
            lastLoss = loss.value().item();
        }
        return this;
    }

    /** Forgets the current task. */
    public void newTask() {
        pastArms.clear();
        pastRewards.clear();
    }

    /**
     * Tells the model the reward of its last pull.
     *
     * @param reward the reward
     */
    public void observe(float reward) {
        pastRewards.add(reward);
    }

    /** Picks the next arm from the in-context history; the observation is ignored. */
    @Override
    public float[] act(float[] observation, boolean explore) {
        int t = pastArms.size();
        if (t >= horizon) {
            throw new IllegalStateException("the context holds " + horizon + " pulls; call newTask");
        }
        int w = arms + 1;
        float[] tokens = new float[horizon * w];
        for (int j = 1; j <= t; j++) {
            tokens[j * w + pastArms.get(j - 1)] = 1;
            tokens[j * w + arms] = pastRewards.get(j - 1);
        }
        float[] all = logits(tokens, 1, false).value().to(DType.FLOAT32).toArray();
        float[] l = new float[arms];
        System.arraycopy(all, t * arms, l, 0, arms);
        int a = explore ? sample(l) : RlTensors.argmax(l);
        pastArms.add(a);
        return new float[] {a};
    }

    private int sample(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double z = 0;
        for (float v : logits) {
            z += Math.exp(v - max);
        }
        double u = random.nextDouble() * z;
        for (int i = 0; i < logits.length; i++) {
            u -= Math.exp(logits[i] - max);
            if (u <= 0) {
                return i;
            }
        }
        return logits.length - 1;
    }

    /**
     * Plays fresh tasks in context, sampling arms, without weight updates.
     *
     * @param tasks draws a task
     * @param count tasks
     * @param seed the seed
     * @return the mean reward at every pull position (the in-context learning curve)
     */
    public double[] evaluate(Function<SplittableRandom, Bandit> tasks, int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] curve = new double[horizon];
        for (int i = 0; i < count; i++) {
            Bandit task = tasks.apply(r);
            newTask();
            for (int t = 0; t < horizon; t++) {
                int a = Math.round(act(new float[] {1}, true)[0]);
                task.reset(r.split());
                float rw = task.step(new float[] {a}).reward();
                observe(rw);
                curve[t] += rw / count;
            }
        }
        return curve;
    }

    @Override
    public String toString() {
        return "AlgorithmDistillation(" + arms + " arms, context " + horizon + ", width " + width + ")";
    }
}
