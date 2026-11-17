package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Decision Transformer (Chen et al., 2021, <a href="https://arxiv.org/abs/2106.01345">arXiv:2106.01345</a>):
 * offline RL as conditional sequence modelling. Trajectories become token sequences
 * {@code (R_1, s_1, a_1, R_2, s_2, a_2, ...)} where {@code R_t} is the return still to
 * come; a causal transformer learns to predict each action from the tokens before it.
 * At test time, prompting with a high target return makes it produce the actions that,
 * in the data, led to such returns: no value functions, no bootstrapping, just
 * supervised learning.
 *
 * <p>The agent is stateful during an episode: call {@link #begin(float)} with the
 * target return at the start, {@link #act} for each step, and {@link #observe(float)}
 * with every reward, or simply use {@link #play}.
 */
public final class DecisionTransformer implements Agent {

    private final int obsDims;
    private final int actions;
    private final int context;
    private final int maxSteps;
    private final int width;
    private final Var returnEmbed;
    private final Var stateEmbed;
    private final Var actionEmbed;
    private final Var timeEmbed;
    private final Var bias;
    private final Var head;
    private final List<Layer> blocks = new ArrayList<>();
    private Optimizer optimizer = Optimizer.adamW(1e-3f, 1e-4f);
    private float returnScale = 1f;
    private int batchSize = 32;
    private SplittableRandom random = new SplittableRandom(0);
    private double lastLoss = Double.NaN;
    private final List<float[]> histStates = new ArrayList<>();
    private final List<Integer> histActions = new ArrayList<>();
    private final List<Float> histReturns = new ArrayList<>();
    private float returnToGo;

    private DecisionTransformer(int obsDims, int actions, int context, int maxSteps, int width, int layers,
            int heads, long seed) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.context = context;
        this.maxSteps = maxSteps;
        this.width = width;
        SplittableRandom r = new SplittableRandom(seed);
        float std = 0.1f;
        returnEmbed = Var.param(Tensor.normal(Shape.of(1, width), 0, std, r));
        stateEmbed = Var.param(Tensor.normal(Shape.of(obsDims, width), 0, std, r));
        actionEmbed = Var.param(Tensor.normal(Shape.of(actions, width), 0, std, r));
        timeEmbed = Var.param(Tensor.normal(Shape.of(maxSteps + 1, width), 0, std, r));
        bias = Var.param(Tensor.zeros(Shape.of(3, width)));
        head = Var.param(Tensor.normal(Shape.of(width, actions), 0, std, r));
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(heads).causal(true).ffnSize(2 * width)
                    .create(Shape.of(3 * context, width), r));
        }
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param context timesteps of history the model sees ({@code 3 * context} tokens)
     * @param maxSteps the longest episode (for timestep embeddings)
     * @param width the model width
     * @param layers transformer blocks
     * @param seed the initialization seed
     * @return the model
     */
    public static DecisionTransformer of(int observationDims, int actions, int context, int maxSteps, int width,
            int layers, long seed) {
        return new DecisionTransformer(observationDims, actions, context, maxSteps, width, layers, 2, seed);
    }

    /** @param scale returns are divided by this before embedding @return this model */
    public DecisionTransformer returnScale(float scale) {
        this.returnScale = scale;
        return this;
    }

    /** @param o the optimizer @return this model */
    public DecisionTransformer optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @param seed the sampling seed @return this model */
    public DecisionTransformer seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(returnEmbed, stateEmbed, actionEmbed, timeEmbed, bias, head));
        for (Layer b : blocks) {
            p.addAll(b.parameters());
        }
        return p;
    }

    /** Action logits at every state token, {@code [batch, context, actions]}. */
    private Var logits(float[] rtg, float[] states, float[] actionHot, float[] timeHot, int b, boolean training) {
        int k = context;
        Var r = Var.of(Tensor.of(Shape.of(b, k, 1), rtg)).matmul(returnEmbed);
        Var s = Var.of(Tensor.of(Shape.of(b, k, obsDims), states)).matmul(stateEmbed);
        Var a = Var.of(Tensor.of(Shape.of(b, k, actions), actionHot)).matmul(actionEmbed);
        Var t = Var.of(Tensor.of(Shape.of(b, k, maxSteps + 1), timeHot)).matmul(timeEmbed);
        Var tokens = Var.concat(2, r.add(t).reshape(b, k, 1, width), s.add(t).reshape(b, k, 1, width),
                a.add(t).reshape(b, k, 1, width)).add(bias).reshape(b, 3 * k, width);
        Var h = tokens;
        for (Layer block : blocks) {
            h = block.forward(h, training);
        }
        return h.reshape(b, k, 3, width).slice(2, 1, 2).reshape(b, k, width).matmul(head);
    }

    /**
     * Trains on logged trajectories.
     *
     * @param data the dataset (discrete actions)
     * @param steps gradient steps
     * @return this model
     */
    public DecisionTransformer train(OfflineDataset data, int steps) {
        List<List<OfflineDataset.Step>> trajs = data.trajectories();
        int k = context;
        for (int it = 0; it < steps; it++) {
            int b = batchSize;
            float[] rtg = new float[b * k];
            float[] states = new float[b * k * obsDims];
            float[] act = new float[b * k * actions];
            float[] time = new float[b * k * (maxSteps + 1)];
            float[] target = new float[b * k * actions];
            for (int i = 0; i < b; i++) {
                List<OfflineDataset.Step> t = trajs.get(random.nextInt(trajs.size()));
                float[] togo = new float[t.size()];
                float acc = 0;
                for (int j = t.size() - 1; j >= 0; j--) {
                    acc += t.get(j).reward();
                    togo[j] = acc;
                }
                int end = random.nextInt(t.size());
                for (int slot = 0; slot < k; slot++) {
                    int j = end - (k - 1 - slot);
                    if (j < 0) {
                        continue;
                    }
                    int row = i * k + slot;
                    rtg[row] = togo[j] / returnScale;
                    System.arraycopy(t.get(j).observation(), 0, states, row * obsDims, obsDims);
                    int a = Math.round(t.get(j).action()[0]);
                    act[row * actions + a] = 1;
                    target[row * actions + a] = 1;
                    time[row * (maxSteps + 1) + Math.min(j, maxSteps)] = 1;
                }
            }
            Var logits = logits(rtg, states, act, time, b, true);
            Var loss = RlTensors.logSoftmax(logits).mul(Tensor.of(Shape.of(b, k, actions), target)).sum()
                    .neg().div(b * k);
            loss.backward();
            RlTensors.clip(parameters(), 1f);
            optimizer.step(parameters());
            lastLoss = loss.value().item();
        }
        return this;
    }

    /**
     * Starts an episode.
     *
     * @param targetReturn the return to ask for
     */
    public void begin(float targetReturn) {
        histStates.clear();
        histActions.clear();
        histReturns.clear();
        returnToGo = targetReturn;
    }

    /**
     * Tells the model the reward of its last action.
     *
     * @param reward the reward
     */
    public void observe(float reward) {
        returnToGo -= reward;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        histStates.add(observation.clone());
        histReturns.add(returnToGo);
        int n = histStates.size();
        int k = context;
        float[] rtg = new float[k];
        float[] states = new float[k * obsDims];
        float[] act = new float[k * actions];
        float[] time = new float[k * (maxSteps + 1)];
        int last = 0;
        for (int slot = 0; slot < k; slot++) {
            int j = n - k + slot;
            if (j < 0) {
                continue;
            }
            rtg[slot] = histReturns.get(j) / returnScale;
            System.arraycopy(histStates.get(j), 0, states, slot * obsDims, obsDims);
            if (j < histActions.size()) {
                act[slot * actions + histActions.get(j)] = 1;
            }
            time[slot * (maxSteps + 1) + Math.min(j, maxSteps)] = 1;
            last = slot;
        }
        float[] all = logits(rtg, states, act, time, 1, false).value().to(DType.FLOAT32).toArray();
        float[] l = new float[actions];
        System.arraycopy(all, last * actions, l, 0, actions);
        int a = RlTensors.argmax(l);
        histActions.add(a);
        return new float[] {a};
    }

    /**
     * Plays one greedy episode asking for a return.
     *
     * @param env the environment
     * @param targetReturn the return to ask for
     * @param seed the episode seed
     * @return the return obtained
     */
    public double play(Environment env, float targetReturn, long seed) {
        begin(targetReturn);
        float[] obs = env.reset(new SplittableRandom(seed));
        double total = 0;
        Environment.Step s;
        do {
            s = env.step(act(obs, false));
            observe(s.reward());
            total += s.reward();
            obs = s.observation();
        } while (!s.done());
        return total;
    }
}
