package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.MetaRlAgent;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.experimental.train.Maml;
import io.github.zzuegg.brainiac.rl.Bandit;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * W6 3.16: meta-RL on a bandit task distribution. Every task is a 5-armed bandit with one
 * good arm (mean 1, the others 0, reward noise 0.5) at a random position; the agent gets a
 * short horizon of pulls. Metric: the mean reward per pull over the horizon on fresh tasks.
 * Compared: RL² and VariBAD (meta-trained, no weight updates on the new task), a
 * MAML-learned softmax policy adapted by one policy-gradient step halfway, and the core PPO
 * learning from scratch on the task.
 *
 * <p>Papers: Duan et al., <a href="https://arxiv.org/abs/1611.02779">arXiv:1611.02779</a> (RL²);
 * Zintgraf et al., <a href="https://arxiv.org/abs/1910.08348">arXiv:1910.08348</a> (VariBAD);
 * Finn et al., <a href="https://arxiv.org/abs/1703.03400">arXiv:1703.03400</a> (MAML).
 */
public final class MetaRlBandits {

    /** Arms per task. */
    public static final int ARMS = 5;

    private MetaRlBandits() {
    }

    /**
     * @param r the random source
     * @return a task: one good arm at a random position
     */
    public static Bandit task(SplittableRandom r) {
        double[] means = new double[ARMS];
        means[r.nextInt(ARMS)] = 1;
        return new Bandit(means, 0.5);
    }

    /**
     * @param horizon pulls per task
     * @param iterations meta-training batches for RL² and VariBAD (32 tasks each)
     * @param evalTasks fresh tasks to evaluate on
     * @param seed the seed
     * @return method name to mean reward per pull
     */
    public static Map<String, Double> run(int horizon, int iterations, int evalTasks, long seed) {
        Function<SplittableRandom, Bandit> dist = MetaRlBandits::task;
        Map<String, Double> out = new LinkedHashMap<>();
        MetaRlAgent rl2 = MetaRlAgent.rl2(ARMS, horizon, 32, seed);
        rl2.metaTrain(dist, iterations);
        out.put("RL2", rl2.evaluate(dist, evalTasks, seed + 100, true));
        MetaRlAgent vb = MetaRlAgent.variBad(ARMS, horizon, 32, 4, seed);
        vb.metaTrain(dist, iterations);
        out.put("VariBAD", vb.evaluate(dist, evalTasks, seed + 100, true));
        out.put("MAML policy gradient", maml(dist, horizon, iterations, evalTasks, seed));
        double ppo = 0;
        SplittableRandom r = new SplittableRandom(seed + 100);
        for (int i = 0; i < evalTasks; i++) {
            Bandit env = dist.apply(r);
            Ppo p = Ppo.of(Nets.mlp(1, ARMS, seed, 16), Nets.mlp(1, 1, seed + 1, 16), env.actionSpace())
                    .rolloutSteps(Math.max(2, horizon / 4)).minibatchSize(Math.max(2, horizon / 4)).seed(seed + i);
            ppo += p.fit(env, horizon).meanReturn(horizon);
        }
        out.put("PPO from scratch (core)", ppo / evalTasks);
        out.put("random", 1.0 / ARMS);
        return out;
    }

    /** One MAML task: REINFORCE on pulls of the current softmax policy. */
    private static final class PullTask implements Maml.Task {

        private final Var logits;
        private final Bandit env;
        private final SplittableRandom random;
        private final int pulls;
        private float[] fixedArms;
        private float[] fixedRewards;

        PullTask(Var logits, Bandit env, SplittableRandom random, int pulls) {
            this.logits = logits;
            this.env = env;
            this.random = random;
            this.pulls = pulls;
        }

        /** Samples pulls from the current policy: arms then rewards. */
        float[][] play() {
            float[] l = logits.value().to(DType.FLOAT32).toArray();
            float[] arms = new float[pulls];
            float[] rewards = new float[pulls];
            for (int i = 0; i < pulls; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (float v : l) {
                    max = Math.max(max, v);
                }
                double z = 0;
                for (float v : l) {
                    z += Math.exp(v - max);
                }
                double u = random.nextDouble() * z;
                int a = l.length - 1;
                for (int k = 0; k < l.length; k++) {
                    u -= Math.exp(l[k] - max);
                    if (u <= 0) {
                        a = k;
                        break;
                    }
                }
                env.reset(random.split());
                arms[i] = a;
                rewards[i] = env.step(new float[] {a}).reward();
            }
            return new float[][] {arms, rewards};
        }

        void fix(float[][] data) {
            fixedArms = data[0];
            fixedRewards = data[1];
        }

        Var loss(float[][] data) {
            float[] adv = data[1].clone();
            double mean = 0;
            for (float v : adv) {
                mean += v / adv.length;
            }
            for (int i = 0; i < adv.length; i++) {
                adv[i] -= (float) mean;
            }
            Tensor onehot = RlTensors.oneHot(data[0], ARMS, logits.value().device());
            Var logp = RlTensors.logSoftmax(logits.add(Tensor.zeros(Shape.of(pulls, ARMS))));
            return logp.mul(onehot).sum(1, true).mul(Tensor.of(Shape.of(pulls, 1), adv)).mean().neg();
        }

        @Override
        public Var supportLoss() {
            return loss(fixedArms != null ? new float[][] {fixedArms, fixedRewards} : play());
        }

        @Override
        public Var queryLoss() {
            return loss(play());
        }
    }

    private static double maml(Function<SplittableRandom, Bandit> dist, int horizon, int iterations, int evalTasks,
            long seed) {
        Var logits = Var.param(Tensor.zeros(Shape.of(1, ARMS)));
        int half = horizon / 2;
        Maml maml = new Maml(List.of(logits), Optimizer.adam(1e-2f), 2f, 1, false);
        SplittableRandom r = new SplittableRandom(seed);
        for (int it = 0; it < iterations / 4; it++) {
            List<PullTask> batch = new ArrayList<>();
            for (int b = 0; b < 8; b++) {
                batch.add(new PullTask(logits, dist.apply(r), r.split(), half));
            }
            maml.metaStep(batch);
        }
        SplittableRandom er = new SplittableRandom(seed + 100);
        double total = 0;
        for (int i = 0; i < evalTasks; i++) {
            List<Tensor> start = maml.snapshot();
            PullTask task = new PullTask(logits, dist.apply(er), er.split(), half);
            float[][] first = task.play();
            task.fix(first);
            maml.adapt(task, 1);
            PullTask rest = new PullTask(logits, task.env, er.split(), horizon - half);
            float[][] second = rest.play();
            for (float v : first[1]) {
                total += v;
            }
            for (float v : second[1]) {
                total += v;
            }
            maml.restore(start);
        }
        return total / (evalTasks * (double) horizon);
    }
}
