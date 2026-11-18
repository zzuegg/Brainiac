package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Tabular Q-learning for reward machines (Toro Icarte et al., 2022,
 * <a href="https://arxiv.org/abs/2010.03950">arXiv:2010.03950</a>): one Q-table per machine
 * state, trained on the {@link RewardMachine.Product product environment}. With
 * {@link #counterfactual(boolean) counterfactual experience} (CRM, the default) every real
 * transition {@code (s, a, s')} is replayed for <em>every</em> machine state {@code u}: the
 * machine says which state {@code u'} and reward the transition would have produced from
 * {@code u}, and {@code Q_u} is updated towards {@code r + gamma max Q_u'(s')}. A single step
 * then teaches the whole task, which makes learning far more sample efficient than
 * Q-learning on the product states alone (counterfactual off).
 */
public final class RewardMachineLearner implements Learner {

    private final RewardMachine machine;
    private final int actions;
    private final List<Map<String, double[]>> tables = new ArrayList<>();
    private boolean counterfactual = true;
    private double learningRate = 0.5;
    private double gamma = 0.95;
    private double epsilon = 0.1;
    private SplittableRandom random = new SplittableRandom(0);

    private RewardMachineLearner(RewardMachine machine, int actions) {
        this.machine = machine;
        this.actions = actions;
        for (int u = 0; u < machine.states(); u++) {
            tables.add(new HashMap<>());
        }
    }

    /**
     * @param machine the reward machine
     * @param actions the number of discrete actions
     * @return the learner, with counterfactual experience
     */
    public static RewardMachineLearner of(RewardMachine machine, int actions) {
        return new RewardMachineLearner(machine, actions);
    }

    /** @param on replay every transition for every machine state (CRM) @return this learner */
    public RewardMachineLearner counterfactual(boolean on) {
        this.counterfactual = on;
        return this;
    }

    /** @param eps the exploration rate @return this learner */
    public RewardMachineLearner epsilon(double eps) {
        this.epsilon = eps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public RewardMachineLearner seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private double[] q(int u, float[] raw) {
        return tables.get(u).computeIfAbsent(Arrays.toString(raw), k -> new double[actions]);
    }

    private int greedy(double[] v) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < v.length; a++) {
            if (v[a] > v[best]) {
                best = a;
                ties = 1;
            } else if (v[a] == v[best] && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    /**
     * Acts on an observation of a product environment that exposes the machine state (the
     * one-hot machine state is the observation's tail).
     */
    @Override
    public float[] act(float[] observation, boolean explore) {
        int n = observation.length - machine.states();
        int u = 0;
        for (int i = 0; i < machine.states(); i++) {
            if (observation[n + i] > 0.5f) {
                u = i;
            }
        }
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {greedy(q(u, Arrays.copyOf(observation, n)))};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env instanceof RewardMachine.Product product) || product.machine() != machine) {
            throw new IllegalArgumentException("needs this machine's product environment");
        }
        Tracker tracker = new Tracker(listener);
        product.reset(random.split());
        while (tracker.steps() < steps) {
            int u = product.machineState();
            float[] raw = product.rawObservation();
            int a = random.nextDouble() < epsilon ? random.nextInt(actions) : greedy(q(u, raw));
            Environment.Step st = product.step(new float[] {a});
            float[] next = product.rawObservation();
            String label = product.label();
            for (int v = 0; v < machine.states(); v++) {
                if (machine.isTerminal(v) || (!counterfactual && v != u)) {
                    continue;
                }
                int v2 = machine.next(v, label);
                double target = machine.reward(v, label);
                if (!machine.isTerminal(v2)) {
                    target += gamma * Arrays.stream(q(v2, next)).max().orElse(0);
                }
                double[] qv = q(v, raw);
                qv[a] += learningRate * (target - qv[a]);
            }
            if (tracker.step(st)) {
                product.reset(random.split());
            }
        }
        return tracker.result();
    }

    @Override
    public String toString() {
        return counterfactual ? "CRM" : "Q-learning on the product";
    }
}
