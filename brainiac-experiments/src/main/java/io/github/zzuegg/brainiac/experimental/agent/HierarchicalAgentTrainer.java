package io.github.zzuegg.brainiac.experimental.agent;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.SimulatedUser;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * ArCHer's hierarchical training (Zhou et al., ICML 2024,
 * <a href="https://arxiv.org/abs/2402.19446">arXiv:2402.19446</a>): at the high level an
 * off-policy {@link TurnLevelCritic} learns {@code Q(context, utterance)} by TD learning across
 * turns from a replay buffer; at the low level the token policy is trained within each turn by
 * policy gradient with the critic's advantage {@code Q(s, a) - V(s)}, where {@code V(s)} is the
 * mean Q of utterances the current policy samples in {@code s}. Trajectory-level policy
 * gradients give one reward to a whole conversation; the critic gives every turn its own
 * credit, and reuses old conversations.
 *
 * <p>{@link #play} runs batched conversations against a {@link SimulatedUser} and counts the
 * user's replies as environment interactions, so other trainers can share it.
 *
 * <p>Options from the paper's recipe, all off by default: {@link #criticWarmup} (critic-only
 * steps before the first actor update), {@link #entropyBonus} (an entropy bonus in the token
 * policy gradient), and, on the critic, clipped double Q and the target rate
 * ({@link TurnLevelCritic#doubleQ}, {@link TurnLevelCritic#targetRate}); the actor/critic
 * learning-rate ratio is the ratio of the two optimizers passed to {@link #step}. Every step
 * also reports the critic's over-estimation: its mean {@code Q(s, a)} on the new turns minus
 * their Monte Carlo returns.
 *
 * @param <I> the user's intent type
 */
public final class HierarchicalAgentTrainer<I> {

    /**
     * A played conversation.
     *
     * @param transitions its turns
     * @param outcome the user's score (0 if the turn limit ended it)
     * @param turns assistant turns
     */
    public record Episode(List<TurnLevelCritic.Transition> transitions, float outcome, int turns) {
    }

    /**
     * What a step did.
     *
     * @param meanOutcome the mean outcome of the conversations played
     * @param tdError the mean squared TD error of the critic updates
     * @param overestimation the critic's mean {@code Q(s, a)} on the new turns minus their
     *     discounted Monte Carlo returns
     */
    public record Stats(double meanOutcome, double tdError, double overestimation) {

        /**
         * @param meanOutcome the mean outcome of the conversations played
         * @param tdError the mean squared TD error of the critic updates
         */
        public Stats(double meanOutcome, double tdError) {
            this(meanOutcome, tdError, 0);
        }
    }

    private final WindowLm policy;
    private final SimulatedUser<I> user;
    private final TurnLevelCritic critic;
    private final int maxTokens;
    private final int[] stops;
    private final int turnLimit;
    private final List<TurnLevelCritic.Transition> replay = new ArrayList<>();
    private int replayLimit = 4096;
    private int criticSteps = 8;
    private int criticBatch = 64;
    private int valueSamples = 4;
    private int warmup;
    private double temperature = 1;
    private long interactions;
    private int criticWarmupSteps;
    private float entropy;
    private long steps;

    private HierarchicalAgentTrainer(WindowLm policy, SimulatedUser<I> user, TurnLevelCritic critic, int maxTokens,
            int[] stops, int turnLimit) {
        this.policy = Objects.requireNonNull(policy);
        this.user = Objects.requireNonNull(user);
        this.critic = Objects.requireNonNull(critic);
        this.maxTokens = maxTokens;
        this.stops = stops.clone();
        this.turnLimit = turnLimit;
    }

    /**
     * @param policy the token-level policy (trained in place)
     * @param user the user simulator
     * @param critic the turn-level critic
     * @param maxTokens the most tokens per utterance
     * @param stops tokens that end an utterance
     * @param turnLimit the most assistant turns
     * @param <I> the intent type
     * @return the trainer
     */
    public static <I> HierarchicalAgentTrainer<I> of(WindowLm policy, SimulatedUser<I> user, TurnLevelCritic critic,
            int maxTokens, int[] stops, int turnLimit) {
        return new HierarchicalAgentTrainer<>(policy, user, critic, maxTokens, stops, turnLimit);
    }

    /**
     * @param steps critic minibatch updates per step
     * @param batch transitions per critic update
     * @param samples utterances sampled per state to estimate {@code V(s)}
     * @param bufferLimit the replay buffer's capacity
     * @param warmupTransitions transitions collected before the policy is first updated
     * @param t the sampling temperature
     * @return this trainer
     */
    public HierarchicalAgentTrainer<I> schedule(int steps, int batch, int samples, int bufferLimit,
            int warmupTransitions, double t) {
        this.warmup = warmupTransitions;
        this.criticSteps = steps;
        this.criticBatch = batch;
        this.valueSamples = samples;
        this.replayLimit = bufferLimit;
        this.temperature = t;
        return this;
    }

    /**
     * @param n steps in which only the critic is updated, before any actor update
     * @return this trainer
     */
    public HierarchicalAgentTrainer<I> criticWarmup(int n) {
        this.criticWarmupSteps = n;
        return this;
    }

    /**
     * @param c the weight of an entropy bonus in the token policy gradient (0: off)
     * @return this trainer
     */
    public HierarchicalAgentTrainer<I> entropyBonus(float c) {
        this.entropy = c;
        return this;
    }

    /** @return the user replies so far, over every {@link #play} through this trainer */
    public long interactions() {
        return interactions;
    }

    /**
     * Plays conversations with this trainer's policy.
     *
     * @param intents one conversation per intent
     * @param t the sampling temperature
     * @param random the random source
     * @return the episodes
     */
    public List<Episode> play(List<I> intents, double t, SplittableRandom random) {
        long[] counter = new long[1];
        List<Episode> out = play(policy, user, intents, maxTokens, stops, turnLimit, t, random, counter);
        interactions += counter[0];
        return out;
    }

    /**
     * Plays conversations in lockstep, one batched policy call per turn.
     *
     * @param lm the assistant
     * @param user the user simulator
     * @param intents one conversation per intent
     * @param maxTokens the most tokens per utterance
     * @param stops tokens that end an utterance
     * @param turnLimit the most assistant turns
     * @param t the sampling temperature
     * @param random the random source
     * @param interactions element 0 is increased by the user replies made
     * @param <I> the intent type
     * @return the episodes
     */
    public static <I> List<Episode> play(WindowLm lm, SimulatedUser<I> user, List<I> intents, int maxTokens,
            int[] stops, int turnLimit, double t, SplittableRandom random, long[] interactions) {
        int n = intents.size();
        int[][] context = new int[n][];
        List<List<TurnLevelCritic.Transition>> transitions = new ArrayList<>();
        float[] outcome = new float[n];
        int[] turns = new int[n];
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            context[i] = user.opening(intents.get(i));
            transitions.add(new ArrayList<>());
        }
        for (int turn = 1; turn <= turnLimit; turn++) {
            List<Integer> live = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    live.add(i);
                }
            }
            if (live.isEmpty()) {
                break;
            }
            int[][] ctx = new int[live.size()][];
            for (int j = 0; j < ctx.length; j++) {
                ctx[j] = context[live.get(j)];
            }
            int[][] said = lm.sampleUntil(ctx, maxTokens, stops, t, random);
            for (int j = 0; j < ctx.length; j++) {
                int i = live.get(j);
                int[] before = context[i];
                int[] after = concat(before, said[j]);
                int[] reply = user.reply(intents.get(i), after, said[j], turn);
                interactions[0]++;
                turns[i] = turn;
                if (reply == null) {
                    outcome[i] = user.outcome(intents.get(i), after, said[j]);
                    transitions.get(i).add(new TurnLevelCritic.Transition(before, said[j], outcome[i], null, true));
                    done[i] = true;
                } else {
                    int[] next = concat(after, reply);
                    boolean last = turn == turnLimit;
                    transitions.get(i).add(new TurnLevelCritic.Transition(before, said[j], 0, last ? null : next,
                            last));
                    context[i] = next;
                    done[i] = last;
                }
            }
        }
        List<Episode> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(new Episode(List.copyOf(transitions.get(i)), outcome[i], turns[i]));
        }
        return out;
    }

    static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * One ArCHer step: play a conversation per intent, add its turns to the replay buffer,
     * update the critic by TD on replayed minibatches, then update the policy within the new
     * turns with the critic's advantages.
     *
     * @param intents the intents to play
     * @param policyOptimizer the policy's optimizer
     * @param criticOptimizer the critic's optimizer
     * @param random the random source
     * @return what happened
     */
    public Stats step(List<I> intents, Optimizer policyOptimizer, Optimizer criticOptimizer,
            SplittableRandom random) {
        steps++;
        List<Episode> eps = play(intents, temperature, random);
        List<TurnLevelCritic.Transition> fresh = new ArrayList<>();
        List<Float> returns = new ArrayList<>();
        double outcome = 0;
        for (Episode e : eps) {
            fresh.addAll(e.transitions());
            outcome += e.outcome();
            int n = e.transitions().size();
            for (int t = 0; t < n; t++) {
                returns.add((float) (Math.pow(critic.gamma(), n - 1 - t) * e.outcome()));
            }
        }
        replay.addAll(fresh);
        while (replay.size() > replayLimit) {
            replay.remove(0);
        }
        double td = 0;
        for (int s = 0; s < criticSteps; s++) {
            List<TurnLevelCritic.Transition> batch = new ArrayList<>();
            for (int k = 0; k < criticBatch; k++) {
                batch.add(replay.get(random.nextInt(replay.size())));
            }
            td += critic.update(batch, sampleNext(batch, random), criticOptimizer) / criticSteps;
        }
        // actor: advantage Q(s, a) - mean Q(s, a') over sampled a'
        List<int[]> cs = new ArrayList<>();
        List<int[]> us = new ArrayList<>();
        for (TurnLevelCritic.Transition t : fresh) {
            cs.add(t.context());
            us.add(t.utterance());
        }
        float[] q = critic.q(cs, us);
        double over = 0;
        for (int i = 0; i < q.length; i++) {
            over += (q[i] - returns.get(i)) / Math.max(1, q.length);
        }
        int[][] rep = new int[fresh.size() * valueSamples][];
        for (int i = 0; i < rep.length; i++) {
            rep[i] = cs.get(i / valueSamples);
        }
        int[][] alt = policy.sampleUntil(rep, maxTokens, stops, temperature, random);
        List<int[]> repList = Arrays.asList(rep);
        float[] qa = critic.q(repList, Arrays.asList(alt));
        float[] raw = new float[fresh.size()];
        for (int i = 0; i < raw.length; i++) {
            double v = 0;
            for (int k = 0; k < valueSamples; k++) {
                v += qa[i * valueSamples + k] / valueSamples;
            }
            raw[i] = (float) (q[i] - v);
        }
        // the advantages are scaled to unit spread; nothing moves before the critic has data
        double sq = 0;
        for (float a : raw) {
            sq += a * a;
        }
        double scale = 1 / Math.sqrt(sq / Math.max(1, raw.length) + 1e-8);
        List<Float> adv = new ArrayList<>();
        for (float a : raw) {
            adv.add((float) (a * scale));
        }
        if (replay.size() >= warmup && steps > criticWarmupSteps) {
            if (entropy > 0) {
                entropyPolicyGradient(cs, us, adv, policyOptimizer);
            } else {
                ToolInterleavedRollout.policyGradient(policy, cs, us, adv, policyOptimizer);
            }
        }
        return new Stats(outcome / eps.size(), td, over);
    }

    /** The policy gradient of {@link ToolInterleavedRollout#policyGradient} minus an entropy bonus. */
    private void entropyPolicyGradient(List<int[]> contexts, List<int[]> chunks, List<Float> advantages,
            Optimizer optimizer) {
        List<int[]> prefixes = new ArrayList<>();
        List<Float> w = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            for (int t = 0; t < chunks.get(i).length; t++) {
                int[] prefix = Arrays.copyOf(contexts.get(i), contexts.get(i).length + t);
                System.arraycopy(chunks.get(i), 0, prefix, contexts.get(i).length, t);
                prefixes.add(prefix);
                w.add(advantages.get(i));
            }
        }
        if (prefixes.isEmpty()) {
            return;
        }
        float[] wa = new float[w.size()];
        for (int i = 0; i < wa.length; i++) {
            wa[i] = w.get(i);
        }
        Var pg = policy.logProbs(contexts, chunks).mul(Var.of(Tensor.of(Shape.of(wa.length, 1), wa))).mean().neg();
        Var logits = policy.logits(prefixes);
        Var logSoft = logits.sub(logits.logsumexp(1, true));
        Var h = logSoft.exp().mul(logSoft).sum(1, false).neg().mean();
        pg.sub(h.mul(entropy)).backward();
        optimizer.step(policy.parameters());
    }

    private List<int[]> sampleNext(List<TurnLevelCritic.Transition> batch, SplittableRandom random) {
        List<Integer> idx = new ArrayList<>();
        List<int[]> ctx = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            if (!batch.get(i).done()) {
                idx.add(i);
                ctx.add(batch.get(i).next());
            }
        }
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            out.add(null);
        }
        if (!ctx.isEmpty()) {
            int[][] said = policy.sampleUntil(ctx.toArray(int[][]::new), maxTokens, stops, temperature, random);
            for (int j = 0; j < said.length; j++) {
                out.set(idx.get(j), said[j]);
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "HierarchicalAgentTrainer(" + critic + ")";
    }
}
