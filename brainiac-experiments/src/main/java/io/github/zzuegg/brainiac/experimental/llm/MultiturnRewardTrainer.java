package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Multiturn-aware rewards (CollabLLM, Wu et al., "CollabLLM: from passive responders to active
 * collaborators", <a href="https://arxiv.org/abs/2502.00640">2502.00640</a>, ICML 2025
 * outstanding paper). Single-turn RLHF rewards each response for its immediate quality, so a
 * model learns to answer at once even when the request is underspecified. CollabLLM instead
 * rewards a response by the <em>future</em> of the conversation it leads to: from the context
 * plus the response, a {@link SimulatedUser} and the current policy play the conversation
 * forward several times, and the multiturn-aware reward is the mean of
 * {@code outcome - lambda * assistantTurns} over those rollouts. A clarifying question earns no
 * immediate credit but a high multiturn reward when it leads to a correct answer, so the
 * policy learns when to ask.
 *
 * <p>Each step plays one conversation per intent with the policy; at every assistant turn it
 * samples a group of candidate responses, scores each with the chosen reward, and hands the
 * groups to a {@link GrpoTrainer} ({@link GrpoTrainer#update}). {@link Mode#SINGLE_TURN} is the
 * baseline: a response is worth the outcome only if the conversation ends with it.
 *
 * @param <I> the intent type of the user simulator
 */
public final class MultiturnRewardTrainer<I> {

    /** How a response is rewarded. */
    public enum Mode {
        /** CollabLLM: the mean outcome of simulated future turns, minus a per-turn cost. */
        MULTITURN,
        /** The baseline: the outcome if the conversation ends here, else 0. */
        SINGLE_TURN
    }

    /**
     * What one step did.
     *
     * @param meanReward the mean reward of the candidate responses
     * @param groups the groups trained on (one per assistant turn played)
     * @param grpo the GRPO statistics
     */
    public record Stats(double meanReward, int groups, GrpoTrainer.Stats grpo) {
    }

    /**
     * A finished conversation.
     *
     * @param context every message and response, concatenated
     * @param turns the assistant responses
     * @param outcome the task score ({@code 0} if it never finished)
     * @param finished whether the user ended it within the turn limit
     */
    public record Conversation(int[] context, int turns, float outcome, boolean finished) {
    }

    private final WindowLm policy;
    private final GrpoTrainer trainer;
    private final SimulatedUser<I> user;
    private Mode mode = Mode.MULTITURN;
    private int groupSize = 8;
    private int maxTokens = 3;
    private int[] stops = new int[0];
    private double temperature = 1;
    private int rollouts = 4;
    private int maxTurns = 3;
    private float turnPenalty = 0.1f;

    private MultiturnRewardTrainer(WindowLm policy, GrpoTrainer trainer, SimulatedUser<I> user) {
        this.policy = Objects.requireNonNull(policy);
        this.trainer = Objects.requireNonNull(trainer);
        this.user = Objects.requireNonNull(user);
    }

    /**
     * @param policy the assistant (the model {@code trainer} trains)
     * @param trainer the GRPO trainer of {@code policy}
     * @param user the user simulator
     * @param <I> the intent type
     * @return the trainer
     */
    public static <I> MultiturnRewardTrainer<I> of(WindowLm policy, GrpoTrainer trainer, SimulatedUser<I> user) {
        return new MultiturnRewardTrainer<>(policy, trainer, user);
    }

    /**
     * @param n candidate responses per turn
     * @param maxNew the most tokens per response
     * @param stopTokens tokens that end a response (included in it)
     * @param t the sampling temperature
     * @return this trainer
     */
    public MultiturnRewardTrainer<I> sampling(int n, int maxNew, int[] stopTokens, double t) {
        this.groupSize = n;
        this.maxTokens = maxNew;
        this.stops = stopTokens.clone();
        this.temperature = t;
        return this;
    }

    /**
     * @param m the reward
     * @param futureRollouts simulated futures per candidate response (multiturn only)
     * @param turnLimit the most assistant turns in a conversation
     * @param lambda the cost per assistant turn
     * @return this trainer
     */
    public MultiturnRewardTrainer<I> reward(Mode m, int futureRollouts, int turnLimit, float lambda) {
        this.mode = Objects.requireNonNull(m);
        this.rollouts = futureRollouts;
        this.maxTurns = turnLimit;
        this.turnPenalty = lambda;
        return this;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** A conversation being simulated. */
    private final class Sim {
        final I intent;
        int[] context;
        int[] last;
        int turns;
        float score;
        boolean done;

        Sim(I intent, int[] context, int[] last, int turns) {
            this.intent = intent;
            this.context = context;
            this.last = last;
            this.turns = turns;
        }
    }

    /** Plays every conversation to its end (or the turn limit), batching the policy's responses. */
    private void finish(List<Sim> sims, double t, SplittableRandom random) {
        List<Sim> live = new ArrayList<>(sims);
        while (!live.isEmpty()) {
            List<Sim> next = new ArrayList<>();
            for (Sim s : live) {
                int[] reply = user.reply(s.intent, s.context, s.last, s.turns);
                if (reply == null) {
                    s.score = user.outcome(s.intent, s.context, s.last);
                    s.done = true;
                } else if (s.turns >= maxTurns) {
                    s.score = 0;   // ran out of turns without finishing
                } else {
                    s.context = concat(s.context, reply);
                    next.add(s);
                }
            }
            if (next.isEmpty()) {
                break;
            }
            int[][] ctx = new int[next.size()][];
            for (int i = 0; i < ctx.length; i++) {
                ctx[i] = next.get(i).context;
            }
            int[][] out = policy.sampleUntil(ctx, maxTokens, stops, t, random);
            for (int i = 0; i < out.length; i++) {
                Sim s = next.get(i);
                s.last = out[i];
                s.context = concat(s.context, out[i]);
                s.turns++;
            }
            live = next;
        }
    }

    /**
     * Plays one whole conversation with the policy.
     *
     * @param intent the user's intent
     * @param t the sampling temperature (0 for greedy)
     * @param random the random source
     * @return the conversation
     */
    public Conversation converse(I intent, double t, SplittableRandom random) {
        return converse(policy, user, intent, maxTokens, stops, maxTurns, t, random);
    }

    /**
     * Plays one whole conversation between any model and user, for evaluation.
     *
     * @param model the assistant
     * @param user the user simulator
     * @param intent the user's intent
     * @param maxNew the most tokens per response
     * @param stops tokens that end a response
     * @param turnLimit the most assistant turns
     * @param t the sampling temperature (0 for greedy)
     * @param random the random source
     * @param <I> the intent type
     * @return the conversation
     */
    public static <I> Conversation converse(WindowLm model, SimulatedUser<I> user, I intent, int maxNew, int[] stops,
            int turnLimit, double t, SplittableRandom random) {
        int[] context = user.opening(intent);
        for (int turn = 1; turn <= turnLimit; turn++) {
            int[] response = model.sampleUntil(new int[][] {context}, maxNew, stops, t, random)[0];
            context = concat(context, response);
            int[] reply = user.reply(intent, context, response, turn);
            if (reply == null) {
                return new Conversation(context, turn, user.outcome(intent, context, response), true);
            }
            context = concat(context, reply);
        }
        return new Conversation(context, turnLimit, 0, false);
    }

    /**
     * One training step: a conversation per intent, a scored group per assistant turn, and one
     * GRPO update on all groups.
     *
     * @param intents the users' intents for this step
     * @param random the random source
     * @return what happened
     */
    public Stats step(List<I> intents, SplittableRandom random) {
        List<int[]> groupPrompts = new ArrayList<>();
        List<int[][]> groups = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        double sum = 0;
        int count = 0;
        for (I intent : intents) {
            int[] context = user.opening(intent);
            for (int turn = 1; turn <= maxTurns; turn++) {
                int[][] ctx = new int[groupSize][];
                Arrays.fill(ctx, context);
                int[][] responses = policy.sampleUntil(ctx, maxTokens, stops, temperature, random);
                float[] r = new float[groupSize];
                if (mode == Mode.SINGLE_TURN) {
                    for (int g = 0; g < groupSize; g++) {
                        int[] c = concat(context, responses[g]);
                        r[g] = user.reply(intent, c, responses[g], turn) == null
                                ? user.outcome(intent, c, responses[g]) : 0;
                    }
                } else {
                    List<Sim> sims = new ArrayList<>();
                    for (int g = 0; g < groupSize; g++) {
                        for (int k = 0; k < rollouts; k++) {
                            sims.add(new Sim(intent, concat(context, responses[g]), responses[g], turn));
                        }
                    }
                    finish(sims, temperature, random);
                    for (int g = 0; g < groupSize; g++) {
                        double s = 0;
                        for (int k = 0; k < rollouts; k++) {
                            Sim sim = sims.get(g * rollouts + k);
                            s += sim.score - turnPenalty * sim.turns;
                        }
                        r[g] = (float) (s / rollouts);
                    }
                }
                for (float x : r) {
                    sum += x;
                    count++;
                }
                groupPrompts.add(context);
                groups.add(responses);
                rewards.add(r);
                // continue the conversation on-policy with the first candidate
                int[] chosen = responses[0];
                int[] c = concat(context, chosen);
                int[] reply = user.reply(intent, c, chosen, turn);
                if (reply == null) {
                    break;
                }
                context = concat(c, reply);
            }
        }
        GrpoTrainer.Stats s = trainer.update(groupPrompts, groups, rewards);
        return new Stats(count == 0 ? 0 : sum / count, groups.size(), s);
    }
}
