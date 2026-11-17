package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.Transition;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A fixed set of logged trajectories, the input of offline (batch) reinforcement learning:
 * {@link BehaviorCloning}, {@link Cql}, {@link Iql} and {@link DecisionTransformer} learn
 * from it without ever acting in the environment. Collect one from any behaviour
 * {@link Agent}, or merge several of different quality.
 */
public final class OfflineDataset {

    /**
     * One logged step.
     *
     * @param observation what the agent saw
     * @param action what it did ({@code {index}} for discrete actions)
     * @param reward the reward
     * @param nextObservation what it saw next
     * @param terminated whether the episode ended in a terminal state
     */
    public record Step(float[] observation, float[] action, float reward, float[] nextObservation,
            boolean terminated) {
    }

    private final List<List<Step>> trajectories;

    private OfflineDataset(List<List<Step>> trajectories) {
        this.trajectories = trajectories;
    }

    /**
     * @param trajectories the episodes, each a list of steps
     * @return the dataset
     */
    public static OfflineDataset of(List<List<Step>> trajectories) {
        List<List<Step>> copy = new ArrayList<>();
        for (List<Step> t : trajectories) {
            if (!t.isEmpty()) {
                copy.add(List.copyOf(t));
            }
        }
        return new OfflineDataset(Collections.unmodifiableList(copy));
    }

    /**
     * Logs episodes of a behaviour policy.
     *
     * @param env the environment
     * @param behaviour the policy, asked with {@code explore = true}
     * @param episodes how many episodes
     * @param seed the episode seed
     * @return the dataset
     */
    public static OfflineDataset collect(Environment env, Agent behaviour, int episodes, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<List<Step>> out = new ArrayList<>();
        for (int e = 0; e < episodes; e++) {
            List<Step> t = new ArrayList<>();
            float[] obs = env.reset(random.split());
            Environment.Step s;
            do {
                float[] a = behaviour.act(obs, true);
                s = env.step(a);
                t.add(new Step(obs, a.clone(), s.reward(), s.observation(), s.terminated()));
                obs = s.observation();
            } while (!s.done());
            out.add(t);
        }
        return of(out);
    }

    /**
     * @param other another dataset
     * @return both datasets' trajectories
     */
    public OfflineDataset merge(OfflineDataset other) {
        List<List<Step>> all = new ArrayList<>(trajectories);
        all.addAll(other.trajectories);
        return of(all);
    }

    /** @return the trajectories */
    public List<List<Step>> trajectories() {
        return trajectories;
    }

    /** @return the number of logged steps */
    public int size() {
        return trajectories.stream().mapToInt(List::size).sum();
    }

    /** @return every step, in order */
    public List<Step> steps() {
        List<Step> all = new ArrayList<>();
        trajectories.forEach(all::addAll);
        return all;
    }

    /** @return the mean undiscounted return of the logged episodes */
    public double meanReturn() {
        double sum = 0;
        for (List<Step> t : trajectories) {
            for (Step s : t) {
                sum += s.reward();
            }
        }
        return sum / Math.max(1, trajectories.size());
    }

    /** @return the highest undiscounted return of any logged episode */
    public double maxReturn() {
        double best = Double.NEGATIVE_INFINITY;
        for (List<Step> t : trajectories) {
            best = Math.max(best, t.stream().mapToDouble(Step::reward).sum());
        }
        return best;
    }

    /**
     * @param fraction the share of best episodes to keep, by return
     * @return a dataset of the top episodes (filtered behaviour cloning uses this)
     */
    public OfflineDataset best(double fraction) {
        List<List<Step>> sorted = new ArrayList<>(trajectories);
        sorted.sort((a, b) -> Double.compare(b.stream().mapToDouble(Step::reward).sum(),
                a.stream().mapToDouble(Step::reward).sum()));
        return of(sorted.subList(0, Math.max(1, (int) Math.round(sorted.size() * fraction))));
    }

    /**
     * Copies every step into a replay buffer, for off-policy learners.
     *
     * @param gamma the discount stored with each transition
     * @return the buffer
     */
    public ReplayBuffer toReplayBuffer(float gamma) {
        Step first = trajectories.getFirst().getFirst();
        ReplayBuffer b = new ReplayBuffer(Math.max(1, size()), first.observation().length, first.action().length);
        for (Step s : steps()) {
            b.add(new Transition(s.observation(), s.action(), s.reward(), s.nextObservation(), s.terminated(), gamma));
        }
        return b;
    }
}
