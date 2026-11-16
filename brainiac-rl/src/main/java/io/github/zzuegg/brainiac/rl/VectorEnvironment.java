package io.github.zzuegg.brainiac.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Several agents' worlds stepped together: either independent copies of an environment, or
 * one shared world with several actors in it, such as cars on the same race track.
 *
 * <p>Every entry sees its own observation, picks its own action and gets its own reward.
 * All actions of a step are applied at once, so in a shared world the actors move at the
 * same time. When an entry's episode ends, the learner calls {@link #reset(int,
 * SplittableRandom)} for that entry alone, while the others carry on.
 *
 * <p>{@link Ppo#fit(VectorEnvironment, long, RlListener)} trains one policy on all
 * entries, which for a shared world is self-play with shared parameters.
 */
public interface VectorEnvironment {

    /** @return how many entries step together */
    int count();

    /** @return the shape and bounds of each entry's observation */
    Space observationSpace();

    /** @return each entry's valid actions */
    Space actionSpace();

    /**
     * Starts every entry afresh.
     *
     * @param random the source of randomness
     * @return the first observation of every entry
     */
    float[][] reset(SplittableRandom random);

    /**
     * Starts a new episode for one entry whose episode just ended.
     *
     * @param index the entry
     * @param random the source of randomness
     * @return the entry's first observation
     */
    float[] reset(int index, SplittableRandom random);

    /**
     * Applies one action per entry, all at once.
     *
     * @param actions one action per entry
     * @return what happened to each entry
     */
    Environment.Step[] step(float[][] actions);

    /**
     * Wraps independent environments.
     *
     * @param envs the environments, with the same spaces
     * @return a vector stepping each of them
     */
    static VectorEnvironment of(List<? extends Environment> envs) {
        if (envs.isEmpty()) {
            throw new IllegalArgumentException("at least one environment is needed");
        }
        List<Environment> list = List.copyOf(envs);
        return new VectorEnvironment() {
            @Override
            public int count() {
                return list.size();
            }

            @Override
            public Space observationSpace() {
                return list.getFirst().observationSpace();
            }

            @Override
            public Space actionSpace() {
                return list.getFirst().actionSpace();
            }

            @Override
            public float[][] reset(SplittableRandom random) {
                float[][] obs = new float[list.size()][];
                for (int i = 0; i < obs.length; i++) {
                    obs[i] = list.get(i).reset(random.split());
                }
                return obs;
            }

            @Override
            public float[] reset(int index, SplittableRandom random) {
                return list.get(index).reset(random);
            }

            @Override
            public Environment.Step[] step(float[][] actions) {
                Environment.Step[] steps = new Environment.Step[list.size()];
                for (int i = 0; i < steps.length; i++) {
                    steps[i] = list.get(i).step(actions[i]);
                }
                return steps;
            }
        };
    }

    /**
     * Joins several vectors with the same spaces into one, for example several races run
     * side by side to gather more experience per update.
     *
     * @param parts the vectors
     * @return a vector whose entries are those of every part, in order
     */
    static VectorEnvironment concat(List<? extends VectorEnvironment> parts) {
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("at least one vector is needed");
        }
        List<VectorEnvironment> list = List.copyOf(parts);
        int[] start = new int[list.size() + 1];
        for (int p = 0; p < list.size(); p++) {
            start[p + 1] = start[p] + list.get(p).count();
        }
        return new VectorEnvironment() {
            @Override
            public int count() {
                return start[list.size()];
            }

            @Override
            public Space observationSpace() {
                return list.getFirst().observationSpace();
            }

            @Override
            public Space actionSpace() {
                return list.getFirst().actionSpace();
            }

            @Override
            public float[][] reset(SplittableRandom random) {
                List<float[]> all = new ArrayList<>();
                for (VectorEnvironment v : list) {
                    all.addAll(List.of(v.reset(random.split())));
                }
                return all.toArray(float[][]::new);
            }

            @Override
            public float[] reset(int index, SplittableRandom random) {
                int p = part(index);
                return list.get(p).reset(index - start[p], random);
            }

            @Override
            public Environment.Step[] step(float[][] actions) {
                Environment.Step[] out = new Environment.Step[count()];
                for (int p = 0; p < list.size(); p++) {
                    float[][] a = java.util.Arrays.copyOfRange(actions, start[p], start[p + 1]);
                    System.arraycopy(list.get(p).step(a), 0, out, start[p], a.length);
                }
                return out;
            }

            private int part(int index) {
                for (int p = 0; p < list.size(); p++) {
                    if (index < start[p + 1]) {
                        return p;
                    }
                }
                throw new IndexOutOfBoundsException(index);
            }
        };
    }
}
