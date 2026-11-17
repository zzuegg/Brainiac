package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Action-chunk helpers shared by the chunked robot policies ({@link DiffusionPolicy},
 * {@link FlowActionPolicy}, {@link FastActionTokenizer}-based policies): cutting
 * demonstrations into (observation, next {@code H} actions) pairs, and the plain
 * receding-horizon executor that runs the first few actions of each chunk and then asks
 * again (Diffusion Policy, Chi et al., <a href="https://arxiv.org/abs/2303.04137">2303.04137</a>).
 */
public final class ActionChunking {
    private ActionChunking() {
    }

    /**
     * Training pairs.
     *
     * @param observations one observation per pair
     * @param chunks the next {@code H} actions after it, flattened row-major ({@code H * D})
     */
    public record Pairs(float[][] observations, float[][] chunks) {
        /** @return the number of pairs */
        public int size() {
            return observations.length;
        }
    }

    /**
     * Cuts every trajectory into one pair per step; chunks running past the end repeat the
     * last action.
     *
     * @param data the demonstrations
     * @param horizon the chunk length H
     * @return the pairs
     */
    public static Pairs pairs(OfflineDataset data, int horizon) {
        List<float[]> obs = new ArrayList<>();
        List<float[]> chunks = new ArrayList<>();
        for (List<OfflineDataset.Step> t : data.trajectories()) {
            int d = t.getFirst().action().length;
            for (int i = 0; i < t.size(); i++) {
                float[] c = new float[horizon * d];
                for (int k = 0; k < horizon; k++) {
                    System.arraycopy(t.get(Math.min(i + k, t.size() - 1)).action(), 0, c, k * d, d);
                }
                obs.add(t.get(i).observation());
                chunks.add(c);
            }
        }
        return new Pairs(obs.toArray(float[][]::new), chunks.toArray(float[][]::new));
    }

    /**
     * Reshapes a flat chunk.
     *
     * @param flat {@code H * D} values
     * @param actionSize D
     * @return {@code [H][D]}
     */
    public static float[][] unflatten(float[] flat, int actionSize) {
        float[][] out = new float[flat.length / actionSize][actionSize];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(flat, i * actionSize, out[i], 0, actionSize);
        }
        return out;
    }

    /**
     * Flattens a chunk.
     *
     * @param chunk {@code [H][D]}
     * @return {@code H * D} values
     */
    public static float[] flatten(float[][] chunk) {
        int d = chunk[0].length;
        float[] out = new float[chunk.length * d];
        for (int i = 0; i < chunk.length; i++) {
            System.arraycopy(chunk[i], 0, out, i * d, d);
        }
        return out;
    }

    /**
     * Receding-horizon control: predict a chunk, execute its first {@code execute} actions
     * open loop, then predict again from the new observation.
     *
     * @param policy the chunk policy
     * @param execute how many actions of each chunk to run, 1 to {@code horizon}
     * @param seed the seed for sampling
     * @return an agent
     */
    public static Agent recedingHorizon(ChunkPolicy policy, int execute, long seed) {
        if (execute < 1 || execute > policy.horizon()) {
            throw new IllegalArgumentException("execute must be in 1.." + policy.horizon());
        }
        SplittableRandom random = new SplittableRandom(seed);
        return new Agent() {
            private float[][] chunk;
            private int next;

            @Override
            public void startEpisode() {
                chunk = null;
            }

            @Override
            public float[] act(float[] observation, boolean explore) {
                if (chunk == null || next >= execute) {
                    chunk = policy.chunk(observation, random);
                    next = 0;
                }
                return chunk[next++].clone();
            }
        };
    }

    /**
     * The mean size of the action change between consecutive steps, a jerk measure.
     *
     * @param actions executed actions in order
     * @return the mean Euclidean norm of {@code a[t] - a[t-1]}
     */
    public static double meanChange(List<float[]> actions) {
        double sum = 0;
        for (int i = 1; i < actions.size(); i++) {
            sum += distance(actions.get(i), actions.get(i - 1));
        }
        return actions.size() < 2 ? 0 : sum / (actions.size() - 1);
    }

    static double distance(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    static float[] clip(float[] a) {
        float[] out = a.clone();
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(-1, Math.min(1, out[i]));
        }
        return out;
    }
}
