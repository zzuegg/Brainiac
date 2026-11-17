package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Runs a {@link ChunkPolicy} with a simulated inference delay, the way a real robot has to:
 * the next chunk is requested {@code delay} steps before it is needed, from the observation
 * at request time, and the current chunk keeps running meanwhile. Three ways to switch:
 *
 * <ul>
 *   <li>{@link Mode#NAIVE}: jump to the new chunk at the step it arrives, dropping its first
 *       {@code delay} (already past) actions. Consecutive chunks may disagree, even pick a
 *       different mode, so actions jump at the boundary.</li>
 *   <li>{@link Mode#TEMPORAL_ENSEMBLE}: ACT's temporal ensembling (Zhao et al.,
 *       <a href="https://arxiv.org/abs/2304.13705">2304.13705</a>): average every chunk that
 *       covers the current step, older ones weighted {@code exp(-m i)} higher. Smooth, but
 *       an average of two modes can be neither.</li>
 *   <li>{@link Mode#INPAINTING}: real-time chunking (Black et al.,
 *       <a href="https://arxiv.org/abs/2506.07339">2506.07339</a>): hand the policy the
 *       actions still to come from the current chunk so it generates a continuation
 *       ({@link ChunkPolicy#chunk(float[], float[][], int, SplittableRandom)}).</li>
 * </ul>
 *
 * <p>It also measures the size of the action change at chunk boundaries, the jerk that
 * naive switching causes.
 */
public final class RealTimeChunking implements Agent {

    /** How to move from one chunk to the next. */
    public enum Mode {
        /** Switch abruptly. */
        NAIVE,
        /** Average overlapping chunks. */
        TEMPORAL_ENSEMBLE,
        /** Inpaint the new chunk to continue the committed actions. */
        INPAINTING
    }

    private record Timed(int start, float[][] actions) {
    }

    private final ChunkPolicy policy;
    private final Mode mode;
    private final int delay;
    private final int period;
    private final SplittableRandom random;
    private float ensembleDecay = 0.1f;
    private final List<Timed> active = new ArrayList<>();
    private Timed pending;
    private int pendingAt;
    private int nextRequest;
    private int step;
    private float[] last;
    private double boundarySum;
    private long boundaries;
    private double changeSum;
    private long changes;

    private RealTimeChunking(ChunkPolicy policy, Mode mode, int delay, int period, long seed) {
        if (delay < 0 || period < 1 || period < delay || delay + period > policy.horizon()) {
            throw new IllegalArgumentException("need 0 <= delay <= period and delay + period <= horizon "
                    + policy.horizon());
        }
        this.policy = policy;
        this.mode = mode;
        this.delay = delay;
        this.period = period;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates an executor.
     *
     * @param policy the chunk policy
     * @param mode how to switch chunks
     * @param delay the inference delay in control steps
     * @param period steps between chunk switches; {@code delay + period} must fit the horizon
     * @param seed the seed for sampling
     * @return the executor
     */
    public static RealTimeChunking of(ChunkPolicy policy, Mode mode, int delay, int period, long seed) {
        return new RealTimeChunking(policy, mode, delay, period, seed);
    }

    /**
     * Sets the temporal-ensembling decay m.
     *
     * @param m the decay
     * @return this executor
     */
    public RealTimeChunking ensembleDecay(float m) {
        this.ensembleDecay = m;
        return this;
    }

    @Override
    public void startEpisode() {
        active.clear();
        pending = null;
        step = 0;
        last = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (step == 0 || active.isEmpty()) {
            active.clear();
            active.add(new Timed(step, policy.chunk(observation, random)));
            nextRequest = step + period - delay;
            pending = null;
        }
        if (step == nextRequest) {
            float[][] chunk;
            if (mode == Mode.INPAINTING) {
                Timed current = active.getLast();
                int from = step - current.start();
                float[][] previous = new float[Math.max(0, current.actions().length - from)][];
                for (int i = 0; i < previous.length; i++) {
                    previous[i] = current.actions()[from + i];
                }
                chunk = policy.chunk(observation, previous, delay, random);
            } else {
                chunk = policy.chunk(observation, random);
            }
            pending = new Timed(step, chunk);
            pendingAt = step + delay;
            nextRequest += period;
        }
        boolean boundary = false;
        if (pending != null && step == pendingAt) {
            if (mode != Mode.TEMPORAL_ENSEMBLE) {
                active.clear();
            }
            active.add(pending);
            pending = null;
            boundary = true;
        }
        active.removeIf(c -> step - c.start() >= c.actions().length);
        float[] action = new float[policy.actionSize()];
        double total = 0;
        for (int i = 0; i < active.size(); i++) {
            Timed c = active.get(i);
            double w = mode == Mode.TEMPORAL_ENSEMBLE ? Math.exp(-ensembleDecay * i) : (i == active.size() - 1 ? 1 : 0);
            float[] a = c.actions()[step - c.start()];
            for (int k = 0; k < action.length; k++) {
                action[k] += (float) (w * a[k]);
            }
            total += w;
        }
        for (int k = 0; k < action.length; k++) {
            action[k] /= (float) total;
        }
        if (last != null) {
            double d = ActionChunking.distance(action, last);
            changeSum += d;
            changes++;
            if (boundary) {
                boundarySum += d;
                boundaries++;
            }
        }
        last = action.clone();
        step++;
        return action;
    }

    /** @return the mean action change at chunk switches over everything run so far */
    public double meanBoundaryChange() {
        return boundaries == 0 ? 0 : boundarySum / boundaries;
    }

    /** @return the mean action change between any two consecutive steps */
    public double meanChange() {
        return changes == 0 ? 0 : changeSum / changes;
    }

    /** @return the switching mode */
    public Mode mode() {
        return mode;
    }
}
