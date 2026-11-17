package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * The Never Give Up intrinsic reward (Badia et al., 2020,
 * <a href="https://arxiv.org/abs/2002.06038">arXiv:2002.06038</a>): an <b>episodic</b> novelty
 * bonus from a k-nearest-neighbour memory of the embeddings seen in the current episode,
 * multiplied by a <b>lifelong</b> novelty modulator from {@link RandomNetworkDistillation}.
 * <pre>
 * K(x, y) = eps / (d(x, y)^2 / mean d_k^2 + eps)
 * r_episodic = 1 / (sqrt(sum over the k nearest K) + c)   (scaled so a new state scores 1)
 * r = r_episodic * clip(1 + (rnd - mean rnd) / std rnd, 1, L)
 * </pre>
 * The episodic part keeps paying for states not yet visited <em>in this episode</em>, so
 * the agent never stops exploring once the lifelong bonus has decayed; the lifelong part
 * prefers states rarely seen across episodes. The memory is cleared at the start of every
 * episode: wrap the environment with {@link #episodic} (or call {@link #newEpisode()}).
 *
 * <p>The paper learns the embedding with an inverse dynamics model; here it is the
 * observation itself unless an {@link #embedding} is given.
 */
public final class NeverGiveUp implements IntrinsicReward {

    private final RandomNetworkDistillation lifelong;
    private final List<float[]> memory = new ArrayList<>();
    private final AgentKit.Running distances = new AgentKit.Running();
    private final AgentKit.Running novelty = new AgentKit.Running();
    private UnaryOperator<float[]> embedding = UnaryOperator.identity();
    private int neighbours = 10;
    private double epsilon = 1e-3;
    private double clusterDistance = 0.008;
    private double pseudoCount = 1e-3;
    private double maxSimilarity = 8;
    private double maxModulation = 5;

    private NeverGiveUp(RandomNetworkDistillation lifelong) {
        this.lifelong = lifelong;
    }

    /**
     * @param lifelong the lifelong novelty, or {@code null} for the episodic bonus alone
     * @return the bonus
     */
    public static NeverGiveUp of(RandomNetworkDistillation lifelong) {
        return new NeverGiveUp(lifelong);
    }

    /** @param f maps an observation to the embedding the memory compares @return this bonus */
    public NeverGiveUp embedding(UnaryOperator<float[]> f) {
        this.embedding = Objects.requireNonNull(f);
        return this;
    }

    /** @param k neighbours in the kernel sum @return this bonus */
    public NeverGiveUp neighbours(int k) {
        this.neighbours = k;
        return this;
    }

    /** @param l the largest lifelong multiplier @return this bonus */
    public NeverGiveUp maxModulation(double l) {
        this.maxModulation = l;
        return this;
    }

    /** Clears the episodic memory. */
    public void newEpisode() {
        memory.clear();
    }

    /** @return the embeddings stored this episode */
    public int memorySize() {
        return memory.size();
    }

    /**
     * Scores an embedding against the episodic memory and adds it.
     *
     * @param e the embedding
     * @return the episodic bonus
     */
    public float episodic(float[] e) {
        double[] d = new double[memory.size()];
        for (int i = 0; i < d.length; i++) {
            float[] m = memory.get(i);
            double s = 0;
            for (int j = 0; j < e.length; j++) {
                s += (e[j] - m[j]) * (e[j] - m[j]);
            }
            d[i] = s;
        }
        memory.add(e.clone());
        double scale = Math.sqrt(epsilon) + pseudoCount;
        if (d.length == 0) {
            return 1;
        }
        Arrays.sort(d);
        int k = Math.min(neighbours, d.length);
        for (int i = 0; i < k; i++) {
            distances.add(d[i]);
        }
        double mean = Math.max(1e-8, distances.mean());
        double sum = 0;
        for (int i = 0; i < k; i++) {
            double dn = Math.max(0, d[i] / mean - clusterDistance);
            sum += epsilon / (dn + epsilon);
        }
        double similarity = Math.sqrt(sum) + pseudoCount;
        return similarity > maxSimilarity ? 0 : (float) Math.min(1, scale / similarity);
    }

    @Override
    public float observe(float[] observation, float[] action, float[] next) {
        float r = episodic(embedding.apply(next));
        if (lifelong == null) {
            return r;
        }
        float err = lifelong.observe(observation, action, next);
        novelty.add(err);
        double alpha = 1 + (err - novelty.mean()) / novelty.std();
        return (float) (r * Math.max(1, Math.min(maxModulation, alpha)));
    }

    /**
     * Wraps an environment so that every reset clears this bonus' episodic memory (the
     * wrapped environment adds no reward; combine with {@link CuriosityEnvironment}).
     *
     * @param inner the environment
     * @return the wrapper
     */
    public Environment episodic(Environment inner) {
        return new Environment() {
            @Override
            public Space observationSpace() {
                return inner.observationSpace();
            }

            @Override
            public Space actionSpace() {
                return inner.actionSpace();
            }

            @Override
            public float[] reset(SplittableRandom random) {
                newEpisode();
                return inner.reset(random);
            }

            @Override
            public Step step(float[] action) {
                return inner.step(action);
            }

            @Override
            public String render() {
                return inner.render();
            }
        };
    }
}
