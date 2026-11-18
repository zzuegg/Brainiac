package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Unsupervised skill discovery as an environment wrapper, so any learner (for example
 * {@code ModernSac}) can pretrain skills without a task reward. At every reset a skill
 * {@code z} is drawn from the prior and appended to the observation; the environment's reward
 * is replaced by an intrinsic one computed from a representation that is trained online on the
 * wrapper's own replay of {@code (s, s', z)}:
 *
 * <ul>
 *   <li>{@link Method#DIAYN} (Eysenbach et al., ICLR 2019,
 *       <a href="https://arxiv.org/abs/1802.06070">arXiv:1802.06070</a>): {@code K} discrete
 *       skills, a discriminator {@code q(z | s')} trained by cross-entropy, reward
 *       {@code log q(z | s') - log p(z)}: skills become distinguishable, not necessarily
 *       far apart;</li>
 *   <li>{@link Method#METRA} (Park et al., ICLR 2024,
 *       <a href="https://arxiv.org/abs/2310.08887">arXiv:2310.08887</a>): unit-vector skills, a
 *       representation {@code phi} trained to maximise {@code (phi(s') - phi(s))^T z} under the
 *       temporal-distance constraint {@code ||phi(s) - phi(s')|| <= 1} with a dual variable,
 *       reward {@code (phi(s') - phi(s))^T z}, which favours covering distant states;</li>
 *   <li>{@link Method#CSF} (Zheng et al., ICLR 2025,
 *       <a href="https://arxiv.org/abs/2412.08021">arXiv:2412.08021</a>): the same reward but
 *       {@code phi} trained with a contrastive bound, {@code -(phi(s') - phi(s))^T z} plus the
 *       log-sum-exp over the batch's other skills as negatives, so no constraint or dual
 *       variable is needed. (The paper's vector-valued successor-feature critic is replaced by
 *       the learner's scalar critic on this reward.)</li>
 *   <li>{@link Method#RANDOM_SKILLS}: skills appended but no intrinsic reward, the baseline.</li>
 * </ul>
 *
 * <p>The representation sees only a projection of the observation ({@code state}, for example
 * the position), as in the papers' x-y prior experiments.
 */
public final class SkillDiscovery implements Environment {

    /** The objective. */
    public enum Method {
        /** mutual information with a skill discriminator */
        DIAYN,
        /** Wasserstein dependency with a temporal-distance constraint */
        METRA,
        /** contrastive successor features */
        CSF,
        /** no intrinsic reward */
        RANDOM_SKILLS
    }

    private record Sample(float[] s, float[] next, float[] z, int index) {
    }

    private final Method method;
    private final Environment inner;
    private final UnaryOperator<float[]> state;
    private final int stateDims;
    private final int skillDims;
    private final Model net;
    private final Optimizer optimizer;
    private final SplittableRandom random;
    private final List<Sample> replay = new ArrayList<>();
    private float[] skill;
    private int skillIndex;
    private float[] fixed;
    private int fixedIndex = -1;
    private float[] lastState;
    private double lambda = 30;
    private long steps;
    private boolean evaluating;

    private SkillDiscovery(Method method, Environment inner, UnaryOperator<float[]> state, int stateDims, int skills,
            long seed) {
        this.method = method;
        this.inner = inner;
        this.state = state;
        this.stateDims = stateDims;
        this.skillDims = skills;
        this.random = new SplittableRandom(seed);
        this.net = Nets.mlp(stateDims, skills, Activation.RELU, seed, 64, 64);
        this.optimizer = Optimizer.adam(1e-3f);
    }

    /**
     * @param method the objective
     * @param inner the environment
     * @param state projects an observation onto the state the representation sees
     * @param stateDims values of that projection
     * @param skills the number of discrete skills (DIAYN) or the skill dimension (METRA, CSF)
     * @param seed the seed
     * @return the wrapper
     */
    public static SkillDiscovery of(Method method, Environment inner, UnaryOperator<float[]> state, int stateDims,
            int skills, long seed) {
        return new SkillDiscovery(Objects.requireNonNull(method), inner, state, stateDims, skills, seed);
    }

    /** @return values per skill vector */
    public int skillDims() {
        return skillDims;
    }

    /** @return the objective */
    public Method method() {
        return method;
    }

    /**
     * @param k the evaluation index
     * @param count how many evaluation skills
     * @return the {@code k}-th evaluation skill: one-hot for DIAYN, evenly spaced unit vectors in
     *     the first two dimensions otherwise
     */
    public float[] evaluationSkill(int k, int count) {
        float[] z = new float[skillDims];
        if (method == Method.DIAYN) {
            z[k % skillDims] = 1;
            return z;
        }
        double angle = 2 * Math.PI * k / count;
        z[0] = (float) Math.cos(angle);
        if (skillDims > 1) {
            z[1] = (float) Math.sin(angle);
        }
        return z;
    }

    /**
     * Fixes the skill of every following episode, for evaluation.
     *
     * @param z the skill, or null to sample again from the prior
     */
    public void fixSkill(float[] z) {
        this.fixed = z == null ? null : z.clone();
        this.fixedIndex = -1;
        if (z != null && method == Method.DIAYN) {
            fixedIndex = RlTensors.argmax(z);
        }
    }

    private void drawSkill() {
        if (fixed != null) {
            skill = fixed.clone();
            skillIndex = fixedIndex;
            return;
        }
        skill = new float[skillDims];
        if (method == Method.DIAYN) {
            skillIndex = random.nextInt(skillDims);
            skill[skillIndex] = 1;
            return;
        }
        double norm = 0;
        for (int i = 0; i < skillDims; i++) {
            skill[i] = (float) random.nextGaussian();
            norm += skill[i] * skill[i];
        }
        for (int i = 0; i < skillDims; i++) {
            skill[i] /= (float) Math.sqrt(norm);
        }
    }

    private float[] augment(float[] obs) {
        float[] o = new float[obs.length + skillDims];
        System.arraycopy(obs, 0, o, 0, obs.length);
        System.arraycopy(skill, 0, o, obs.length, skillDims);
        return o;
    }

    @Override
    public Space observationSpace() {
        return Space.box(inner.observationSpace().size() + skillDims, -1, 1);
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom r) {
        drawSkill();
        float[] obs = inner.reset(r);
        lastState = state.apply(obs);
        return augment(obs);
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        float[] next = state.apply(s.observation());
        float reward = intrinsic(lastState, next);
        if (!evaluating) {
            replay.add(new Sample(lastState, next, skill.clone(), skillIndex));
            if (replay.size() > 20_000) {
                replay.remove(random.nextInt(replay.size()));
            }
            if (method != Method.RANDOM_SKILLS && ++steps % 2 == 0 && replay.size() >= 64) {
                train();
            }
        }
        lastState = next;
        return new Step(augment(s.observation()), reward, s.terminated(), s.truncated());
    }

    private float intrinsic(float[] s, float[] next) {
        switch (method) {
            case RANDOM_SKILLS -> {
                return 0;
            }
            case DIAYN -> {
                float[] logits = RlTensors.row(net, next);
                double max = Double.NEGATIVE_INFINITY;
                for (float l : logits) {
                    max = Math.max(max, l);
                }
                double sum = 0;
                for (float l : logits) {
                    sum += Math.exp(l - max);
                }
                double logq = logits[skillIndex] - max - Math.log(sum);
                return (float) (logq + Math.log(skillDims));
            }
            default -> {
                float[] a = RlTensors.row(net, s);
                float[] b = RlTensors.row(net, next);
                double r = 0;
                for (int i = 0; i < skillDims; i++) {
                    r += (b[i] - a[i]) * skill[i];
                }
                return (float) r;
            }
        }
    }

    private void train() {
        int b = 64;
        float[] s = new float[b * stateDims];
        float[] n = new float[b * stateDims];
        float[] z = new float[b * skillDims];
        for (int i = 0; i < b; i++) {
            Sample x = replay.get(random.nextInt(replay.size()));
            System.arraycopy(x.s(), 0, s, i * stateDims, stateDims);
            System.arraycopy(x.next(), 0, n, i * stateDims, stateDims);
            if (method == Method.DIAYN) {
                z[i * skillDims + x.index()] = 1;
            } else {
                System.arraycopy(x.z(), 0, z, i * skillDims, skillDims);
            }
        }
        Tensor zt = Tensor.of(Shape.of(b, skillDims), z);
        Var loss;
        if (method == Method.DIAYN) {
            Var logits = net.forward(Var.of(Tensor.of(Shape.of(b, stateDims), n)), true);
            loss = RlTensors.logSoftmax(logits).mul(zt).sum(1, true).mean().neg();
        } else {
            Var both = net.forward(Var.of(Tensor.concat(0, Tensor.of(Shape.of(b, stateDims), s),
                    Tensor.of(Shape.of(b, stateDims), n))), true);
            Var delta = both.slice(0, b, 2 * b).sub(both.slice(0, 0, b));
            Var align = delta.mul(zt).sum(1, true);
            if (method == Method.METRA) {
                float eps = 1e-3f;
                Var slack = delta.square().sum(1, true).neg().add(1f).minimum(eps);
                loss = align.mean().neg().sub(slack.mean().mul((float) lambda));
                double meanSlack = slack.value().mean().item();
                lambda = Math.max(0, lambda - meanSlack);
            } else {
                // InfoNCE: positives on the diagonal of delta z'^T, negatives the batch's other skills
                Var scores = delta.matmul(Var.of(zt.transpose()));
                Var lse = scores.logsumexp(1, true);
                loss = align.sub(lse).mean().neg();
            }
        }
        loss.backward();
        optimizer.step(net.parameters());
    }

    /**
     * Rolls out every evaluation skill greedily and counts the distinct cells of a grid over
     * {@code [-1, 1]^2} that the first two state values visit.
     *
     * @param agent the trained agent (acting on the wrapped observations)
     * @param skills evaluation skills
     * @param grid cells per axis
     * @param seed the episode seed
     * @return {@code {distinct cells, final positions [skill][2] flattened}}
     */
    public double[] coverage(Agent agent, int skills, int grid, long seed) {
        evaluating = true;
        Set<Integer> cells = new HashSet<>();
        double[] out = new double[1 + 2 * skills];
        for (int k = 0; k < skills; k++) {
            fixSkill(evaluationSkill(k, skills));
            agent.startEpisode();
            float[] obs = reset(new SplittableRandom(seed));
            Step s;
            float[] pos = lastState;
            do {
                s = step(agent.act(obs, false));
                obs = s.observation();
                pos = lastState;
                int cx = (int) Math.max(0, Math.min(grid - 1, Math.floor((pos[0] + 1) / 2 * grid)));
                int cy = (int) Math.max(0, Math.min(grid - 1, Math.floor((pos[1] + 1) / 2 * grid)));
                cells.add(cx * grid + cy);
            } while (!s.done());
            out[1 + 2 * k] = pos[0];
            out[2 + 2 * k] = pos[1];
        }
        fixSkill(null);
        evaluating = false;
        out[0] = cells.size();
        return out;
    }

    @Override
    public String toString() {
        return "Skills(" + method + ", " + inner + ")";
    }
}
