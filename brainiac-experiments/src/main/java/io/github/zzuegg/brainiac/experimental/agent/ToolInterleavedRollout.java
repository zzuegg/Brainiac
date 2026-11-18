package io.github.zzuegg.brainiac.experimental.agent;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.ToolRuntime;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Reinforcement learning with tool calls inside the rollout, as in ReTool (Feng et al. 2025,
 * <a href="https://arxiv.org/abs/2504.11536">arXiv:2504.11536</a>) and ToolRL (Qian et al. 2025,
 * <a href="https://arxiv.org/abs/2504.13958">arXiv:2504.13958</a>). Groups of completions are
 * sampled in a batch; whenever a completion writes the result marker the {@link ToolRuntime.Tool}
 * runs and its result is spliced in, and the completion continues conditioned on it. The update
 * is GRPO's group-normalised advantage ({@link GrpoTrainer#advantages}) applied by policy
 * gradient to the tokens the model wrote only: spliced tool output is masked out, split off with
 * {@link ToolRuntime#trainingChunks}. ({@link GrpoTrainer#update} itself has no token mask,
 * hence this wrapper.)
 *
 * <p>Generation can be constrained to always or never call, the two fixed baselines.
 */
public final class ToolInterleavedRollout {

    /** Whether the model may call. */
    public enum CallPolicy {
        /** The model decides. */
        FREE,
        /** The first token is forced to open a call. */
        ALWAYS,
        /** Opening a call is masked out. */
        NEVER
    }

    /**
     * One rollout.
     *
     * @param prompt the prompt
     * @param completion everything after the prompt, spliced results included
     * @param modelWritten per completion token, whether the model wrote it
     * @param calls tool calls that ran
     */
    public record Rollout(int[] prompt, int[] completion, boolean[] modelWritten, int calls) {
    }

    /** Scores a rollout. */
    @FunctionalInterface
    public interface Reward {
        /**
         * @param prompt the prompt
         * @param rollout the rollout
         * @return the reward
         */
        float score(int[] prompt, Rollout rollout);
    }

    /**
     * What a training step did.
     *
     * @param meanReward the mean reward of the sampled rollouts
     * @param callsPerRollout tool calls per rollout
     */
    public record Stats(double meanReward, double callsPerRollout) {
    }

    private final ToolRuntime runtime;
    private final ToolRuntime.Tool tool;
    private final Tokenizer tokenizer;
    private final int open;
    private final int marker;
    private final int close;
    private final int end;
    private final int maxNew;

    private ToolInterleavedRollout(Tokenizer tokenizer, ToolRuntime.Tool tool, char open, char marker, char close,
            int end, int maxNew) {
        this.tokenizer = Objects.requireNonNull(tokenizer);
        this.tool = Objects.requireNonNull(tool);
        this.runtime = ToolRuntime.of(tokenizer, tool, open, marker, close);
        this.open = tokenizer.encode(String.valueOf(open))[0];
        this.marker = tokenizer.encode(String.valueOf(marker))[0];
        this.close = tokenizer.encode(String.valueOf(close))[0];
        this.end = end;
        this.maxNew = maxNew;
    }

    /**
     * @param tokenizer the model's tokenizer (one token per bracket character)
     * @param tool the tool
     * @param open the character that opens a call
     * @param marker the character after which the result is spliced in
     * @param close the character that closes a call (spliced by the runtime)
     * @param end the end token
     * @param maxNew the most tokens the model may write per rollout
     * @return the rollout engine
     */
    public static ToolInterleavedRollout of(Tokenizer tokenizer, ToolRuntime.Tool tool, char open, char marker,
            char close, int end, int maxNew) {
        return new ToolInterleavedRollout(tokenizer, tool, open, marker, close, end, maxNew);
    }

    /** @return the runtime with the same tool and brackets */
    public ToolRuntime runtime() {
        return runtime;
    }

    /**
     * Generates rollouts for a batch of prompts, one network call per position.
     *
     * @param lm the model
     * @param prompts the prompts
     * @param policy whether calls are free, forced or forbidden
     * @param temperature the sampling temperature, 0 for greedy
     * @param random the random source
     * @return one rollout per prompt
     */
    public List<Rollout> generate(WindowLm lm, List<int[]> prompts, CallPolicy policy, double temperature,
            SplittableRandom random) {
        int n = prompts.size();
        List<List<Integer>> seqs = new ArrayList<>();
        List<List<Boolean>> written = new ArrayList<>();
        int[] callStart = new int[n];
        int[] calls = new int[n];
        int[] produced = new int[n];
        boolean[] done = new boolean[n];
        for (int[] p : prompts) {
            List<Integer> s = new ArrayList<>();
            for (int t : p) {
                s.add(t);
            }
            seqs.add(s);
            written.add(new ArrayList<>());
        }
        Arrays.fill(callStart, -1);
        for (int step = 0; step < maxNew; step++) {
            List<int[]> live = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    live.add(seqs.get(i).stream().mapToInt(Integer::intValue).toArray());
                    who.add(i);
                }
            }
            if (live.isEmpty()) {
                break;
            }
            float[] logits = lm.logits(live).value().toArray();
            int v = lm.vocabSize();
            for (int j = 0; j < who.size(); j++) {
                int i = who.get(j);
                float[] row = Arrays.copyOfRange(logits, j * v, (j + 1) * v);
                if (policy == CallPolicy.NEVER) {
                    row[open] = Float.NEGATIVE_INFINITY;
                }
                int tok = policy == CallPolicy.ALWAYS && produced[i] == 0 ? open : sample(row, temperature, random);
                List<Integer> s = seqs.get(i);
                s.add(tok);
                written.get(i).add(true);
                produced[i]++;
                if (tok == open) {
                    callStart[i] = s.size();
                } else if (tok == marker && callStart[i] >= 0) {
                    int[] args = s.subList(callStart[i], s.size() - 1).stream().mapToInt(Integer::intValue).toArray();
                    for (int r : tokenizer.encode(tool.call(tokenizer.decode(args)))) {
                        s.add(r);
                        written.get(i).add(false);
                    }
                    s.add(close);
                    written.get(i).add(false);
                    callStart[i] = -1;
                    calls[i]++;
                } else if (tok == end) {
                    done[i] = true;
                }
            }
        }
        List<Rollout> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] p = prompts.get(i);
            int[] all = seqs.get(i).stream().mapToInt(Integer::intValue).toArray();
            boolean[] w = new boolean[all.length - p.length];
            for (int k = 0; k < w.length; k++) {
                w[k] = written.get(i).get(k);
            }
            out.add(new Rollout(p.clone(), Arrays.copyOfRange(all, p.length, all.length), w, calls[i]));
        }
        return out;
    }

    private static int sample(float[] logits, double temperature, SplittableRandom random) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        if (temperature <= 0) {
            return best;
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp((logits[i] - logits[best]) / temperature);
            sum += p[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return best;
    }

    /**
     * One GRPO step with tools: a group of rollouts per prompt, group-normalised advantages,
     * and a policy-gradient update on the model-written tokens only.
     *
     * @param lm the policy
     * @param promptSource draws a prompt
     * @param prompts prompts per step
     * @param group rollouts per prompt
     * @param temperature the sampling temperature
     * @param reward the reward
     * @param optimizer the optimizer
     * @param random the random source
     * @return what happened
     */
    public Stats step(WindowLm lm, Function<SplittableRandom, int[]> promptSource, int prompts, int group,
            double temperature, Reward reward, Optimizer optimizer, SplittableRandom random) {
        List<int[]> batch = new ArrayList<>();
        for (int p = 0; p < prompts; p++) {
            int[] prompt = promptSource.apply(random);
            for (int g = 0; g < group; g++) {
                batch.add(prompt);
            }
        }
        List<Rollout> rollouts = generate(lm, batch, CallPolicy.FREE, temperature, random);
        List<int[]> contexts = new ArrayList<>();
        List<int[]> chunks = new ArrayList<>();
        List<Float> advantages = new ArrayList<>();
        double sum = 0;
        double callSum = 0;
        for (int p = 0; p < prompts; p++) {
            float[] r = new float[group];
            for (int g = 0; g < group; g++) {
                Rollout ro = rollouts.get(p * group + g);
                r[g] = reward.score(ro.prompt(), ro);
                sum += r[g];
                callSum += ro.calls();
            }
            float[] a = GrpoTrainer.advantages(r, true);
            for (int g = 0; g < group; g++) {
                if (a[g] == 0) {
                    continue;
                }
                Rollout ro = rollouts.get(p * group + g);
                for (int[][] chunk : runtime.trainingChunks(ro.prompt(), ro.completion())) {
                    contexts.add(chunk[0]);
                    chunks.add(chunk[1]);
                    advantages.add(a[g]);
                }
            }
        }
        policyGradient(lm, contexts, chunks, advantages, optimizer);
        return new Stats(sum / rollouts.size(), callSum / rollouts.size());
    }

    /**
     * A policy-gradient update on chosen chunks: minimises
     * {@code -mean over tokens of advantage * log p(token)}, each chunk's tokens sharing its
     * advantage. Chunks are what the model wrote; anything else (tool output, user turns) is
     * context only.
     *
     * @param lm the policy
     * @param contexts what each chunk was written after
     * @param chunks the model-written tokens
     * @param advantages one advantage per chunk
     * @param optimizer the optimizer
     */
    public static void policyGradient(WindowLm lm, List<int[]> contexts, List<int[]> chunks, List<Float> advantages,
            Optimizer optimizer) {
        int tokens = 0;
        for (int[] c : chunks) {
            tokens += c.length;
        }
        if (tokens == 0) {
            return;
        }
        float[] w = new float[tokens];
        int k = 0;
        for (int i = 0; i < chunks.size(); i++) {
            for (int t = 0; t < chunks.get(i).length; t++) {
                w[k++] = advantages.get(i);
            }
        }
        Var loss = lm.logProbs(contexts, chunks).mul(Var.of(Tensor.of(Shape.of(tokens, 1), w))).mean().neg();
        loss.backward();
        optimizer.step(lm.parameters());
    }

    @Override
    public String toString() {
        return "ToolInterleavedRollout(maxNew " + maxNew + ")";
    }
}
