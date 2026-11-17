package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MemAgent (Yu et al., "MemAgent: reshaping long-context LLM with multi-conv RL-based memory
 * agent", <a href="https://arxiv.org/abs/2507.02259">arXiv:2507.02259</a>): a model with a short
 * window reads an arbitrarily long input chunk by chunk and, after every chunk, <em>overwrites</em>
 * a fixed-size memory with new tokens of its own choosing; the answer is then written from the
 * memory alone. Cost is linear in the input length and the window never grows. What to keep is
 * learned by reinforcement learning on the final answer only: every memory write of an episode
 * gets that episode's advantage, normalized within a group of episodes on the same question
 * (the paper's multi-conversation DAPO, here plain group-normalized REINFORCE).
 *
 * <p>A write step reads {@code query ++ memory ++ chunk} and writes {@code memoryTokens} tokens
 * followed by the end token; a short write is padded.
 */
public final class MemoryOverwriteAgent {
    private final WindowLm lm;
    private final int memoryTokens;
    private final int chunk;
    private final int end;
    private final int pad;
    private long calls;

    private MemoryOverwriteAgent(WindowLm lm, int memoryTokens, int chunk, int end, int pad) {
        this.lm = lm;
        this.memoryTokens = memoryTokens;
        this.chunk = chunk;
        this.end = end;
        this.pad = pad;
    }

    /**
     * Creates an agent.
     *
     * @param lm the memory writer, trained in place; its window should cover a write prompt
     * @param memoryTokens the memory size
     * @param chunk tokens read per step
     * @param end the token that ends a write
     * @param pad the token of an empty memory slot
     * @return the agent
     */
    public static MemoryOverwriteAgent of(WindowLm lm, int memoryTokens, int chunk, int end, int pad) {
        return new MemoryOverwriteAgent(lm, memoryTokens, chunk, end, pad);
    }

    /**
     * One read of a long input.
     *
     * @param prompts the write prompts, one per chunk
     * @param writes the writes, each {@code memoryTokens} tokens and the end token
     * @param memory the final memory
     */
    public record Episode(List<int[]> prompts, List<int[]> writes, int[] memory) {
    }

    /** @return an empty memory */
    public int[] emptyMemory() {
        int[] m = new int[memoryTokens];
        Arrays.fill(m, pad);
        return m;
    }

    /**
     * The write prompt.
     *
     * @param query the question
     * @param memory the current memory
     * @param piece the chunk
     * @return {@code query ++ memory ++ piece}
     */
    public int[] writePrompt(int[] query, int[] memory, int[] piece) {
        int[] p = Arrays.copyOf(query, query.length + memory.length + piece.length);
        System.arraycopy(memory, 0, p, query.length, memory.length);
        System.arraycopy(piece, 0, p, query.length + memory.length, piece.length);
        return p;
    }

    /**
     * Reads several long inputs in lockstep, one batched write per chunk position.
     *
     * @param contexts the inputs
     * @param queries one question per input
     * @param temperature write sampling temperature; 0 is greedy
     * @param random the random source
     * @return one episode per input
     */
    public List<Episode> read(List<int[]> contexts, List<int[]> queries, double temperature,
            SplittableRandom random) {
        int n = contexts.size();
        int[][] memory = new int[n][];
        List<List<int[]>> prompts = new ArrayList<>();
        List<List<int[]>> writes = new ArrayList<>();
        int longest = 0;
        for (int i = 0; i < n; i++) {
            memory[i] = emptyMemory();
            prompts.add(new ArrayList<>());
            writes.add(new ArrayList<>());
            longest = Math.max(longest, contexts.get(i).length);
        }
        for (int start = 0; start < longest; start += chunk) {
            List<int[]> batch = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int[] c = contexts.get(i);
                if (start < c.length) {
                    batch.add(writePrompt(queries.get(i), memory[i],
                            Arrays.copyOfRange(c, start, Math.min(c.length, start + chunk))));
                    who.add(i);
                }
            }
            int[][] out = lm.sample(batch.toArray(int[][]::new), memoryTokens + 1, end, temperature, random);
            calls += batch.size();
            for (int j = 0; j < out.length; j++) {
                int i = who.get(j);
                int[] w = out[j];
                int[] m = emptyMemory();
                for (int k = 0; k < Math.min(memoryTokens, w.length); k++) {
                    m[k] = w[k] == end ? pad : w[k];
                }
                prompts.get(i).add(batch.get(j));
                writes.get(i).add(w);
                memory[i] = m;
            }
        }
        List<Episode> episodes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            episodes.add(new Episode(prompts.get(i), writes.get(i), memory[i]));
        }
        return episodes;
    }

    /**
     * One policy-gradient step: each group of consecutive episodes answered the same question;
     * a write's advantage is its episode's reward minus the group mean, over the group's
     * standard deviation.
     *
     * @param episodes the episodes, in groups of {@code groupSize}
     * @param rewards one reward per episode
     * @param groupSize episodes per question
     * @param optimizer the optimizer
     * @return the share of episodes with a nonzero advantage
     */
    public double reinforce(List<Episode> episodes, float[] rewards, int groupSize, Optimizer optimizer) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        int used = 0;
        for (int g = 0; g + groupSize <= episodes.size(); g += groupSize) {
            float[] a = GrpoTrainer.advantages(Arrays.copyOfRange(rewards, g, g + groupSize), true);
            for (int e = 0; e < groupSize; e++) {
                if (a[e] == 0) {
                    continue;
                }
                used++;
                Episode ep = episodes.get(g + e);
                for (int t = 0; t < ep.prompts().size(); t++) {
                    ps.add(ep.prompts().get(t));
                    cs.add(ep.writes().get(t));
                    for (int k = 0; k < ep.writes().get(t).length; k++) {
                        adv.add(a[e]);
                    }
                }
            }
        }
        if (ps.isEmpty()) {
            return 0;
        }
        float[] w = new float[adv.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = adv.get(i);
        }
        Var loss = lm.logProbs(ps, cs).mul(Var.of(Tensor.of(Shape.of(w.length, 1), w))).mean().neg();
        loss.backward();
        optimizer.step(lm.parameters());
        return (double) used / episodes.size();
    }

    /** @return write calls so far */
    public long calls() {
        return calls;
    }

    /** @return the memory writer */
    public WindowLm model() {
        return lm;
    }

    @Override
    public String toString() {
        return "MemoryOverwriteAgent(memory=" + memoryTokens + ", chunk=" + chunk + ")";
    }
}
