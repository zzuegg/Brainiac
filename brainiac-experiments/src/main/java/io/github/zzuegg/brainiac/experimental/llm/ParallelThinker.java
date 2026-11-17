package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Native parallel thinking, ParaThinker (Wen et al., "ParaThinker: native parallel thinking as a
 * new paradigm to scale LLM test-time compute",
 * <a href="https://arxiv.org/abs/2509.04475">arXiv:2509.04475</a>): instead of one long chain of
 * thought, several reasoning paths are generated in parallel (one batched pass, so the latency of
 * one path) and a <em>trained</em> summarizer reads all of them and writes the answer. Unlike a
 * majority vote it can weigh a path by its content, recognise a minority path that is consistent,
 * and use partial work from wrong paths.
 *
 * <p>Here the paths come from a {@link WindowLm} generator sampled at a temperature, each right-aligned
 * in a fixed-length block so every path's conclusion sits at the same input positions, and the merger is a second
 * {@link WindowLm} whose prompt is the question (optionally; without it the merger has to
 * aggregate rather than solve the task again) followed by the concatenated paths, and whose
 * completion is the answer. The merger is
 * trained by supervised learning on the generator's own paths for training questions.
 */
public final class ParallelThinker {
    private final WindowLm generator;
    private final WindowLm merger;
    private final int paths;
    private final int pathTokens;
    private final int readTokens;
    private final int eos;
    private final int pad;
    private final double temperature;
    private final int maxAnswer;
    private final int questionTokens;

    private ParallelThinker(WindowLm generator, int paths, int pathTokens, int readTokens, int eos, int pad, double temperature,
            int maxAnswer, int questionTokens, int hidden, long seed) {
        this.generator = generator;
        this.paths = paths;
        this.pathTokens = pathTokens;
        this.readTokens = readTokens;
        this.eos = eos;
        this.pad = pad;
        this.temperature = temperature;
        this.maxAnswer = maxAnswer;
        this.questionTokens = questionTokens;
        this.merger = WindowLm.of(generator.vocabSize(), questionTokens + paths * readTokens + 1, hidden, seed);
    }

    /**
     * Creates a parallel thinker with an untrained merger.
     *
     * @param generator the path generator, left unchanged
     * @param paths parallel paths per question
     * @param pathTokens the longest path
     * @param readTokens how many tokens at the end of each path the merger reads (its
     *     conclusion); shorter paths are padded in front
     * @param eos the end token of a path and of the answer
     * @param pad the padding token
     * @param temperature the sampling temperature of the paths
     * @param maxAnswer the longest merged answer
     * @param questionTokens the question block the merger reads before the paths; 0 hides the
     *     question
     * @param hidden the merger's hidden width
     * @param seed the merger's seed
     * @return the thinker
     */
    public static ParallelThinker of(WindowLm generator, int paths, int pathTokens, int readTokens, int eos, int pad,
            double temperature, int maxAnswer, int questionTokens, int hidden, long seed) {
        return new ParallelThinker(generator, paths, pathTokens, readTokens, eos, pad, temperature, maxAnswer, questionTokens,
                hidden, seed);
    }

    /**
     * Samples the parallel paths of several questions in one batch.
     *
     * @param prompts the questions
     * @param random the random source
     * @return {@code [question][path]} completions
     */
    public int[][][] paths(List<int[]> prompts, SplittableRandom random) {
        int[][] batch = new int[prompts.size() * paths][];
        for (int i = 0; i < batch.length; i++) {
            batch[i] = prompts.get(i / paths);
        }
        int[][] out = generator.sample(batch, pathTokens, eos, temperature, random);
        int[][][] grouped = new int[prompts.size()][paths][];
        for (int i = 0; i < out.length; i++) {
            grouped[i / paths][i % paths] = out[i];
        }
        return grouped;
    }

    /**
     * The merger's prompt: the paths, each right-aligned in a block of the same length (padded
     * in front), so every path's conclusion sits at the same input positions.
     *
     * @param question the question
     * @param pathSet the paths of one question
     * @return the merger input tokens
     */
    public int[] mergerPrompt(int[] question, int[][] pathSet) {
        int[] in = new int[questionTokens + paths * readTokens];
        Arrays.fill(in, pad);
        int q = Math.min(questionTokens, question.length);
        System.arraycopy(question, question.length - q, in, questionTokens - q, q);
        for (int p = 0; p < paths; p++) {
            int len = Math.min(readTokens, pathSet[p].length);
            System.arraycopy(pathSet[p], pathSet[p].length - len, in, questionTokens + (p + 1) * readTokens - len,
                    len);
        }
        return in;
    }

    /**
     * Trains the merger on fresh paths for the training questions.
     *
     * @param prompts training questions
     * @param answer the target answer completion of a question
     * @param rounds path sets sampled per question
     * @param steps supervised steps
     * @param learningRate the learning rate
     * @param random the random source
     * @return the final loss per token
     */
    public float train(List<int[]> prompts, Function<int[], int[]> answer, int rounds, int steps, float learningRate,
            SplittableRandom random) {
        List<int[]> in = new ArrayList<>();
        List<int[]> target = new ArrayList<>();
        for (int r = 0; r < rounds; r++) {
            int[][][] sets = paths(prompts, random);
            for (int q = 0; q < prompts.size(); q++) {
                in.add(mergerPrompt(prompts.get(q), sets[q]));
                target.add(answer.apply(prompts.get(q)));
            }
        }
        Optimizer opt = Optimizer.adam(learningRate);
        float loss = 0;
        for (int s = 0; s < steps; s++) {
            loss = merger.supervisedStep(in, target, opt);
        }
        return loss;
    }

    /**
     * An answer with its cost.
     *
     * @param answer the merged answer
     * @param paths the paths it read
     * @param tokens tokens generated, paths and answer
     */
    public record Output(int[] answer, int[][] paths, int tokens) {
    }

    /**
     * Answers questions: parallel paths, then the merger.
     *
     * @param prompts the questions
     * @param random the random source
     * @return one output per question
     */
    public List<Output> answer(List<int[]> prompts, SplittableRandom random) {
        int[][][] sets = paths(prompts, random);
        List<Output> out = new ArrayList<>();
        for (int q = 0; q < prompts.size(); q++) {
            int[] merged = merger.greedy(mergerPrompt(prompts.get(q), sets[q]), maxAnswer, eos);
            int tokens = merged.length;
            for (int[] p : sets[q]) {
                tokens += p.length;
            }
            out.add(new Output(merged, sets[q], tokens));
        }
        return out;
    }

    /** @return the merger, trained by {@link #train} */
    public WindowLm merger() {
        return merger;
    }

    @Override
    public String toString() {
        return "ParallelThinker(paths=" + paths + ", T=" + temperature + ")";
    }
}
