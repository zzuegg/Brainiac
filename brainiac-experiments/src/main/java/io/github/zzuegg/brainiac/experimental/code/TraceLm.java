package io.github.zzuegg.brainiac.experimental.code;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A language model trained on execution text ({@link ExecutionTracer}): direct output
 * prediction, a scratchpad of intermediate states, or CWM-style execution traces (Meta CWM,
 * <a href="https://arxiv.org/abs/2510.02387">arXiv:2510.02387</a>). Output prediction is then
 * text generation: greedy decoding of the target after the prompt. In the trace format the
 * line events are part of the program text, which the model does not have to invent: for a
 * straight-line program the decoder writes {@code LINE op} for the next line itself (as an
 * interpreter steps through the source) and the model predicts only the resulting state, so
 * each prediction depends on one line and one state however long the program is.
 *
 * <p>The model is a {@link WindowLm}, which sees a fixed window of recent tokens: enough for one
 * frame of a trace, but not for a long program and its input at once, which is exactly the
 * difference between the formats.
 */
public final class TraceLm {

    private final ExecutionTracer tracer;
    private final ExecutionTracer.Format format;
    private final WindowLm lm;
    private long tokens;

    private TraceLm(ExecutionTracer tracer, ExecutionTracer.Format format, WindowLm lm) {
        this.tracer = Objects.requireNonNull(tracer);
        this.format = Objects.requireNonNull(format);
        this.lm = Objects.requireNonNull(lm);
    }

    /**
     * @param tracer the tracer and tokenizer
     * @param format the text it learns
     * @param window tokens the model sees
     * @param hidden the hidden width
     * @param seed the seed
     * @return an untrained model
     */
    public static TraceLm of(ExecutionTracer tracer, ExecutionTracer.Format format, int window, int hidden,
            long seed) {
        return new TraceLm(tracer, format, WindowLm.of(tracer.vocabSize(), window, hidden, seed));
    }

    /** @return the language model */
    public WindowLm model() {
        return lm;
    }

    /** @return the format */
    public ExecutionTracer.Format format() {
        return format;
    }

    /** @return tokens decoded by {@link #predict} so far */
    public long tokensDecoded() {
        return tokens;
    }

    /**
     * One supervised step on the execution text of some programs.
     *
     * @param programs the programs
     * @param inputs their inputs
     * @param optimizer the optimizer
     * @return the mean token loss
     */
    public float trainStep(List<List<String>> programs, List<int[]> inputs, Optimizer optimizer) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < programs.size(); i++) {
            ExecutionTracer.Trace t = tracer.trace(programs.get(i), inputs.get(i));
            if (t.output() == null) {
                continue;
            }
            ps.add(tracer.prompt(programs.get(i), inputs.get(i)));
            cs.add(tracer.target(format, t));
        }
        return ps.isEmpty() ? 0 : lm.supervisedStep(ps, cs, optimizer);
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            best = v[i] > v[best] ? i : best;
        }
        return best;
    }

    /**
     * Predicts a program's output by greedy decoding.
     *
     * @param program the program
     * @param input the input
     * @param maxTokens the most tokens of target text
     * @return the predicted output, or {@code null} if the text is malformed
     */
    public int[] predict(List<String> program, int[] input, int maxTokens) {
        int[] prompt = tracer.prompt(program, input);
        List<Integer> gen = new ArrayList<>();
        int line = 0;
        if (format == ExecutionTracer.Format.TRACE) {
            gen.add(tracer.lineToken());
            gen.add(tracer.opToken(program.get(line++)));
        }
        int stateTokens = 0;
        while (gen.size() < maxTokens) {
            int[] seq = Arrays.copyOf(prompt, prompt.length + gen.size());
            for (int i = 0; i < gen.size(); i++) {
                seq[prompt.length + i] = gen.get(i);
            }
            int tok = argmax(lm.logits(seq));
            tokens++;
            if (format == ExecutionTracer.Format.TRACE && tok != tracer.sep() && tok != tracer.end()
                    && (tok > tracer.overflow() || stateTokens >= 16)) {
                tok = tracer.sep();   // only values belong inside a frame
            }
            gen.add(tok);
            if (tok == tracer.end()) {
                break;
            }
            if (format == ExecutionTracer.Format.TRACE) {
                stateTokens = tok == tracer.sep() ? 0 : stateTokens + 1;
                if (tok == tracer.sep()) {
                    if (line < program.size()) {
                        gen.add(tracer.lineToken());
                        gen.add(tracer.opToken(program.get(line++)));
                    } else {
                        gen.add(tracer.end());
                        break;
                    }
                }
            }
        }
        return tracer.output(format, gen.stream().mapToInt(Integer::intValue).toArray());
    }
}
