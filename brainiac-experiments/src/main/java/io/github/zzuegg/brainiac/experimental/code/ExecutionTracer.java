package io.github.zzuegg.brainiac.experimental.code;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * Line-by-line execution traces of straight-line list programs, and their tokenization, for
 * models that learn to predict what code does. A program is a sequence of primitive names run
 * left to right on an integer list, with semantics supplied by the caller (so any DSL fits). A
 * trace records, for every executed line, the line number, the primitive and the state (the
 * list) before and after it, the frames that Meta's Code World Model mid-trains on (CWM,
 * <a href="https://arxiv.org/abs/2510.02387">arXiv:2510.02387</a>: local-variable state, then
 * the executed line, then the resulting state, framed by call, line and return events).
 *
 * <p>{@link Format} lays a program and its input out as a prompt and a target in one of three
 * ways, the conditions of the execution-prediction comparison:
 * <ul>
 *   <li>{@link Format#DIRECT}: the output alone (Zaremba &amp; Sutskever, "Learning to
 *       execute", <a href="https://arxiv.org/abs/1410.4615">arXiv:1410.4615</a>);</li>
 *   <li>{@link Format#STATES}: a scratchpad of the state after every line, with no line
 *       markers (Nye et al., "Show your work",
 *       <a href="https://arxiv.org/abs/2112.00114">arXiv:2112.00114</a>);</li>
 *   <li>{@link Format#TRACE}: CWM-style frames {@code LINE op state SEP}, one per executed line;
 *       the line event names the primitive executed, so the next state depends only on the
 *       previous one and the line.</li>
 * </ul>
 * List values {@code >= values} share an overflow token.
 */
public final class ExecutionTracer {

    /** How a program's execution is written out. */
    public enum Format {
        /** Program and input, then the output. */
        DIRECT,
        /** Then every intermediate state, separated, then the end. */
        STATES,
        /** Then one {@code LINE op state SEP} frame per executed line, then the end. */
        TRACE
    }

    /**
     * One executed line.
     *
     * @param line the line number (0-based)
     * @param op the primitive
     * @param before the state before the line
     * @param after the state after it ({@code null} on a runtime error)
     */
    public record Frame(int line, String op, int[] before, int[] after) {
    }

    /**
     * A program's execution.
     *
     * @param program the program
     * @param input the input
     * @param frames one per executed line
     * @param output the final state ({@code null} on a runtime error)
     */
    public record Trace(List<String> program, int[] input, List<Frame> frames, int[] output) {
    }

    private final List<String> primitives;
    private final BiFunction<String, int[], int[]> semantics;
    private final int values;

    private ExecutionTracer(List<String> primitives, BiFunction<String, int[], int[]> semantics, int values) {
        this.primitives = List.copyOf(primitives);
        this.semantics = Objects.requireNonNull(semantics);
        this.values = values;
    }

    /**
     * @param primitives the DSL's primitive names
     * @param semantics applies a primitive to a list ({@code null} on failure)
     * @param values distinct list values with their own token
     * @return the tracer
     */
    public static ExecutionTracer of(List<String> primitives, BiFunction<String, int[], int[]> semantics,
            int values) {
        return new ExecutionTracer(primitives, semantics, values);
    }

    /** @return the primitives */
    public List<String> primitives() {
        return primitives;
    }

    /**
     * Runs a program and records every line.
     *
     * @param program the primitives
     * @param input the input list
     * @return the trace
     */
    public Trace trace(List<String> program, int[] input) {
        List<Frame> frames = new ArrayList<>();
        int[] state = input;
        for (int i = 0; i < program.size(); i++) {
            int[] next = semantics.apply(program.get(i), state);
            frames.add(new Frame(i, program.get(i), state, next));
            if (next == null) {
                return new Trace(List.copyOf(program), input, frames, null);
            }
            state = next;
        }
        return new Trace(List.copyOf(program), input, frames, state);
    }

    /**
     * Runs a program.
     *
     * @param program the primitives
     * @param input the input
     * @return the output, or {@code null} on a runtime error
     */
    public int[] run(List<String> program, int[] input) {
        return trace(program, input).output();
    }

    // ------------------------------------------------------------------ tokens

    /** @return the token of the overflow value */
    public int overflow() {
        return values;
    }

    /** @return the token of the first primitive */
    public int opBase() {
        return values + 1;
    }

    /** @return the call event, which opens the program */
    public int call() {
        return opBase() + primitives.size();
    }

    /** @return the line event, which opens a frame */
    public int lineToken() {
        return call() + 1;
    }

    /** @return the separator: ends the program, a state and a frame */
    public int sep() {
        return call() + 2;
    }

    /** @return the arrow that ends the prompt */
    public int arrow() {
        return call() + 3;
    }

    /** @return the end token */
    public int end() {
        return call() + 4;
    }

    /** @return the vocabulary size */
    public int vocabSize() {
        return call() + 5;
    }

    /**
     * @param v a list value
     * @return its token
     */
    public int valueToken(int v) {
        return v >= 0 && v < values ? v : values;
    }

    /**
     * @param op a primitive
     * @return its token
     */
    public int opToken(String op) {
        int i = primitives.indexOf(op);
        if (i < 0) {
            throw new IllegalArgumentException("unknown primitive " + op);
        }
        return opBase() + i;
    }

    private void addList(List<Integer> out, int[] xs) {
        for (int x : xs) {
            out.add(valueToken(x));
        }
    }

    private static int[] array(List<Integer> xs) {
        return xs.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * The prompt of every format: {@code CALL op... SEP input... ARROW}.
     *
     * @param program the program
     * @param input the input
     * @return the tokens
     */
    public int[] prompt(List<String> program, int[] input) {
        List<Integer> out = new ArrayList<>();
        out.add(call());
        for (String op : program) {
            out.add(opToken(op));
        }
        out.add(sep());
        addList(out, input);
        out.add(arrow());
        return array(out);
    }

    /**
     * The target of a format, for teacher forcing.
     *
     * @param format the format
     * @param trace the execution (must have finished without error)
     * @return the tokens, end token included
     */
    public int[] target(Format format, Trace trace) {
        List<Integer> out = new ArrayList<>();
        switch (format) {
            case DIRECT -> addList(out, trace.output());
            case STATES -> {
                for (Frame f : trace.frames()) {
                    addList(out, f.after());
                    out.add(sep());
                }
            }
            case TRACE -> {
                for (Frame f : trace.frames()) {
                    out.add(lineToken());
                    out.add(opToken(f.op()));
                    addList(out, f.after());
                    out.add(sep());
                }
            }
        }
        out.add(end());
        return array(out);
    }

    /**
     * Reads the predicted output off a generated target: the values after the last separator
     * (or all values, for {@link Format#DIRECT}), up to the end token.
     *
     * @param format the format
     * @param generated the generated tokens
     * @return the output list, or {@code null} if malformed
     */
    public int[] output(Format format, int[] generated) {
        int stop = generated.length;
        for (int i = 0; i < generated.length; i++) {
            if (generated[i] == end()) {
                stop = i;
                break;
            }
        }
        if (stop == generated.length) {
            return null;
        }
        int[] body = Arrays.copyOf(generated, stop);
        if (format == Format.DIRECT) {
            for (int t : body) {
                if (t > values) {
                    return null;
                }
            }
            return body;
        }
        // the last complete state: between the two last separators (or the start)
        int last = -1;
        int before = -1;
        for (int i = 0; i < body.length; i++) {
            if (body[i] == sep()) {
                before = last;
                last = i;
            }
        }
        if (last < 0) {
            return null;
        }
        int from = before + 1;
        if (format == Format.TRACE) {
            from += 2;   // LINE op
        }
        if (from > last) {
            return null;
        }
        int[] state = Arrays.copyOfRange(body, from, last);
        for (int t : state) {
            if (t > values) {
                return null;
            }
        }
        return state;
    }

    /**
     * Encodes a whole trace (or any list of states) for feedback: one {@code state SEP} per line
     * of the trace, truncated to {@code maxTokens}.
     *
     * @param trace the execution
     * @param maxTokens the most tokens
     * @return the tokens
     */
    public int[] stateTokens(Trace trace, int maxTokens) {
        List<Integer> out = new ArrayList<>();
        for (Frame f : trace.frames()) {
            if (f.after() == null) {
                break;
            }
            addList(out, f.after());
            out.add(sep());
        }
        int[] a = array(out);
        return a.length > maxTokens ? Arrays.copyOf(a, maxTokens) : a;
    }
}
