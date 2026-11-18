package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiFunction;

/**
 * RobustFill-style neural program generation from input/output examples: an
 * encoder-decoder ({@link TransformerSeq2Seq}) reads each example pair as one token
 * sequence ({@code input SEP output}), decodes the program token by token for every pair
 * at once, and pools the per-example next-token scores by a maximum ("late pooling"), so
 * the program is conditioned on all examples. Decoding is a beam search with execution
 * filtering: finished candidates are run on the examples and only those consistent with
 * all of them are returned, best first.
 *
 * <p>Programs are sequences of primitive names over integer lists, executed by a
 * caller-supplied semantics, so the generator is independent of the DSL. List values
 * {@code >= VALUES} share an overflow token.
 *
 * <p>Paper: Devlin, Uesato, Bhupatiraju, Singh, Mohamed, Kohli, "RobustFill: Neural
 * Program Learning under Noisy I/O" (ICML 2017),
 * <a href="https://arxiv.org/abs/1703.07469">arXiv:1703.07469</a>.
 */
public final class ProgramGenerator {

    /** Distinct list values with their own token. */
    public static final int VALUES = 40;
    private static final int OVERFLOW = VALUES;
    private static final int SEP = VALUES + 1;
    private static final int PAD = VALUES + 2;
    private static final int SOURCE_VOCAB = VALUES + 3;
    private static final int SOURCE_LENGTH = 16;

    private final List<String> primitives;
    private final BiFunction<String, int[], int[]> semantics;
    private final int maxLength;
    private final int end;
    private final TransformerSeq2Seq model;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);

    /**
     * Creates an untrained generator.
     *
     * @param primitives the DSL's primitive names
     * @param semantics applies a primitive to a list ({@code null} on failure)
     * @param maxLength the longest program it writes
     * @param width the model width
     * @param seed the seed
     */
    public ProgramGenerator(List<String> primitives, BiFunction<String, int[], int[]> semantics, int maxLength,
            int width, long seed) {
        this.primitives = List.copyOf(primitives);
        this.semantics = semantics;
        this.maxLength = maxLength;
        this.end = primitives.size();
        this.model = TransformerSeq2Seq.of(SOURCE_VOCAB, primitives.size() + 1, width, 4, 1, seed);
    }

    private static int[] source(int[] in, int[] out) {
        int[] s = new int[SOURCE_LENGTH];
        Arrays.fill(s, PAD);
        int k = 0;
        for (int i = 0; i < Math.min(in.length, 7) && k < SOURCE_LENGTH; i++) {
            s[k++] = Math.min(OVERFLOW, Math.max(0, in[i]));
        }
        s[k++] = SEP;
        for (int i = 0; i < out.length && k < SOURCE_LENGTH; i++) {
            s[k++] = Math.min(OVERFLOW, Math.max(0, out[i]));
        }
        return s;
    }

    /** The decoder's next-token log-probabilities after each prefix, pooled over examples. */
    private Var pooledLogits(List<int[][]> inputs, List<int[][]> outputs, int[][] prefixes, boolean training) {
        int b = inputs.size();
        int e = inputs.get(0).length;
        int[][] src = new int[b * e][];
        int[][] dec = new int[b * e][];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < e; j++) {
                src[i * e + j] = source(inputs.get(i)[j], outputs.get(i)[j]);
                dec[i * e + j] = prefixes[i];
            }
        }
        int len = prefixes[0].length;
        Var logits = model.decode(model.encode(src, training), dec, training);   // [b*e, len, V]
        return logits.reshape(b, e, len, end + 1).max(1, false);                  // late max-pooling
    }

    /**
     * One teacher-forced training step.
     *
     * @param inputs per task, its example inputs (the same count for every task)
     * @param outputs per task, its outputs
     * @param programs per task, the program that produced them
     * @return the loss
     */
    public float trainStep(List<int[][]> inputs, List<int[][]> outputs, List<List<String>> programs) {
        int b = programs.size();
        int len = maxLength + 1;
        int[][] prefixes = new int[b][len];
        float[] target = new float[b * len];
        for (int i = 0; i < b; i++) {
            List<String> p = programs.get(i);
            prefixes[i][0] = end + 1;   // begin-of-sequence
            for (int t = 0; t < len; t++) {
                int tok = t < p.size() ? primitives.indexOf(p.get(t)) : t == p.size() ? end : -1;
                target[i * len + t] = tok;
                if (t + 1 < len) {
                    prefixes[i][t + 1] = Math.max(0, tok);
                }
            }
        }
        Var logits = pooledLogits(inputs, outputs, prefixes, true).reshape(b * len, end + 1);
        Var loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(b * len), target));
        float v = loss.value().item();
        loss.backward();
        optimizer.step(model.parameters());
        return v;
    }

    private record Beam(List<String> program, double logP) {
    }

    /**
     * Beam search with execution filtering.
     *
     * @param inputs the task's example inputs
     * @param outputs its outputs
     * @param width the beam width
     * @return the finished programs that reproduce every example, most probable first
     */
    public List<List<String>> search(int[][] inputs, int[][] outputs, int width) {
        List<Beam> beams = List.of(new Beam(List.of(), 0));
        List<Beam> consistent = new ArrayList<>();
        for (int t = 0; t <= maxLength && !beams.isEmpty(); t++) {
            int[][] prefixes = new int[beams.size()][t + 1];
            List<int[][]> ins = new ArrayList<>();
            List<int[][]> outs = new ArrayList<>();
            for (int i = 0; i < beams.size(); i++) {
                prefixes[i][0] = end + 1;
                for (int k = 0; k < t; k++) {
                    prefixes[i][k + 1] = primitives.indexOf(beams.get(i).program().get(k));
                }
                ins.add(inputs);
                outs.add(outputs);
            }
            int t0 = t;
            float[] logp = Var.noGrad(() -> {
                Var z = pooledLogits(ins, outs, prefixes, false).slice(1, t0, t0 + 1).reshape(prefixes.length,
                        end + 1);
                return z.sub(z.logsumexp(1, true)).value().toArray();
            });
            List<Beam> next = new ArrayList<>();
            for (int i = 0; i < beams.size(); i++) {
                Beam b = beams.get(i);
                for (int tok = 0; tok <= end; tok++) {
                    double s = b.logP() + logp[i * (end + 1) + tok];
                    if (tok == end) {
                        if (!b.program().isEmpty() && consistent(b.program(), inputs, outputs)) {
                            consistent.add(new Beam(b.program(), s));
                        }
                    } else if (t < maxLength) {
                        List<String> p = new ArrayList<>(b.program());
                        p.add(primitives.get(tok));
                        next.add(new Beam(p, s));
                    }
                }
            }
            next.sort(Comparator.comparingDouble(Beam::logP).reversed());
            beams = next.subList(0, Math.min(width, next.size()));
        }
        consistent.sort(Comparator.comparingDouble(Beam::logP).reversed());
        List<List<String>> out = new ArrayList<>();
        for (Beam b : consistent) {
            out.add(b.program());
        }
        return out;
    }

    private boolean consistent(List<String> program, int[][] inputs, int[][] outputs) {
        for (int e = 0; e < inputs.length; e++) {
            int[] xs = inputs[e];
            for (String op : program) {
                xs = xs == null ? null : semantics.apply(op, xs);
            }
            if (xs == null || !Arrays.equals(xs, outputs[e])) {
                return false;
            }
        }
        return true;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return model.parameters();
    }
}
