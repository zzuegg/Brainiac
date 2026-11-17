package io.github.zzuegg.brainiac.experimental.nn;

import java.util.ArrayList;
import java.util.List;

/**
 * A trained stack of {@link LogicGateLayer}s compiled to a pure boolean circuit: each gate
 * becomes one or two bit operations on {@code long}s, so one pass evaluates 64 samples
 * at once with no arithmetic at all, the end point of quantization that differentiable
 * logic gate networks are after. {@link #groupSum} turns output bits into class scores
 * the same way the training head does (popcount per class group).
 */
public final class LogicCircuit {
    private final List<int[]> gates = new ArrayList<>();
    private final List<int[]> left = new ArrayList<>();
    private final List<int[]> right = new ArrayList<>();
    private final int inputs;

    private LogicCircuit(List<LogicGateLayer> layers) {
        this.inputs = layers.get(0).inputs();
        for (LogicGateLayer l : layers) {
            gates.add(l.gates());
            left.add(l.leftInputs());
            right.add(l.rightInputs());
        }
    }

    /**
     * Compiles layers, taking each gate's most likely function.
     *
     * @param layers the layers, input to output
     * @return the circuit
     */
    public static LogicCircuit compile(List<LogicGateLayer> layers) {
        return new LogicCircuit(layers);
    }

    /**
     * Applies one gate to 64 lanes of bits.
     *
     * @param gate the gate id 0..15 (see {@link LogicGateLayer#NAMES})
     * @param a the left inputs
     * @param b the right inputs
     * @return the outputs
     */
    public static long gate(int gate, long a, long b) {
        return switch (gate) {
            case 0 -> 0L;
            case 1 -> a & b;
            case 2 -> a & ~b;
            case 3 -> a;
            case 4 -> ~a & b;
            case 5 -> b;
            case 6 -> a ^ b;
            case 7 -> a | b;
            case 8 -> ~(a | b);
            case 9 -> ~(a ^ b);
            case 10 -> ~b;
            case 11 -> a | ~b;
            case 12 -> ~a;
            case 13 -> ~a | b;
            case 14 -> ~(a & b);
            case 15 -> -1L;
            default -> throw new IllegalArgumentException("gate id " + gate);
        };
    }

    /**
     * Evaluates the circuit on bit-sliced inputs.
     *
     * @param in {@code in[i]} holds input bit {@code i} of 64 samples (sample k in bit k)
     * @return {@code out[j]} holds output bit {@code j} of the same 64 samples
     */
    public long[] evaluate(long[] in) {
        if (in.length != inputs) {
            throw new IllegalArgumentException("need " + inputs + " input words, got " + in.length);
        }
        long[] x = in;
        for (int l = 0; l < gates.size(); l++) {
            int[] g = gates.get(l);
            int[] a = left.get(l);
            int[] b = right.get(l);
            long[] y = new long[g.length];
            for (int i = 0; i < g.length; i++) {
                y[i] = gate(g[i], x[a[i]], x[b[i]]);
            }
            x = y;
        }
        return x;
    }

    /**
     * Class scores of up to 64 samples: the number of set output bits in each of
     * {@code classes} equal groups.
     *
     * @param out the circuit outputs
     * @param classes the number of classes
     * @param samples how many of the 64 lanes are in use
     * @return {@code [samples][classes]} counts
     */
    public static int[][] groupSum(long[] out, int classes, int samples) {
        int per = out.length / classes;
        int[][] scores = new int[samples][classes];
        for (int c = 0; c < classes; c++) {
            for (int j = c * per; j < (c + 1) * per; j++) {
                long w = out[j];
                for (int s = 0; s < samples; s++) {
                    scores[s][c] += (int) (w >>> s & 1L);
                }
            }
        }
        return scores;
    }

    /** @return the total number of gates */
    public int size() {
        return gates.stream().mapToInt(g -> g.length).sum();
    }
}
