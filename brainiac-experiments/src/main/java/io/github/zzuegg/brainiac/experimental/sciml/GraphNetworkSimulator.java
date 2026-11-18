package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A graph network simulator (Sanchez-Gonzalez et al., ICML 2020) for 2-D particles in a box.
 * Particles are nodes joined to every particle within a connectivity radius; an
 * encoder-processor-decoder GNN predicts each particle's acceleration, which a semi-implicit
 * Euler step integrates:
 *
 * <pre>
 * nodes: the last C velocities and the clipped distances to the four walls
 * edges: relative displacement / R and distance / R
 * e' = e + phi([e, h_sender, h_receiver]),  h' = h + psi([h, sum_receiving e'])   (M steps)
 * a = dec(h);  v_{t+1} = v_t + a;  x_{t+1} = x_t + v_{t+1}
 * </pre>
 *
 * <p>Because every function acts per node or per edge, one trained model runs on any number of
 * particles. {@link #loss} adds random-walk noise to the input positions (and corrects the
 * target so the next position stays exact), so the model learns to undo its own rollout
 * errors.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2002.09405">arXiv:2002.09405</a>.
 */
public final class GraphNetworkSimulator {

    private final int history;
    private final double radius;
    private final double scale;
    private final double accelerationScale;
    private final Var[] nodeEncoder;
    private final Var[] edgeEncoder;
    private final List<Var[][]> processor = new ArrayList<>();
    private final Var[] decoder;

    private GraphNetworkSimulator(int history, int width, int steps, double radius, double scale,
            double accelerationScale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.history = history;
        this.radius = radius;
        this.scale = scale;
        this.accelerationScale = accelerationScale;
        this.nodeEncoder = mlp(r, 2 * history + 4, width, width);
        this.edgeEncoder = mlp(r, 3, width, width);
        for (int i = 0; i < steps; i++) {
            processor.add(new Var[][] {mlp(r, 3 * width, width, width), mlp(r, 2 * width, width, width)});
        }
        this.decoder = mlp(r, width, width, 2);
    }

    private static Var[] mlp(SplittableRandom r, int in, int hidden, int out) {
        return new Var[] {
            Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r)),
            Var.param(Tensor.zeros(Shape.of(hidden))),
            Var.param(Tensor.normal(Shape.of(hidden, out), 0, (float) Math.sqrt(1.0 / hidden), r)),
            Var.param(Tensor.zeros(Shape.of(out)))};
    }

    private static Var apply(Var[] m, Var x) {
        Var h = x.matmul(m[0]).add(m[1]).relu();
        return h.matmul(m[2]).add(m[3]);
    }

    /**
     * Creates the simulator.
     *
     * @param history the velocities per node, {@code C}
     * @param width the hidden width
     * @param steps message-passing steps, {@code M}
     * @param radius the connectivity radius
     * @param scale velocities and accelerations are divided by it (a fixed normalisation)
     * @param seed the seed
     * @return the simulator
     */
    public static GraphNetworkSimulator of(int history, int width, int steps, double radius, double scale,
            long seed) {
        return new GraphNetworkSimulator(history, width, steps, radius, scale, scale, seed);
    }

    /**
     * Creates the simulator with separate velocity and acceleration normalisations (the
     * paper normalises both by dataset statistics; per-frame accelerations are usually much
     * smaller than velocities, so one shared scale leaves the regression target tiny).
     *
     * @param history the velocities per node, {@code C}
     * @param width the hidden width
     * @param steps message-passing steps, {@code M}
     * @param radius the connectivity radius
     * @param velocityScale per-frame displacements are divided by it
     * @param accelerationScale per-frame second differences are divided by it
     * @param seed the seed
     * @return the simulator
     */
    public static GraphNetworkSimulator of(int history, int width, int steps, double radius, double velocityScale,
            double accelerationScale, long seed) {
        return new GraphNetworkSimulator(history, width, steps, radius, velocityScale, accelerationScale, seed);
    }

    /** @return the position frames a prediction needs, {@code C + 1} */
    public int frames() {
        return history + 1;
    }

    /**
     * Predicted accelerations for a batch of systems (each its own graph, of any size).
     *
     * @param systems per system, positions of the last {@code C + 1} frames {@code [C+1][n][2]}
     * @return accelerations, all particles stacked in system order, {@code [sum n, 2]}
     */
    public Var accelerations(List<double[][][]> systems) {
        int total = 0;
        for (double[][][] s : systems) {
            total += s[0].length;
        }
        List<int[]> edges = new ArrayList<>();
        List<float[]> edgeFeatures = new ArrayList<>();
        float[] nodes = new float[total * (2 * history + 4)];
        int offset = 0;
        for (double[][][] s : systems) {
            int n = s[0].length;
            double[][] x = s[history];
            for (int i = 0; i < n; i++) {
                int base = (offset + i) * (2 * history + 4);
                for (int c = 0; c < history; c++) {
                    for (int k = 0; k < 2; k++) {
                        nodes[base + 2 * c + k] = (float) ((s[c + 1][i][k] - s[c][i][k]) / scale);
                    }
                }
                for (int k = 0; k < 2; k++) {
                    nodes[base + 2 * history + 2 * k] = (float) Math.clamp(x[i][k] / radius, -1, 1);
                    nodes[base + 2 * history + 2 * k + 1] = (float) Math.clamp((1 - x[i][k]) / radius, -1, 1);
                }
                for (int j = 0; j < n; j++) {
                    if (i == j) {
                        continue;
                    }
                    double dx = x[i][0] - x[j][0];
                    double dy = x[i][1] - x[j][1];
                    double d = Math.sqrt(dx * dx + dy * dy);
                    if (d < radius) {
                        edges.add(new int[] {offset + j, offset + i});   // sender j, receiver i
                        edgeFeatures.add(new float[] {(float) (dx / radius), (float) (dy / radius),
                            (float) (d / radius)});
                    }
                }
            }
            offset += n;
        }
        int e = edges.size();
        Var h = apply(nodeEncoder, Var.of(Tensor.of(Shape.of(total, 2 * history + 4), nodes)));
        if (e == 0) {
            return apply(decoder, h);
        }
        float[] ef = new float[e * 3];
        float[] sender = new float[e * total];
        float[] receiver = new float[e * total];
        for (int k = 0; k < e; k++) {
            System.arraycopy(edgeFeatures.get(k), 0, ef, 3 * k, 3);
            sender[k * total + edges.get(k)[0]] = 1;
            receiver[k * total + edges.get(k)[1]] = 1;
        }
        Var s = Var.of(Tensor.of(Shape.of(e, total), sender));
        Var rcv = Var.of(Tensor.of(Shape.of(e, total), receiver));
        Var rcvT = Var.of(Tensor.of(Shape.of(e, total), receiver).transpose());
        Var ev = apply(edgeEncoder, Var.of(Tensor.of(Shape.of(e, 3), ef)));
        for (Var[][] step : processor) {
            ev = ev.add(apply(step[0], Var.concat(1, ev, s.matmul(h), rcv.matmul(h))));
            h = h.add(apply(step[1], Var.concat(1, h, rcvT.matmul(ev))));
        }
        return apply(decoder, h);
    }

    /**
     * The training loss with random-walk input noise.
     *
     * @param systems per system, positions of {@code C + 2} consecutive frames
     *     {@code [C+2][n][2]}; the last frame is the target
     * @param noise the random-walk noise standard deviation per frame (in position units), 0
     *     for none
     * @param random the noise source
     * @return the mean squared error of normalised accelerations
     */
    public Var loss(List<double[][][]> systems, double noise, SplittableRandom random) {
        List<double[][][]> inputs = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        for (double[][][] s : systems) {
            int n = s[0].length;
            double[][][] in = new double[history + 1][n][2];
            double[][] walk = new double[n][2];
            for (int f = 0; f <= history; f++) {
                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < 2; k++) {
                        if (noise > 0 && f > 0) {
                            walk[i][k] += noise / Math.sqrt(history) * random.nextGaussian();
                        }
                        in[f][i][k] = s[f][i][k] + walk[i][k];
                    }
                }
            }
            inputs.add(in);
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 2; k++) {
                    // the acceleration that lands exactly on the true next position
                    double a = s[history + 1][i][k] - 2 * in[history][i][k] + in[history - 1][i][k];
                    targets.add((float) (a / accelerationScale));
                }
            }
        }
        float[] t = new float[targets.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = targets.get(i);
        }
        return accelerations(inputs).sub(Var.of(Tensor.of(Shape.of(t.length / 2, 2), t))).square().mean();
    }

    /**
     * Rolls one system forward.
     *
     * @param start the first {@code C + 1} frames {@code [C+1][n][2]}
     * @param steps the steps
     * @return the predicted positions {@code [steps][n][2]}
     */
    public double[][][] rollout(double[][][] start, int steps) {
        int n = start[0].length;
        List<double[][]> frames = new ArrayList<>();
        for (double[][] f : start) {
            frames.add(MechanicalSystems.copy(f));
        }
        double[][][] out = new double[steps][][];
        for (int t = 0; t < steps; t++) {
            double[][][] window = frames.subList(frames.size() - history - 1, frames.size()).toArray(new double[0][][]);
            float[] a = Var.noGrad(() -> accelerations(List.<double[][][]>of(window))).value().toArray();
            double[][] x = window[history];
            double[][] prev = window[history - 1];
            double[][] next = new double[n][2];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 2; k++) {
                    double v = x[i][k] - prev[i][k] + a[2 * i + k] * accelerationScale;
                    next[i][k] = x[i][k] + v;
                }
            }
            frames.add(next);
            out[t] = next;
        }
        return out;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(nodeEncoder));
        p.addAll(List.of(edgeEncoder));
        for (Var[][] s : processor) {
            p.addAll(List.of(s[0]));
            p.addAll(List.of(s[1]));
        }
        p.addAll(List.of(decoder));
        return p;
    }

    @Override
    public String toString() {
        return "GraphNetworkSimulator(C=" + history + ", M=" + processor.size() + ", R=" + radius + ")";
    }
}
