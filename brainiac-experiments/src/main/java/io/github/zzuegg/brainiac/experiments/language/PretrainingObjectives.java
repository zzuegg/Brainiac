package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FimTransform;
import io.github.zzuegg.brainiac.experimental.llm.PermutationLm;
import io.github.zzuegg.brainiac.experimental.llm.SpanCorruption;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 6 items 2.16 and 2.17: T5 span corruption and the XLNet permutation LM next to the
 * masked LM, ELECTRA and fill-in-the-middle, on the counting language of
 * {@link ElectraVsMaskedLm} (10 tokens, modulus 16, strides 1-5) at equal compute (parameters
 * times tokens processed per step, matched to the masked LM's budget). Every encoder is width 32
 * with two blocks:
 *
 * <ul>
 *   <li>{@link Objective#MASKED_LM} and {@link Objective#ELECTRA}: {@link ElectraVsMaskedLm}'s
 *       trainers;</li>
 *   <li>{@link Objective#SPAN_CORRUPTION}: {@link SpanCorruption} (one span of two tokens per
 *       sequence) into a {@link TransformerSeq2Seq} encoder-decoder;</li>
 *   <li>{@link Objective#PERMUTATION_LM}: {@link PermutationLm}, predicting the last 3 of 10
 *       positions of each random order;</li>
 *   <li>{@link Objective#FIM}: a causal decoder trained on {@link FimTransform}ed sequences.</li>
 * </ul>
 *
 * Reported: the stride-probe accuracy of {@link ElectraVsMaskedLm} (a linear probe on the frozen
 * features of the middle position, 100 labelled sequences), and infilling of a missing
 * two-token span (exact match). ELECTRA's infill uses its small generator.
 */
public final class PretrainingObjectives {

    /** The pre-training objective. */
    public enum Objective {
        /** BERT masked LM, 15% of tokens */
        MASKED_LM,
        /** replaced-token detection */
        ELECTRA,
        /** T5 span corruption, encoder-decoder */
        SPAN_CORRUPTION,
        /** XLNet two-stream permutation LM */
        PERMUTATION_LM,
        /** causal LM on fill-in-the-middle documents */
        FIM
    }

    private static final int L = ElectraVsMaskedLm.LENGTH;
    private static final int V = ElectraVsMaskedLm.VOCAB;
    private static final int[] STRIDES = {1, 2, 3, 4, 5};
    private static final int WIDTH = 32;

    private PretrainingObjectives() {
    }

    /**
     * A pre-trained model's two uses.
     *
     * @param features {@code [n][L]} sequences to {@code [n, L', width]} features and the index
     *     of the middle token in them
     * @param infill sequences and a span start to the predicted two tokens per sequence
     * @param steps the optimizer steps it was trained for
     */
    private record Pretrained(Function<int[][], Var> features, int middle, Infill infill, int steps) {
    }

    /** Fills a two-token span. */
    @FunctionalInterface
    private interface Infill {
        int[][] fill(int[][] tokens, int[] starts);
    }

    private static long size(List<Var> params) {
        long c = 0;
        for (Var p : params) {
            c += p.shape().size();
        }
        return c;
    }

    private static Function<int[][], Var> hooked(Model m, Function<int[][], Tensor> input) {
        int last = m.graph().size() - 2;
        return seqs -> {
            Var[] captured = new Var[1];
            m.forward(Var.of(input.apply(seqs)), false, (step, out) -> {
                if (step.index() == last) {
                    captured[0] = out;
                }
                return out;
            });
            return captured[0];
        };
    }

    private static Tensor ids(int[][] seqs) {
        float[] f = new float[seqs.length * seqs[0].length];
        for (int i = 0; i < seqs.length; i++) {
            for (int t = 0; t < seqs[i].length; t++) {
                f[i * seqs[i].length + t] = seqs[i][t];
            }
        }
        return Tensor.of(Shape.of(seqs.length, seqs[0].length), f);
    }

    private static int argmax(float[] v, int offset, int n) {
        int best = 0;
        for (int c = 1; c < n; c++) {
            if (v[offset + c] > v[offset + best]) {
                best = c;
            }
        }
        return best;
    }

    /** Infilling with a masked-LM style model: mask both positions, take each argmax. */
    private static Infill maskedInfill(Model m) {
        return (seqs, starts) -> {
            int[][] masked = new int[seqs.length][];
            for (int i = 0; i < seqs.length; i++) {
                masked[i] = seqs[i].clone();
                masked[i][starts[i]] = V;
                masked[i][starts[i] + 1] = V;
            }
            float[] logits = m.forward(Var.of(ids(masked))).value().toArray();
            int[][] out = new int[seqs.length][2];
            for (int i = 0; i < seqs.length; i++) {
                for (int k = 0; k < 2; k++) {
                    out[i][k] = argmax(logits, (i * L + starts[i] + k) * (V + 1), V);
                }
            }
            return out;
        };
    }

    /**
     * Pre-trains with one objective at the compute of {@code mlmSteps} masked-LM steps.
     *
     * @param o the objective
     * @param mlmSteps the masked-LM steps that set the budget
     * @param seed the seed
     * @return the model's uses
     */
    private static Pretrained pretrain(Objective o, int mlmSteps, long seed) {
        long mlmParams = ElectraVsMaskedLm.encoder(WIDTH, V + 1, seed).parameterCount();
        double budget = (double) mlmSteps * mlmParams * L;
        SplittableRandom r = new SplittableRandom(seed + 1);
        switch (o) {
            case MASKED_LM -> {
                Model m = ElectraVsMaskedLm.maskedLm(mlmSteps, seed);
                return new Pretrained(hooked(m, PretrainingObjectives::ids), L / 2, maskedInfill(m), mlmSteps);
            }
            case ELECTRA -> {
                Model probeSize = ElectraVsMaskedLm.encoder(8, V + 1, seed);
                long cost = (ElectraVsMaskedLm.encoder(WIDTH, 1, seed).parameterCount() + probeSize.parameterCount()) * L;
                int steps = (int) Math.round(budget / cost);
                Model[] e = ElectraVsMaskedLm.electra(steps, seed);
                return new Pretrained(hooked(e[0], PretrainingObjectives::ids), L / 2, maskedInfill(e[1]), steps);
            }
            case SPAN_CORRUPTION -> {
                SpanCorruption sc = SpanCorruption.of(V, 1, 2);
                TransformerSeq2Seq t5 = TransformerSeq2Seq.of(V + 1, V + 1, WIDTH, 2, 2, seed);
                long enc = size(t5.encoderParameters());
                long cost = enc * sc.inputLength(L) + (size(t5.parameters()) - enc) * sc.targetLength();
                int steps = (int) Math.round(budget / cost);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    int[][] batch = CountingTasks.sequences(r, 32, L, V, STRIDES);
                    int[][] in = new int[32][];
                    int[][] out = new int[32][];
                    for (int i = 0; i < 32; i++) {
                        SpanCorruption.Example ex = sc.apply(batch[i], r);
                        in[i] = ex.input();
                        out[i] = ex.target();
                    }
                    t5.loss(in, out).backward();
                    opt.step(t5.parameters());
                }
                Infill fill = (seqs, starts) -> {
                    int[][] in = new int[seqs.length][];
                    for (int i = 0; i < seqs.length; i++) {
                        in[i] = sc.apply(seqs[i], new int[] {starts[i]}).input();
                    }
                    int[][] g = t5.greedy(in, 3);
                    int[][] out = new int[seqs.length][];
                    for (int i = 0; i < seqs.length; i++) {
                        out[i] = new int[] {g[i][1], g[i][2]};
                    }
                    return out;
                };
                return new Pretrained(seqs -> t5.encode(seqs, false), L / 2, fill, steps);
            }
            case PERMUTATION_LM -> {
                PermutationLm x = PermutationLm.of(V, L, WIDTH, 2, 2, seed);
                long cost = 2 * x.parameterCount() * L;   // two streams
                int steps = (int) Math.round(budget / cost);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    x.loss(CountingTasks.sequences(r, 32, L, V, STRIDES), 3, r).backward();
                    opt.step(x.parameters());
                }
                Infill fill = (seqs, starts) -> {
                    boolean[][] missing = new boolean[seqs.length][L];
                    for (int i = 0; i < seqs.length; i++) {
                        missing[i][starts[i]] = true;
                        missing[i][starts[i] + 1] = true;
                    }
                    int[][] done = x.infill(seqs, missing);
                    int[][] out = new int[seqs.length][];
                    for (int i = 0; i < seqs.length; i++) {
                        out[i] = new int[] {done[i][starts[i]], done[i][starts[i] + 1]};
                    }
                    return out;
                };
                return new Pretrained(x::features, L / 2, fill, steps);
            }
            case FIM -> {
                int len = L + FimTransform.overhead();
                FimTransform fim = FimTransform.of(V, V + 1, V + 2, FimTransform.Mode.PSM);
                Model m = Model.sequential().input(len).embedding(V + 3, WIDTH).positionalEncoding()
                        .add(TransformerBlock.of(2).causal(true).rotary(100f))
                        .add(TransformerBlock.of(2).causal(true).rotary(100f)).dense(V + 3).seed(seed).build();
                long cost = m.parameterCount() * len;
                int steps = (int) Math.round(budget / cost);
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    int[][] batch = CountingTasks.sequences(r, 32, L, V, STRIDES);
                    float[] x = new float[32 * len];
                    float[] y = new float[32 * len];
                    for (int i = 0; i < 32; i++) {
                        int[] d = fim.apply(batch[i], r);
                        for (int t = 0; t < len; t++) {
                            x[i * len + t] = d[t];
                            y[i * len + t] = t + 1 < len ? d[t + 1] : -1;
                        }
                    }
                    Var logits = m.forward(Var.of(Tensor.of(Shape.of(32, len), x)), true).reshape(32 * len, V + 3);
                    Functional.crossEntropy(logits, Tensor.of(Shape.of(32 * len), y)).backward();
                    opt.step(m.parameters());
                }
                Function<int[][], Tensor> whole = seqs -> {
                    int[][] d = new int[seqs.length][];
                    for (int i = 0; i < seqs.length; i++) {
                        d[i] = fim.apply(seqs[i], L, L);
                    }
                    return ids(d);
                };
                Infill fill = (seqs, starts) -> {
                    int[][] prompts = new int[seqs.length][len];
                    int[] cursor = new int[seqs.length];
                    for (int i = 0; i < seqs.length; i++) {
                        int[] p = fim.infillPrompt(java.util.Arrays.copyOfRange(seqs[i], 0, starts[i]),
                                java.util.Arrays.copyOfRange(seqs[i], starts[i] + 2, L));
                        System.arraycopy(p, 0, prompts[i], 0, p.length);
                        cursor[i] = p.length;
                    }
                    int[][] out = new int[seqs.length][2];
                    for (int k = 0; k < 2; k++) {
                        float[] logits = m.forward(Var.of(ids(prompts))).value().toArray();
                        for (int i = 0; i < seqs.length; i++) {
                            int tok = argmax(logits, (i * len + cursor[i] - 1) * (V + 3), V);
                            out[i][k] = tok;
                            prompts[i][cursor[i]++] = tok;
                        }
                    }
                    return out;
                };
                return new Pretrained(hooked(m, whole), 1 + L / 2, fill, steps);
            }
            default -> throw new IllegalArgumentException(o.toString());
        }
    }

    /**
     * Pre-trains and evaluates one objective.
     *
     * @param o the objective
     * @param mlmSteps the masked-LM steps (of 32 sequences) that set the compute budget
     * @param seed the seed
     * @return {@code {stride-probe accuracy, two-token infill exact match, steps trained}}
     */
    public static double[] run(Objective o, int mlmSteps, long seed) {
        Pretrained p = pretrain(o, mlmSteps, seed);
        double probe = probe(p, seed + 5);
        SplittableRandom r = new SplittableRandom(seed + 9);
        int n = 300;
        int[][] seqs = CountingTasks.sequences(r, n, L, V, STRIDES);
        int[] starts = new int[n];
        for (int i = 0; i < n; i++) {
            starts[i] = 1 + r.nextInt(L - 3);
        }
        int[][] fill = p.infill().fill(seqs, starts);
        int hit = 0;
        for (int i = 0; i < n; i++) {
            hit += fill[i][0] == seqs[i][starts[i]] && fill[i][1] == seqs[i][starts[i] + 1] ? 1 : 0;
        }
        return new double[] {probe, (double) hit / n, p.steps()};
    }

    private static Tensor middle(Pretrained p, int[][] seqs) {
        Var f = p.features().apply(seqs);
        return f.value().slice(1, p.middle(), p.middle() + 1).reshape(seqs.length, f.shape().dim(2));
    }

    /** {@link ElectraVsMaskedLm#probe}'s protocol on any feature function. */
    private static double probe(Pretrained p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 100;
        int[][] train = new int[n][];
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            int k = i % STRIDES.length;
            train[i] = CountingTasks.sequences(r, 1, L, V, STRIDES[k])[0];
            labels[i] = k;
        }
        Tensor x = middle(p, train);
        Model probe = Model.sequential().input(x.shape().dim(1)).dense(STRIDES.length).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor mean = x.mean(0, true);
        Tensor std = x.sub(mean).square().mean(0, true).add(1e-6f).sqrt();
        Tensor xs = x.sub(mean).div(std);
        for (int s = 0; s < 300; s++) {
            Functional.crossEntropy(probe.forward(Var.of(xs), true), Tensor.of(Shape.of(n), labels)).backward();
            opt.step(probe.parameters());
        }
        int test = 500;
        int[][] seqs = new int[test][];
        float[] truth = new float[test];
        for (int i = 0; i < test; i++) {
            int k = r.nextInt(STRIDES.length);
            truth[i] = k;
            seqs[i] = CountingTasks.sequences(r, 1, L, V, STRIDES[k])[0];
        }
        Tensor out = probe.forward(Var.of(middle(p, seqs).sub(mean).div(std))).value();
        return out.argmax(1).equalTo(Tensor.of(Shape.of(test), truth)).mean().item();
    }
}
