package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * DeepConf, "Deep think with confidence" (Fu et al., <a href="https://arxiv.org/abs/2508.15260">2508.15260</a>):
 * a test-time scaling strategy that spends samples like self-consistency but reads the
 * model's own token confidence to drop weak reasoning traces. A token's confidence is
 * {@code C_i = -(1/k) sum_{j <= k} log P_i(j)} over its top-{@code k} candidates (high when
 * the distribution is peaked); a sliding window of {@code n} tokens averages it into a
 * <em>group confidence</em>, and a trace is judged by its <em>lowest</em> group confidence,
 * the weakest stretch of its reasoning.
 * <ul>
 *   <li>{@link #offline}: sample {@code N} traces, keep the top {@code eta} share by lowest
 *       group confidence and take a confidence-weighted majority vote.</li>
 *   <li>{@link #online}: sample a few warm-up traces, set the stopping threshold at the
 *       confidence that keeps the top {@code eta} share of them, then generate the rest token
 *       by token and stop any trace as soon as its group confidence falls below the
 *       threshold. Stopped traces do not vote, and the tokens they would have cost are saved.</li>
 * </ul>
 * Next to {@link TestTimeCompute}'s self-consistency and verifier-ranked best-of-N, it needs
 * no verifier at all.
 */
public final class DeepConf {

    /**
     * One reasoning trace.
     *
     * @param tokens the generated tokens
     * @param confidences the confidence of each token
     * @param lowestGroupConfidence the lowest sliding-window mean confidence
     * @param stopped whether it was cut off early by the online threshold
     */
    public record Trace(int[] tokens, float[] confidences, double lowestGroupConfidence, boolean stopped) {
    }

    /**
     * A voting result.
     *
     * @param answer the winning answer, or {@code null}
     * @param tokens tokens generated, over all traces
     * @param traces traces generated
     * @param voters traces that voted
     * @param <T> the answer type
     */
    public record Vote<T>(T answer, long tokens, int traces, int voters) {
    }

    private final WindowLm model;
    private final int maxTokens;
    private final int[] stops;
    private double temperature = 1;
    private int topK = 5;
    private int window = 2;

    private DeepConf(WindowLm model, int maxTokens, int[] stops) {
        this.model = Objects.requireNonNull(model);
        this.maxTokens = maxTokens;
        this.stops = stops.clone();
    }

    /**
     * @param model the reasoner
     * @param maxTokens the most tokens per trace
     * @param stops tokens that end a trace
     * @return the strategy
     */
    public static DeepConf of(WindowLm model, int maxTokens, int... stops) {
        return new DeepConf(model, maxTokens, stops);
    }

    /**
     * @param t the sampling temperature
     * @param k the candidates per token in the confidence (paper: about 20; at most the vocabulary)
     * @param n the group window in tokens (paper: 2048 for long chains)
     * @return this strategy
     */
    public DeepConf settings(double t, int k, int n) {
        this.temperature = t;
        this.topK = k;
        this.window = n;
        return this;
    }

    /**
     * @param logits next-token logits
     * @param k the candidates
     * @return {@code -(1/k)} times the summed log-probabilities of the {@code k} most likely tokens
     */
    public static double tokenConfidence(float[] logits, int k) {
        double[] p = LlmTensors.softmax(logits, 1);
        double[] sorted = p.clone();
        Arrays.sort(sorted);
        double s = 0;
        int kk = Math.min(k, sorted.length);
        for (int j = 0; j < kk; j++) {
            s += Math.log(Math.max(sorted[sorted.length - 1 - j], 1e-30));
        }
        return -s / kk;
    }

    /**
     * @param confidences per-token confidences
     * @param n the window
     * @return the lowest mean over any {@code n} consecutive tokens (the mean of all of them
     *     when there are fewer)
     */
    public static double lowestGroupConfidence(float[] confidences, int n) {
        if (confidences.length == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        int w = Math.min(n, confidences.length);
        double best = Double.POSITIVE_INFINITY;
        for (int end = w; end <= confidences.length; end++) {
            double s = 0;
            for (int i = end - w; i < end; i++) {
                s += confidences[i];
            }
            best = Math.min(best, s / w);
        }
        return best;
    }

    /**
     * Samples traces in a batch, one network call per position, stopping a trace early when
     * its current group confidence drops below {@code threshold}.
     *
     * @param prompt the prompt
     * @param count how many traces
     * @param threshold the online stopping threshold, or {@link Double#NEGATIVE_INFINITY} for none
     * @param random the random source
     * @return the traces
     */
    public List<Trace> sample(int[] prompt, int count, double threshold, SplittableRandom random) {
        int vocab = model.vocabSize();
        int[][] seqs = new int[count][];
        List<List<Float>> conf = new ArrayList<>();
        boolean[] done = new boolean[count];
        boolean[] stopped = new boolean[count];
        for (int i = 0; i < count; i++) {
            seqs[i] = new int[0];
            conf.add(new ArrayList<>());
        }
        for (int t = 0; t < maxTokens; t++) {
            List<int[]> live = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                if (!done[i]) {
                    int[] ctx = Arrays.copyOf(prompt, prompt.length + seqs[i].length);
                    System.arraycopy(seqs[i], 0, ctx, prompt.length, seqs[i].length);
                    live.add(ctx);
                    who.add(i);
                }
            }
            if (live.isEmpty()) {
                break;
            }
            float[] logits = LlmTensors.values(model.logits(live));
            for (int j = 0; j < who.size(); j++) {
                int i = who.get(j);
                float[] row = Arrays.copyOfRange(logits, j * vocab, (j + 1) * vocab);
                int tok = LlmTensors.sample(row, temperature, random);
                seqs[i] = Arrays.copyOf(seqs[i], seqs[i].length + 1);
                seqs[i][seqs[i].length - 1] = tok;
                List<Float> c = conf.get(i);
                c.add((float) tokenConfidence(row, topK));
                for (int s : stops) {
                    done[i] |= tok == s;
                }
                if (!done[i] && c.size() >= window) {
                    double g = 0;
                    for (int q = c.size() - window; q < c.size(); q++) {
                        g += c.get(q);
                    }
                    if (g / window < threshold) {
                        done[i] = true;
                        stopped[i] = true;
                    }
                }
            }
        }
        List<Trace> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            float[] c = new float[conf.get(i).size()];
            for (int q = 0; q < c.length; q++) {
                c[q] = conf.get(i).get(q);
            }
            out.add(new Trace(seqs[i], c, lowestGroupConfidence(c, window), stopped[i]));
        }
        return out;
    }

    /**
     * The confidence-filtered, confidence-weighted vote over finished traces.
     *
     * @param traces the traces
     * @param extract a trace's answer, or {@code null}
     * @param keepTop the share of traces kept, by lowest group confidence (1 keeps all)
     * @param <T> the answer type
     * @return the answer with the largest confidence total, or {@code null}
     */
    public static <T> T vote(List<Trace> traces, Function<int[], T> extract, double keepTop) {
        List<Trace> finished = new ArrayList<>();
        for (Trace t : traces) {
            if (!t.stopped()) {
                finished.add(t);
            }
        }
        finished.sort((a, b) -> Double.compare(b.lowestGroupConfidence(), a.lowestGroupConfidence()));
        int keep = Math.max(1, (int) Math.ceil(keepTop * finished.size()));
        Map<T, Double> total = new HashMap<>();
        for (Trace t : finished.subList(0, Math.min(keep, finished.size()))) {
            T a = extract.apply(t.tokens());
            if (a != null) {
                total.merge(a, t.lowestGroupConfidence(), Double::sum);
            }
        }
        return total.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    private static long tokens(List<Trace> traces) {
        long n = 0;
        for (Trace t : traces) {
            n += t.tokens().length;
        }
        return n;
    }

    /**
     * DeepConf-offline: {@code n} full traces, filtered and weighted.
     *
     * @param prompt the prompt
     * @param n traces
     * @param keepTop the share kept (paper: 0.1 aggressive, 0.9 conservative)
     * @param extract a trace's answer
     * @param random the random source
     * @param <T> the answer type
     * @return the vote
     */
    public <T> Vote<T> offline(int[] prompt, int n, double keepTop, Function<int[], T> extract,
            SplittableRandom random) {
        List<Trace> traces = sample(prompt, n, Double.NEGATIVE_INFINITY, random);
        return new Vote<>(vote(traces, extract, keepTop), tokens(traces), n, (int) Math.ceil(keepTop * n));
    }

    /**
     * DeepConf-online: warm-up traces set the threshold, the rest stop early below it.
     *
     * @param prompt the prompt
     * @param n traces in total
     * @param warmup warm-up traces (paper: 16)
     * @param keepTop the share of warm-up traces the threshold keeps
     * @param extract a trace's answer
     * @param random the random source
     * @param <T> the answer type
     * @return the vote
     */
    public <T> Vote<T> online(int[] prompt, int n, int warmup, double keepTop, Function<int[], T> extract,
            SplittableRandom random) {
        List<Trace> warm = sample(prompt, warmup, Double.NEGATIVE_INFINITY, random);
        double[] lows = warm.stream().mapToDouble(Trace::lowestGroupConfidence).sorted().toArray();
        int idx = Math.max(0, Math.min(lows.length - 1, (int) Math.floor((1 - keepTop) * lows.length)));
        double threshold = lows[idx];
        List<Trace> all = new ArrayList<>(warm);
        all.addAll(sample(prompt, n - warmup, threshold, random));
        List<Trace> voters = new ArrayList<>();
        for (Trace t : all) {
            if (!t.stopped() && t.lowestGroupConfidence() >= threshold) {
                voters.add(t);
            }
        }
        T answer = voters.isEmpty() ? vote(all, extract, 1) : vote(voters, extract, 1);
        return new Vote<>(answer, tokens(all), n, voters.size());
    }
}
