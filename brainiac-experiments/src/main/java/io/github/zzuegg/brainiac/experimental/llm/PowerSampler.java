package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Training-free sampling from the <em>power distribution</em> {@code p(x)^alpha} of a base model
 * over whole completions (Karan and Du, "Reasoning with sampling: your language model is secretly
 * a reasoner", <a href="https://arxiv.org/abs/2510.14901">arXiv:2510.14901</a>). Low-temperature
 * sampling sharpens each next-token distribution on its own, which is not the same thing: the
 * power distribution also favours tokens whose <em>futures</em> are likely, the kind of
 * look-ahead that RL post-training is said to add.
 *
 * <p>The sampler runs a Metropolis-Hastings chain per prompt. It starts from a completion drawn
 * at temperature {@code 1/alpha}; each step picks a position uniformly, resamples the suffix from
 * there at temperature {@code 1/alpha}, and accepts with
 *
 * <pre>
 * A = min(1, [p(x')^alpha q(x | x')] / [p(x)^alpha q(x' | x)])
 * </pre>
 *
 * <p>where {@code q} is the suffix proposal (the position choice included). All chains of a batch
 * move together, so one network call serves every chain at each generated position.
 */
public final class PowerSampler {
    private final WindowLm lm;
    private final double alpha;
    private final int steps;
    private final int maxTokens;
    private final int eos;
    private long generated;
    private long proposals;
    private long accepted;

    private PowerSampler(WindowLm lm, double alpha, int steps, int maxTokens, int eos) {
        this.lm = lm;
        this.alpha = alpha;
        this.steps = steps;
        this.maxTokens = maxTokens;
        this.eos = eos;
    }

    /**
     * Creates a sampler.
     *
     * @param lm the base model, left unchanged
     * @param alpha the power, above 1 to sharpen
     * @param steps Metropolis-Hastings steps per chain
     * @param maxTokens the longest completion
     * @param eos the end token
     * @return the sampler
     */
    public static PowerSampler of(WindowLm lm, double alpha, int steps, int maxTokens, int eos) {
        if (alpha <= 0 || steps < 0 || maxTokens < 1) {
            throw new IllegalArgumentException("need alpha > 0, steps >= 0 and maxTokens >= 1");
        }
        return new PowerSampler(lm, alpha, steps, maxTokens, eos);
    }

    /** A completion with its per-token log-probabilities under the base model and the proposal. */
    private record Path(int[] tokens, double[] logP, double[] logQ) {
        double sumP(int from) {
            double s = 0;
            for (int i = from; i < logP.length; i++) {
                s += logP[i];
            }
            return s;
        }

        double sumQ(int from) {
            double s = 0;
            for (int i = from; i < logQ.length; i++) {
                s += logQ[i];
            }
            return s;
        }
    }

    /**
     * Draws one completion per prompt from the power distribution (approximately, after the
     * configured number of steps).
     *
     * @param prompts the prompts; repeat a prompt for several independent samples
     * @param random the random source
     * @return the completions, each including its end token if one was produced
     */
    public int[][] sample(int[][] prompts, SplittableRandom random) {
        int n = prompts.length;
        int[] cut = new int[n];
        Path[] current = extend(prompts, new Path[n], cut, random);
        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < n; i++) {
                cut[i] = random.nextInt(current[i].tokens().length);
            }
            Path[] proposal = extend(prompts, current, cut, random);
            for (int i = 0; i < n; i++) {
                Path x = current[i];
                Path y = proposal[i];
                double logA = alpha * (y.sumP(cut[i]) - x.sumP(cut[i]))
                        + (x.sumQ(cut[i]) - Math.log(y.tokens().length))
                        - (y.sumQ(cut[i]) - Math.log(x.tokens().length));
                proposals++;
                if (Math.log(random.nextDouble()) < logA) {
                    current[i] = y;
                    accepted++;
                }
            }
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = current[i].tokens();
        }
        return out;
    }

    /**
     * Keeps the first {@code cut[i]} tokens of each path and samples the rest at temperature
     * {@code 1/alpha}, recording both log-probabilities of every token.
     */
    private Path[] extend(int[][] prompts, Path[] from, int[] cut, SplittableRandom random) {
        int n = prompts.length;
        int[][] tokens = new int[n][maxTokens];
        double[][] logP = new double[n][maxTokens];
        double[][] logQ = new double[n][maxTokens];
        int[] length = new int[n];
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (from[i] != null) {
                length[i] = cut[i];
                System.arraycopy(from[i].tokens(), 0, tokens[i], 0, cut[i]);
                System.arraycopy(from[i].logP(), 0, logP[i], 0, cut[i]);
                System.arraycopy(from[i].logQ(), 0, logQ[i], 0, cut[i]);
            }
        }
        while (true) {
            List<int[]> prefixes = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i] && length[i] < maxTokens) {
                    int[] p = Arrays.copyOf(prompts[i], prompts[i].length + length[i]);
                    System.arraycopy(tokens[i], 0, p, prompts[i].length, length[i]);
                    prefixes.add(p);
                    who.add(i);
                }
            }
            if (prefixes.isEmpty()) {
                break;
            }
            float[] logits = LlmTensors.values(lm.logits(prefixes));
            int vocab = lm.vocabSize();
            for (int j = 0; j < who.size(); j++) {
                int i = who.get(j);
                float[] row = Arrays.copyOfRange(logits, j * vocab, (j + 1) * vocab);
                double[] base = LogitMath.logSoftmax(row);
                float[] sharp = new float[vocab];
                for (int v = 0; v < vocab; v++) {
                    sharp[v] = (float) (row[v] * alpha);
                }
                double[] prop = LogitMath.logSoftmax(sharp);
                double[] probs = new double[vocab];
                for (int v = 0; v < vocab; v++) {
                    probs[v] = Math.exp(prop[v]);
                }
                int tok = LlmTensors.sample(probs, random);
                tokens[i][length[i]] = tok;
                logP[i][length[i]] = base[tok];
                logQ[i][length[i]] = prop[tok];
                length[i]++;
                generated++;
                if (tok == eos) {
                    done[i] = true;
                }
            }
        }
        Path[] out = new Path[n];
        for (int i = 0; i < n; i++) {
            out[i] = new Path(Arrays.copyOf(tokens[i], length[i]), Arrays.copyOf(logP[i], length[i]),
                    Arrays.copyOf(logQ[i], length[i]));
        }
        return out;
    }

    /** @return tokens generated so far, the initial draws and every proposal included */
    public long generatedTokens() {
        return generated;
    }

    /** @return the share of accepted Metropolis-Hastings proposals so far */
    public double acceptanceRate() {
        return proposals == 0 ? 0 : (double) accepted / proposals;
    }

    @Override
    public String toString() {
        return "PowerSampler(alpha=" + alpha + ", steps=" + steps + ")";
    }
}
