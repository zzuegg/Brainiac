package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.MaskedDiffusion;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * X28: a masked-diffusion text model (LLaDA-style, see {@code MaskedDiffusion}) next to an
 * autoregressive model of the same size, both trained on the same tiny word corpus. The
 * diffusion model writes every position at once and commits to the most confident
 * letters first, so its trace shows a word crystallizing out of {@code ______}; the
 * autoregressive model writes left to right. Both are scored by how many of their
 * samples are real corpus words.
 */
public final class TextDiffusionVsAutoregressive {
    /** The character shown for a still-masked position. */
    public static final char MASK_CHAR = '_';
    private final List<String> corpus;
    private final String alphabet;
    private final int length;
    private final int mask;
    private final int bos;
    private final MaskedDiffusion diffusion;
    private final Model autoregressive;
    private final Optimizer diffusionOptimizer = Optimizer.adam(3e-3f);
    private final Optimizer arOptimizer = Optimizer.adam(3e-3f);
    private final SplittableRandom random;

    private TextDiffusionVsAutoregressive(List<String> corpus, int width, int layers, long seed) {
        this.corpus = List.copyOf(corpus);
        this.length = corpus.get(0).length();
        StringBuilder letters = new StringBuilder();
        for (String w : corpus) {
            if (w.length() != length) {
                throw new IllegalArgumentException("all words need the same length");
            }
            for (char c : w.toCharArray()) {
                if (letters.indexOf(String.valueOf(c)) < 0) {
                    letters.append(c);
                }
            }
        }
        this.alphabet = letters.toString();
        this.mask = alphabet.length();
        this.bos = alphabet.length() + 1;
        int vocab = alphabet.length() + 2;
        this.random = new SplittableRandom(seed);
        this.diffusion = MaskedDiffusion.of(model(length, vocab, width, layers, false, seed), vocab, mask);
        this.autoregressive = model(length, vocab, width, layers, true, seed);
    }

    /**
     * Creates both models.
     *
     * @param corpus words, all of the same length
     * @param width the transformer width
     * @param layers the number of transformer blocks
     * @param seed the seed
     * @return the experiment
     */
    public static TextDiffusionVsAutoregressive of(List<String> corpus, int width, int layers, long seed) {
        return new TextDiffusionVsAutoregressive(corpus, width, layers, seed);
    }

    private static Model model(int length, int vocab, int width, int layers, boolean causal, long seed) {
        Model.Builder b = Model.sequential().input(length).embedding(vocab, width).positionalEncoding();
        for (int i = 0; i < layers; i++) {
            b.add(TransformerBlock.of(2).causal(causal).rotary(100f));
        }
        return b.dense(vocab).seed(seed).build();
    }

    private int[][] batch(int n) {
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = encode(corpus.get(random.nextInt(corpus.size())));
        }
        return out;
    }

    int[] encode(String word) {
        int[] ids = new int[word.length()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = alphabet.indexOf(word.charAt(i));
        }
        return ids;
    }

    /**
     * Turns ids back into text, showing masks as {@link #MASK_CHAR}.
     *
     * @param ids token ids
     * @return the text
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(id < alphabet.length() ? alphabet.charAt(id) : MASK_CHAR);
        }
        return sb.toString();
    }

    /**
     * Trains both models for some steps on random corpus batches.
     *
     * @param steps the number of optimizer steps for each model
     * @param batch the batch size
     * @return {@code {final diffusion loss, final autoregressive loss}}
     */
    public float[] train(int steps, int batch) {
        float dl = 0;
        float al = 0;
        for (int s = 0; s < steps; s++) {
            Var loss = diffusion.loss(batch(batch), random);
            dl = loss.value().item();
            loss.backward();
            diffusionOptimizer.step(diffusion.model().parameters());

            int[][] words = batch(batch);
            float[] in = new float[batch * length];
            float[] target = new float[batch * length];
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < length; j++) {
                    in[i * length + j] = j == 0 ? bos : words[i][j - 1];
                    target[i * length + j] = words[i][j];
                }
            }
            Var logits = autoregressive.forward(Var.of(Tensor.of(Shape.of(batch, length), in)), true);
            Var ce = Functional.crossEntropy(logits.reshape(batch * length, mask + 2),
                    Tensor.of(Shape.of(batch * length), target));
            al = ce.value().item();
            ce.backward();
            arOptimizer.step(autoregressive.parameters());
        }
        return new float[] {dl, al};
    }

    /**
     * Samples words with the diffusion model.
     *
     * @param count the number of words
     * @param steps the unmasking steps
     * @param temperature the sampling temperature
     * @param seed the seed
     * @param trace receives the partially unmasked text after every step, one list entry per
     *     step holding all words; or {@code null}
     * @return the words
     */
    public List<String> sampleDiffusion(int count, int steps, float temperature, long seed, List<List<String>> trace) {
        List<int[][]> raw = trace == null ? null : new ArrayList<>();
        int[][] ids = diffusion.sample(count, length, steps, null, Sampling.temperature(temperature),
                new SplittableRandom(seed), raw);
        if (trace != null) {
            for (int[][] step : raw) {
                List<String> words = new ArrayList<>();
                for (int[] w : step) {
                    words.add(decode(w));
                }
                trace.add(words);
            }
        }
        List<String> out = new ArrayList<>();
        for (int[] w : ids) {
            out.add(decode(w));
        }
        return out;
    }

    /**
     * Samples words left to right with the autoregressive model.
     *
     * @param count the number of words
     * @param temperature the sampling temperature
     * @param seed the seed
     * @return the words
     */
    public List<String> sampleAutoregressive(int count, float temperature, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Sampling sampling = Sampling.temperature(temperature);
        int vocab = mask + 2;
        int[][] seq = new int[count][length + 1];
        for (int[] s : seq) {
            s[0] = bos;
        }
        for (int pos = 0; pos < length; pos++) {
            float[] in = new float[count * length];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < length; j++) {
                    in[i * length + j] = j <= pos ? seq[i][j] : bos;   // later inputs are not seen (causal)
                }
            }
            float[] logits = autoregressive.forward(Var.of(Tensor.of(Shape.of(count, length), in)), false).value()
                    .toArray();
            for (int i = 0; i < count; i++) {
                float[] row = java.util.Arrays.copyOfRange(logits, (i * length + pos) * vocab,
                        (i * length + pos + 1) * vocab);
                row[mask] = Float.NEGATIVE_INFINITY;
                row[bos] = Float.NEGATIVE_INFINITY;
                seq[i][pos + 1] = sampling.sample(row, new int[0], r);
            }
        }
        List<String> out = new ArrayList<>();
        for (int[] s : seq) {
            out.add(decode(java.util.Arrays.copyOfRange(s, 1, length + 1)));
        }
        return out;
    }

    /**
     * Scores samples.
     *
     * @param words generated words
     * @return the share that are corpus words
     */
    public double validShare(List<String> words) {
        Set<String> known = Set.copyOf(corpus);
        return words.stream().filter(known::contains).count() / (double) words.size();
    }
}
