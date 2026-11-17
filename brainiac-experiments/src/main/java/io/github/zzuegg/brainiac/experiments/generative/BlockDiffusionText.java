package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.llm.BlockDiffusionLm;
import io.github.zzuegg.brainiac.experimental.llm.MaskedDiffusion;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Block diffusion against its two limits on variable-length text, as in
 * {@link TextDiffusionVsAutoregressive} but with sentences of a tiny grammar ("owl ate big
 * rat") of 11 or 15 characters, padded with an end token: {@link BlockDiffusionLm} with
 * blocks of 4 (BD3-LM, <a href="https://arxiv.org/abs/2503.09573">arXiv:2503.09573</a>), the
 * same model with blocks of 1 (exactly autoregressive) and a {@link MaskedDiffusion} over the
 * whole 16 positions. Reported: held-out perplexity from the NELBO (exact for the
 * autoregressive model), tokens generated per denoiser pass, and the share of samples that
 * are grammatical.
 */
public final class BlockDiffusionText {
    private static final int LENGTH = 16;
    private static final String ALPHABET = " abcdefghijklmnopqrstuvwxyz";
    private static final int EOS = ALPHABET.length();
    private static final int MASK = EOS + 1;
    private static final int VOCAB = MASK + 1;

    private BlockDiffusionText() {
    }

    /**
     * One model's outcome.
     *
     * @param perplexity held-out perplexity bound per token
     * @param tokensPerStep generated tokens per denoiser forward pass
     * @param valid share of samples that are sentences of the grammar
     */
    public record Outcome(double perplexity, double tokensPerStep, double valid) {
    }

    /** @return every sentence of the grammar */
    public static List<String> sentences() {
        List<String> out = new ArrayList<>();
        for (String s : new String[] {"cat", "dog", "owl"}) {
            for (String v : new String[] {"saw", "ate"}) {
                for (String a : new String[] {"", "big ", "odd "}) {
                    for (String o : new String[] {"eel", "rat", "bug"}) {
                        out.add(s + " " + v + " " + a + o);
                    }
                }
            }
        }
        return out;
    }

    private static int[] encode(String s) {
        int[] ids = new int[LENGTH];
        Arrays.fill(ids, EOS);
        for (int i = 0; i < s.length(); i++) {
            ids[i] = ALPHABET.indexOf(s.charAt(i));
        }
        return ids;
    }

    private static String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id >= EOS) {
                break;
            }
            sb.append(ALPHABET.charAt(id));
        }
        return sb.toString();
    }

    private static int[][] batch(List<String> corpus, int n, SplittableRandom r) {
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = encode(corpus.get(r.nextInt(corpus.size())));
        }
        return out;
    }

    /**
     * Trains the three models on the same batches and measures them.
     *
     * @param steps training steps
     * @param seed the seed
     * @return outcome per model
     */
    public static Map<String, Outcome> run(int steps, long seed) {
        List<String> corpus = sentences();
        Set<String> known = Set.copyOf(corpus);
        BlockDiffusionLm ar = BlockDiffusionLm.of(VOCAB, MASK, EOS, LENGTH, 1, 32, 2, 2, seed);
        BlockDiffusionLm bd3 = BlockDiffusionLm.of(VOCAB, MASK, EOS, LENGTH, 4, 32, 2, 2, seed);
        Model.Builder b = Model.sequential().input(LENGTH).embedding(VOCAB, 32).positionalEncoding();
        for (int i = 0; i < 2; i++) {
            b.add(TransformerBlock.of(2).causal(false));
        }
        MaskedDiffusion mdlm = MaskedDiffusion.of(b.dense(VOCAB).seed(seed).build(), VOCAB, MASK);
        Optimizer arOpt = Optimizer.adam(3e-3f);
        Optimizer bdOpt = Optimizer.adam(3e-3f);
        Optimizer mdOpt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][] x = batch(corpus, 16, r);
            ar.loss(x, r).backward();
            arOpt.step(ar.parameters());
            bd3.loss(x, r).backward();
            bdOpt.step(bd3.parameters());
            mdlm.loss(x, r).backward();
            mdOpt.step(mdlm.model().parameters());
        }
        int[][] held = batch(corpus, 64, new SplittableRandom(seed + 9));
        Sampling sampling = Sampling.temperature(0.7f);
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Object[] m : new Object[][] {{"autoregressive (B=1)", ar, 1}, {"BD3-LM (B=4)", bd3, 2}}) {
            BlockDiffusionLm model = (BlockDiffusionLm) m[1];
            double nll = model.nelbo(held, 4, new SplittableRandom(seed + 3));
            BlockDiffusionLm.Samples smp = model.sample(32, (Integer) m[2], sampling, new SplittableRandom(seed + 4));
            long ok = smp.tokens().stream().filter(t -> known.contains(decode(t))).count();
            out.put((String) m[0], new Outcome(Math.exp(nll), smp.tokensPerStep(), ok / 32.0));
        }
        double nll = 0;
        SplittableRandom nr = new SplittableRandom(seed + 3);
        for (int d = 0; d < 4; d++) {
            nll += mdlm.loss(held, nr).value().item() / 4;
        }
        int samplingSteps = 8;
        int[][] smp = mdlm.sample(32, LENGTH, samplingSteps, null, sampling, new SplittableRandom(seed + 4), null);
        long ok = 0;
        long tokens = 0;
        for (int[] t : smp) {
            String text = decode(t);
            ok += known.contains(text) ? 1 : 0;
            tokens += Math.min(LENGTH, text.length() + 1);
        }
        out.put("masked diffusion (B=16)", new Outcome(Math.exp(nll), (double) tokens / (32 * samplingSteps),
                ok / 32.0));
        return out;
    }
}
