package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DiscreteFlowMatching;
import io.github.zzuegg.brainiac.experimental.nn.BayesianFlowNetwork;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Discrete and binary generation beyond masked diffusion (wave 6, items 4.13 and 4.14).
 *
 * <p><b>Words</b> ({@link #words}): on the {@link TextDiffusionVsAutoregressive} corpus and
 * transformer, {@link DiscreteFlowMatching} with a mask source (with and without its
 * corrector) and with a uniform source, and a discrete {@link BayesianFlowNetwork} (an MLP
 * on the six letter posteriors), against the masked-diffusion (LLaDA sampler) and
 * autoregressive models of {@link TextDiffusionVsAutoregressive}, all trained for the same
 * steps. Score: the share of samples that are corpus words, at a fixed number of sampling
 * steps.
 *
 * <p><b>Binary images</b> ({@link #images}): {@link ToyImages} pixels as 64 binary variables
 * for the discrete BFN, and as reals in [-1, 1] for the continuous BFN and for DDPM / DDIM
 * ({@link TinyDdpm}), scored by {@link GenerativeScores#images} at 10 and 50 steps.
 */
@SuppressWarnings("exports")
public final class DiscreteGenerators {
    private DiscreteGenerators() {
    }

    private static Model transformer(int length, int vocab, long seed) {
        Model.Builder b = Model.sequential().input(length).embedding(vocab, 32).positionalEncoding();
        for (int i = 0; i < 2; i++) {
            b.add(TransformerBlock.of(2).causal(false).rotary(100f));
        }
        return b.dense(vocab).seed(seed).build();
    }

    /**
     * Trains every word model and samples it.
     *
     * @param corpus words of one length
     * @param iterations training steps of 32 words for every model
     * @param steps sampling step counts for the non-autoregressive models
     * @param seed the seed
     * @return per method and step count ({@code "<method> <steps>"}), the share of 64 samples
     *     that are corpus words; {@code "autoregressive"} once
     */
    public static Map<String, Double> words(List<String> corpus, int iterations, int[] steps, long seed) {
        int length = corpus.get(0).length();
        StringBuilder letters = new StringBuilder();
        for (String w : corpus) {
            for (char c : w.toCharArray()) {
                if (letters.indexOf(String.valueOf(c)) < 0) {
                    letters.append(c);
                }
            }
        }
        String alphabet = letters.toString();
        int mask = alphabet.length();
        int vocab = mask + 1;
        int[][] encoded = new int[corpus.size()][length];
        for (int w = 0; w < corpus.size(); w++) {
            for (int j = 0; j < length; j++) {
                encoded[w][j] = alphabet.indexOf(corpus.get(w).charAt(j));
            }
        }
        TextDiffusionVsAutoregressive baseline = TextDiffusionVsAutoregressive.of(corpus, 32, 2, seed);
        baseline.train(iterations, 32);
        DiscreteFlowMatching dfmMask = DiscreteFlowMatching.of(transformer(length, vocab, seed), vocab, mask,
                DiscreteFlowMatching.Source.MASK, 1f);
        DiscreteFlowMatching dfmUniform = DiscreteFlowMatching.of(transformer(length, vocab, seed), vocab, mask,
                DiscreteFlowMatching.Source.UNIFORM, 1f);
        BayesianFlowNetwork bfn = BayesianFlowNetwork.discrete(length, alphabet.length(), 128, 3f, seed);
        Optimizer maskOpt = Optimizer.adam(3e-3f);
        Optimizer uniformOpt = Optimizer.adam(3e-3f);
        Optimizer bfnOpt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            int[][] batch = new int[32][];
            for (int b = 0; b < batch.length; b++) {
                batch[b] = encoded[r.nextInt(encoded.length)];
            }
            dfmMask.loss(batch, r).backward();
            maskOpt.step(dfmMask.model().parameters());
            dfmUniform.loss(batch, r).backward();
            uniformOpt.step(dfmUniform.model().parameters());
            bfn.discreteLoss(batch, r).backward();
            Nets.clip(bfn.parameters(), 5f);
            bfnOpt.step(bfn.parameters());
        }
        int n = 64;
        float temperature = 0.7f;
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("autoregressive", baseline.validShare(baseline.sampleAutoregressive(n, temperature, seed + 1)));
        for (int k : steps) {
            out.put("masked diffusion " + k, baseline.validShare(baseline.sampleDiffusion(n, k, temperature, seed + 1,
                    null)));
            out.put("DFM mask " + k, share(baseline, alphabet, dfmMask.sample(n, length, k, 0, temperature,
                    new SplittableRandom(seed + 1), null)));
            out.put("DFM mask + corrector " + k, share(baseline, alphabet, dfmMask.sample(n, length, k, 1f,
                    temperature, new SplittableRandom(seed + 1), null)));
            out.put("DFM uniform " + k, share(baseline, alphabet, dfmUniform.sample(n, length, k, 0, temperature,
                    new SplittableRandom(seed + 1), null)));
            out.put("DFM uniform + corrector " + k, share(baseline, alphabet, dfmUniform.sample(n, length, k, 1f,
                    temperature, new SplittableRandom(seed + 1), null)));
            out.put("BFN " + k, share(baseline, alphabet, bfn.sampleDiscrete(n, k, new SplittableRandom(seed + 1))));
        }
        return out;
    }

    private static double share(TextDiffusionVsAutoregressive scorer, String alphabet, int[][] ids) {
        List<String> words = new ArrayList<>();
        for (int[] w : ids) {
            StringBuilder sb = new StringBuilder();
            for (int id : w) {
                sb.append(id < alphabet.length() ? alphabet.charAt(id) : '_');
            }
            words.add(sb.toString());
        }
        return scorer.validShare(words);
    }

    /**
     * Trains the image models and samples them.
     *
     * @param iterations training steps of 64 images
     * @param steps the sampling steps to compare
     * @param seed the seed
     * @return per method and step count, the image score
     */
    public static Map<String, GenerativeScores.ImageScore> images(int iterations, int[] steps, long seed) {
        BayesianFlowNetwork binary = BayesianFlowNetwork.discrete(ToyImages.PIXELS, 2, 256, 3f, seed);
        BayesianFlowNetwork real = BayesianFlowNetwork.continuous(ToyImages.PIXELS, 256, 0.02f, seed);
        TinyDdpm ddpm = TinyDdpm.of(MlpDenoiser.of(ToyImages.PIXELS, 0, 128, 2, seed).timeScale(1f / 100), 100, true,
                3e-3f, 0, seed);
        Optimizer binOpt = Optimizer.adam(2e-3f);
        Optimizer realOpt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int[] none = new int[64];
        Arrays.fill(none, -1);
        for (int i = 0; i < iterations; i++) {
            float[][] x = PointClouds.images(64, r);
            int[][] bits = new int[x.length][ToyImages.PIXELS];
            for (int b = 0; b < x.length; b++) {
                for (int p = 0; p < ToyImages.PIXELS; p++) {
                    bits[b][p] = x[b][p] > 0 ? 1 : 0;
                }
            }
            binary.discreteLoss(bits, r).backward();
            Nets.clip(binary.parameters(), 5f);
            binOpt.step(binary.parameters());
            Var cl = real.continuousLoss(Nets.matrix(x), r);
            cl.backward();
            Nets.clip(real.parameters(), 5f);
            realOpt.step(real.parameters());
            ddpm.trainStep(x, none);
        }
        Map<String, GenerativeScores.ImageScore> out = new LinkedHashMap<>();
        int n = 128;
        int[] sampleNone = new int[n];
        Arrays.fill(sampleNone, -1);
        for (int s : steps) {
            int[][] bits = binary.sampleDiscrete(n, s, new SplittableRandom(seed + s));
            float[][] img = new float[n][ToyImages.PIXELS];
            for (int i = 0; i < n; i++) {
                for (int p = 0; p < ToyImages.PIXELS; p++) {
                    img[i][p] = bits[i][p] == 1 ? 1 : -1;
                }
            }
            out.put("BFN discrete " + s, GenerativeScores.images(img));
            out.put("BFN continuous " + s, GenerativeScores.images(Nets.rows(real.sampleContinuous(n, s,
                    new SplittableRandom(seed + s)))));
            out.put("DDIM " + s, GenerativeScores.images(ddpm.sampleDdim(sampleNone, s, 1f, seed + s, null)));
        }
        out.put("DDPM 100", GenerativeScores.images(ddpm.sampleDdpm(sampleNone, 1f, seed, null)));
        return out;
    }
}
