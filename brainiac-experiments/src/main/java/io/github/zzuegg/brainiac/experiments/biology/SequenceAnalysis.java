package io.github.zzuegg.brainiac.experiments.biology;

import io.github.zzuegg.brainiac.experimental.bio.KmerClassifier;
import io.github.zzuegg.brainiac.experimental.bio.LearnedScoring;
import io.github.zzuegg.brainiac.experimental.bio.MotifCnn;
import io.github.zzuegg.brainiac.experimental.bio.MotifFinder;
import io.github.zzuegg.brainiac.experimental.bio.ProfileHmm;
import io.github.zzuegg.brainiac.experimental.bio.SequenceAlignment;
import io.github.zzuegg.brainiac.experimental.data.BioSequences;
import io.github.zzuegg.brainiac.experimental.data.MinHashDeduplicator;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Sequence analysis on synthetic proteins and DNA with known truth (wave 12, items
 * 12.4.1-12.4.3).
 *
 * <ul>
 *   <li>{@link #alignment}: alignment accuracy of homologous protein pairs at increasing
 *       divergence for the edit-distance alignment, Smith-Waterman with an identity matrix,
 *       Needleman-Wunsch with affine gaps, and Smith-Waterman with scores learned through the
 *       differentiable soft Smith-Waterman (a substitution matrix, and window-3 conv
 *       embeddings); plus how well the learned matrix recovers the generator's hidden
 *       substitution groups.</li>
 *   <li>{@link #homology}: homology search ROC AUC (one homologue among random decoys per
 *       query) for affine SW (identity and learned), negative edit distance and 3-mer
 *       Jaccard similarity (via {@link MinHashDeduplicator#jaccard}).</li>
 *   <li>{@link #families}: family detection AUROC of profile HMMs (star alignment plus
 *       Baum-Welch) against the best pairwise SW score to the training members, a flat
 *       {@link HiddenMarkovModel}, a 2-mer logistic regression and a 1-D CNN.</li>
 *   <li>{@link #motifs}: motif discovery with MEME, the Gibbs sampler, k-mer enrichment and a
 *       DeepBind CNN (PWM correlation with the planted motif, site F1, bound/unbound AUROC,
 *       with a 4-mer logistic regression as the classification baseline).</li>
 * </ul>
 */
public final class SequenceAnalysis {

    private static final SequenceAlignment.Scoring IDENTITY = SequenceAlignment.Scoring.identity(20, 2, -1, 3, 1);

    private SequenceAnalysis() {
    }

    private static LearnedScoring[] train(int pairs, int length, SplittableRandom rnd) {
        List<BioSequences.Pair> data = new ArrayList<>();
        for (int i = 0; i < pairs; i++) {
            data.add(BioSequences.homologousPair(length, 0.3 + 0.4 * rnd.nextDouble(), 0.05, 20, rnd));
        }
        LearnedScoring matrix = new LearnedScoring(LearnedScoring.Mode.MATRIX, 20, 0, 1);
        matrix.fit(data, 6, 0.1);
        LearnedScoring conv = new LearnedScoring(LearnedScoring.Mode.CONV_EMBEDDING, 20, 24, 2);
        conv.fit(data, 6, 0.03);
        return new LearnedScoring[] {matrix, conv};
    }

    /**
     * Alignment accuracy against divergence.
     *
     * @param divergences substitution rates of the test pairs
     * @param trainPairs training pairs for the learned scores (divergence 0.3-0.7)
     * @param testPairs test pairs per divergence
     * @param length sequence length
     * @param seed the seed
     * @return method to accuracy per divergence; the key {@code "learned matrix group recovery"}
     *     holds the mean learned score of within-group minus between-group substitutions
     */
    public static Map<String, double[]> alignment(double[] divergences, int trainPairs, int testPairs, int length,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        LearnedScoring[] learned = train(trainPairs, length, rnd);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"edit distance", "SW identity, affine gaps", "NW identity, affine gaps",
            "SW learned matrix (soft-SW CRF)", "SW learned conv embeddings"};
        for (String n : names) {
            out.put(n, new double[divergences.length]);
        }
        for (int d = 0; d < divergences.length; d++) {
            for (int t = 0; t < testPairs; t++) {
                BioSequences.Pair p = BioSequences.homologousPair(length, divergences[d], 0.05, 20, rnd);
                double[] acc = {
                    SequenceAlignment.editAlignment(p.a(), p.b()).accuracy(p.match()),
                    SequenceAlignment.smithWaterman(p.a(), p.b(), IDENTITY).accuracy(p.match()),
                    SequenceAlignment.needlemanWunsch(p.a(), p.b(), IDENTITY).accuracy(p.match()),
                    learned[0].align(p.a(), p.b()).accuracy(p.match()),
                    learned[1].align(p.a(), p.b()).accuracy(p.match())};
                for (int m = 0; m < names.length; m++) {
                    out.get(names[m])[d] += acc[m] / testPairs;
                }
            }
        }
        double[][] s = learned[0].substitution();
        double within = 0;
        double between = 0;
        int nw = 0;
        int nb = 0;
        for (int a = 0; a < 20; a++) {
            for (int b = 0; b < 20; b++) {
                if (a == b) {
                    continue;
                }
                if (BioSequences.proteinGroup(a) == BioSequences.proteinGroup(b)) {
                    within += s[a][b];
                    nw++;
                } else {
                    between += s[a][b];
                    nb++;
                }
            }
        }
        out.put("learned matrix group recovery", new double[] {within / nw - between / nb});
        return out;
    }

    /**
     * Homology search: for each query one homologue and {@code decoys} random sequences.
     *
     * @param divergences divergences of the homologues
     * @param queries queries per divergence
     * @param decoys decoys per query
     * @param length sequence length
     * @param seed the seed
     * @return method to ROC AUC per divergence
     */
    public static Map<String, double[]> homology(double[] divergences, int queries, int decoys, int length,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        LearnedScoring matrix = train(30, length, rnd)[0];
        String[] names = {"SW identity", "SW learned matrix", "-edit distance", "3-mer Jaccard"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            out.put(n, new double[divergences.length]);
        }
        for (int d = 0; d < divergences.length; d++) {
            int total = queries * (decoys + 1);
            double[][] scores = new double[names.length][total];
            boolean[] positive = new boolean[total];
            int idx = 0;
            for (int q = 0; q < queries; q++) {
                BioSequences.Pair p = BioSequences.homologousPair(length, divergences[d], 0.05, 20, rnd);
                String qa = BioSequences.text(p.a(), BioSequences.PROTEIN);
                for (int c = 0; c <= decoys; c++) {
                    int[] target = c == 0 ? p.b() : BioSequences.randomSequence(p.b().length, 20, rnd);
                    positive[idx] = c == 0;
                    scores[0][idx] = SequenceAlignment.smithWaterman(p.a(), target, IDENTITY).score();
                    scores[1][idx] = matrix.align(p.a(), target).score();
                    scores[2][idx] = -SequenceAlignment.editDistance(p.a(), target);
                    scores[3][idx] = MinHashDeduplicator.jaccard(qa, BioSequences.text(target, BioSequences.PROTEIN), 3);
                    idx++;
                }
            }
            for (int m = 0; m < names.length; m++) {
                out.get(names[m])[d] = MlMetrics.auroc(scores[m], positive);
            }
        }
        return out;
    }

    /**
     * Family detection: every family's profile is built from {@code trainMembers} members and
     * scores held-out members (positives) against random decoys and the other families' held-out
     * members (negatives); AUROC is averaged over families.
     *
     * @param familyCount families
     * @param trainMembers training members per family
     * @param testMembers held-out members per family
     * @param identity the family identity
     * @param seed the seed
     * @return method to {@code {mean AUROC}}
     */
    public static Map<String, double[]> families(int familyCount, int trainMembers, int testMembers, double identity,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int length = 48;
        List<List<int[]>> fam = new ArrayList<>();
        for (int f = 0; f < familyCount; f++) {
            fam.add(BioSequences.family(trainMembers + testMembers, length, identity, 0.03, rnd));
        }
        List<int[]> decoys = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            decoys.add(BioSequences.randomSequence(length, 20, rnd));
        }
        List<int[]> trainDecoys = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            trainDecoys.add(BioSequences.randomSequence(length, 20, rnd));
        }
        double[] bg = new double[20];
        Arrays.fill(bg, 1 / 20.0);
        String[] names = {"profile HMM (forward log-odds)", "profile HMM (Viterbi)", "best pairwise SW",
            "flat HMM (5 states)", "2-mer logistic regression", "1-D CNN"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            out.put(n, new double[1]);
        }
        for (int f = 0; f < familyCount; f++) {
            List<int[]> train = fam.get(f).subList(0, trainMembers);
            ProfileHmm hmm = ProfileHmm.fromUnaligned(train, bg, 1.0);
            hmm.baumWelch(train, 4, 1.0);
            List<double[]> obs = new ArrayList<>();
            for (int[] s : train) {
                obs.add(toDouble(s));
            }
            HiddenMarkovModel flat = HiddenMarkovModel.randomDiscrete(5, 20, seed + f).baumWelch(obs, 10);
            int[][] clsX = new int[trainMembers + trainDecoys.size()][];
            boolean[] clsY = new boolean[clsX.length];
            for (int i = 0; i < trainMembers; i++) {
                clsX[i] = train.get(i);
                clsY[i] = true;
            }
            for (int i = 0; i < trainDecoys.size(); i++) {
                clsX[trainMembers + i] = trainDecoys.get(i);
            }
            KmerClassifier kmer = new KmerClassifier(2, 20).fit(clsX, clsY, 150, 1e-3);
            MotifCnn cnn = new MotifCnn(20, length, 6, 8, seed + f);
            cnn.fit(clsX, clsY, 150, 16, 0.01f, seed + f);
            List<int[]> test = new ArrayList<>();
            List<Boolean> pos = new ArrayList<>();
            for (int g = 0; g < familyCount; g++) {
                for (int[] s : fam.get(g).subList(trainMembers, trainMembers + testMembers)) {
                    test.add(s);
                    pos.add(g == f);
                }
            }
            for (int[] s : decoys) {
                test.add(s);
                pos.add(false);
            }
            boolean[] positive = new boolean[test.size()];
            double[][] sc = new double[names.length][test.size()];
            double[] cnnScores = cnn.scores(test.toArray(new int[0][]));
            for (int i = 0; i < test.size(); i++) {
                int[] s = test.get(i);
                positive[i] = pos.get(i);
                sc[0][i] = hmm.logOdds(s);
                sc[1][i] = hmm.viterbiLogOdds(s);
                double best = Double.NEGATIVE_INFINITY;
                for (int[] t : train) {
                    best = Math.max(best, SequenceAlignment.smithWaterman(s, t, IDENTITY).score());
                }
                sc[2][i] = best;
                sc[3][i] = flat.logLikelihood(toDouble(s)) + s.length * Math.log(20);
                sc[4][i] = kmer.score(s);
                sc[5][i] = cnnScores[i];
            }
            for (int m = 0; m < names.length; m++) {
                out.get(names[m])[0] += MlMetrics.auroc(sc[m], positive) / familyCount;
            }
        }
        return out;
    }

    private static double[] toDouble(int[] s) {
        double[] d = new double[s.length];
        for (int i = 0; i < s.length; i++) {
            d[i] = s[i];
        }
        return d;
    }

    /**
     * Motif discovery on DNA with a planted PWM.
     *
     * @param n sequences (half bound)
     * @param length sequence length
     * @param width motif width
     * @param occupancy share of bound sequences carrying a site
     * @param seed the seed
     * @return method to {@code {PWM correlation, site F1 (bound sequences), bound/unbound AUROC}}
     *     (NaN where a method does not provide the quantity)
     */
    public static Map<String, double[]> motifs(int n, int length, int width, double occupancy, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        BioSequences.MotifSet all = BioSequences.motifs(2 * n, length, width, occupancy, rnd);
        BioSequences.MotifSet train = half(all, 0, n);
        BioSequences.MotifSet test = half(all, n, n);
        List<int[]> bound = new ArrayList<>();
        List<Integer> truthSites = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (train.bound()[i]) {
                bound.add(train.sequences()[i]);
                truthSites.add(train.sites()[i]);
            }
        }
        int[][] pos = bound.toArray(new int[0][]);
        int[] truth = truthSites.stream().mapToInt(Integer::intValue).toArray();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (MotifFinder.Method m : MotifFinder.Method.values()) {
            MotifFinder.Motif motif = MotifFinder.find(m, pos, width, 4, seed);
            double[] testScores = new double[n];
            for (int i = 0; i < n; i++) {
                testScores[i] = bestSite(motif.pwm(), test.sequences()[i]);
            }
            out.put(m.name(), new double[] {MotifFinder.pwmCorrelation(motif.pwm(), train.pwm()),
                MotifFinder.siteF1(motif.sites(), truth, width), MlMetrics.auroc(testScores, test.bound())});
        }
        MotifCnn cnn = new MotifCnn(4, length, width, 8, seed);
        cnn.fit(train.sequences(), train.bound(), 300, 32, 0.01f, seed);
        double[][] pwm = cnn.filterPwms(pos)[cnn.topFilter()];
        out.put("DeepBind CNN", new double[] {MotifFinder.pwmCorrelation(pwm, train.pwm()), Double.NaN,
            MlMetrics.auroc(cnn.scores(test.sequences()), test.bound())});
        KmerClassifier kmer = new KmerClassifier(4, 4).fit(train.sequences(), train.bound(), 200, 1e-3);
        double[] ks = new double[n];
        for (int i = 0; i < n; i++) {
            ks[i] = kmer.score(test.sequences()[i]);
        }
        out.put("4-mer logistic regression", new double[] {Double.NaN, Double.NaN, MlMetrics.auroc(ks, test.bound())});
        return out;
    }

    private static BioSequences.MotifSet half(BioSequences.MotifSet all, int from, int n) {
        return new BioSequences.MotifSet(Arrays.copyOfRange(all.sequences(), from, from + n),
                Arrays.copyOfRange(all.bound(), from, from + n), Arrays.copyOfRange(all.sites(), from, from + n),
                all.pwm());
    }

    private static double bestSite(double[][] pwm, int[] s) {
        double best = Double.NEGATIVE_INFINITY;
        for (int j = 0; j + pwm.length <= s.length; j++) {
            double v = 0;
            for (int w = 0; w < pwm.length; w++) {
                v += Math.log(pwm[w][s[j + w]] / 0.25);
            }
            best = Math.max(best, v);
        }
        return best;
    }
}
