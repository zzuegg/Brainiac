package io.github.zzuegg.brainiac.experiments.biology;

import io.github.zzuegg.brainiac.experimental.bio.CoevolutionScores;
import io.github.zzuegg.brainiac.experimental.bio.PhyloGfn;
import io.github.zzuegg.brainiac.experimental.bio.Phylogenetics;
import io.github.zzuegg.brainiac.experimental.bio.PottsModel;
import io.github.zzuegg.brainiac.experimental.bio.RnaFolding;
import io.github.zzuegg.brainiac.experimental.data.BioSequences;
import io.github.zzuegg.brainiac.experimental.data.BioSequences.Tree;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Structure and phylogeny on synthetic data with known truth (wave 12, items 12.4.4-12.4.6).
 *
 * <ul>
 *   <li>{@link #phylogeny}: Robinson-Foulds distance to the true tree for UPGMA, neighbour
 *       joining and maximum likelihood with NNI search (from NJ), against a random tree, on
 *       clock-like and strongly non-clock trees, with the log-likelihood gap of the ML tree to
 *       the true topology.</li>
 *   <li>{@link #phyloGfn}: PhyloGFN's samples (trained with the unchanged {@code GFlowNet}):
 *       the best and mean parsimony of the sampled trees against NJ and ML, and the number of
 *       distinct topologies within 2 substitutions of the best known score.</li>
 *   <li>{@link #contacts}: top-K contact precision (K = the number of planted contacts,
 *       separation of at least 3) for MI, MI+APC, a one-hot lasso with APC and plmDCA with and
 *       without APC, optionally under phylogenetic correlation.</li>
 *   <li>{@link #folding}: base-pair F1 and the share of valid structures for Nussinov, the
 *       hand-set energy model (MFE and McCaskill MEA), the CRF-trained weights and an
 *       unconstrained pairing CNN with greedy decoding.</li>
 * </ul>
 */
public final class StructureAndPhylogeny {

    private StructureAndPhylogeny() {
    }

    /**
     * Tree reconstruction accuracy.
     *
     * @param taxa leaves
     * @param sites alignment columns
     * @param clock whether the true tree is clock-like
     * @param reps replicate trees
     * @param seed the seed
     * @return method to {@code {mean normalised RF distance, mean log-likelihood minus the true
     *     topology's (both with optimised branch lengths)}}
     */
    public static Map<String, double[]> phylogeny(int taxa, int sites, boolean clock, int reps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"random tree", "UPGMA (JC)", "neighbour joining (K2P)", "ML + NNI (from NJ)"};
        for (String n : names) {
            out.put(n, new double[2]);
        }
        for (int r = 0; r < reps; r++) {
            Tree truth = BioSequences.randomTree(taxa, 0.4, clock, rnd);
            int[][] aln = BioSequences.evolve(truth, sites, 2.0, rnd);
            double trueLl = Phylogenetics.logLikelihood(Phylogenetics.optimiseBranches(truth, aln, 1, 2), aln, 1);
            Tree random = BioSequences.randomTree(taxa, 0.4, true, rnd);
            Tree up = Phylogenetics.upgma(Phylogenetics.distances(aln, false));
            Tree nj = Phylogenetics.neighbourJoining(Phylogenetics.distances(aln, true));
            Tree ml = Phylogenetics.nniSearch(nj, aln, 1, 10);
            Tree[] trees = {random, up, nj, ml};
            for (int m = 0; m < trees.length; m++) {
                double ll = Phylogenetics.logLikelihood(Phylogenetics.optimiseBranches(trees[m], aln, 1, 1), aln, 1);
                out.get(names[m])[0] += trees[m].robinsonFoulds(truth) / reps;
                out.get(names[m])[1] += (ll - trueLl) / reps;
            }
        }
        return out;
    }

    /**
     * PhyloGFN against distance and likelihood methods on parsimony.
     *
     * @param taxa leaves (at most 10 for the test budget)
     * @param sites alignment columns
     * @param steps GFlowNet training steps
     * @param seed the seed
     * @return {@code {NJ parsimony, ML parsimony, true-tree parsimony, PhyloGFN best, PhyloGFN mean,
     *     distinct sampled topologies within 2 of the best known, samples, RF of the best sample
     *     to the truth}}
     */
    public static double[] phyloGfn(int taxa, int sites, int steps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Tree truth = BioSequences.randomTree(taxa, 0.5, false, rnd);
        int[][] aln = BioSequences.evolve(truth, sites, 2.0, rnd);
        Tree nj = Phylogenetics.neighbourJoining(Phylogenetics.distances(aln, true));
        Tree ml = Phylogenetics.nniSearch(nj, aln, 1, 6);
        int pNj = Phylogenetics.parsimony(nj, aln);
        int pMl = Phylogenetics.parsimony(ml, aln);
        int pTrue = Phylogenetics.parsimony(truth, aln);
        PhyloGfn gfn = new PhyloGfn(aln, 4.0, pNj);
        int samples = 64;
        List<int[]> drawn = gfn.trainAndSample(steps, 16, samples, seed);
        int best = Integer.MAX_VALUE;
        double mean = 0;
        List<Integer> scores = new ArrayList<>();
        Tree bestTree = null;
        for (int[] s : drawn) {
            int p = gfn.parsimony(s);
            scores.add(p);
            mean += p / (double) samples;
            if (p < best) {
                best = p;
                bestTree = gfn.toTree(s);
            }
        }
        int reference = Math.min(Math.min(best, pNj), Math.min(pMl, pTrue));
        Set<Set<String>> distinct = new HashSet<>();
        for (int i = 0; i < drawn.size(); i++) {
            if (scores.get(i) <= reference + 2) {
                distinct.add(gfn.toTree(drawn.get(i)).splits());
            }
        }
        return new double[] {pNj, pMl, pTrue, best, mean, distinct.size(), samples, bestTree.robinsonFoulds(truth)};
    }

    /**
     * Contact prediction from coevolution.
     *
     * @param length positions
     * @param states states q
     * @param n sequences
     * @param strength the planted coupling strength
     * @param phylogeny whether samples are phylogenetically clustered
     * @param seed the seed
     * @return method to {@code {top-K precision}}
     */
    public static Map<String, double[]> contacts(int length, int states, int n, double strength, boolean phylogeny,
            long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        BioSequences.PottsSamples data = BioSequences.potts(length, states, n, strength, phylogeny, rnd);
        int k = 0;
        for (int a = 0; a < length; a++) {
            for (int b = a + 3; b < length; b++) {
                k += data.contacts()[a][b] ? 1 : 0;
            }
        }
        double[][] mi = CoevolutionScores.mutualInformation(data.sequences(), states, 0.5);
        double[][] lasso = CoevolutionScores.lasso(data.sequences(), states, 0.01, 40);
        PottsModel potts = PottsModel.fitPseudoLikelihood(data.sequences(), states, 0.01, 150);
        double[][] fro = potts.frobeniusScores();
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("MI", new double[] {PottsModel.topPrecision(mi, data.contacts(), k, 3)});
        out.put("MI + APC", new double[] {PottsModel.topPrecision(PottsModel.apc(mi), data.contacts(), k, 3)});
        out.put("one-hot lasso + APC",
                new double[] {PottsModel.topPrecision(PottsModel.apc(lasso), data.contacts(), k, 3)});
        out.put("plmDCA", new double[] {PottsModel.topPrecision(fro, data.contacts(), k, 3)});
        out.put("plmDCA + APC", new double[] {PottsModel.topPrecision(PottsModel.apc(fro), data.contacts(), k, 3)});
        return out;
    }

    /**
     * RNA secondary structure prediction.
     *
     * @param train training sequences for the learned models
     * @param test test sequences
     * @param minLength shortest sequence
     * @param maxLength longest sequence
     * @param seed the seed
     * @return method to {@code {mean base-pair F1, share of valid structures}}
     */
    public static Map<String, double[]> folding(int train, int test, int minLength, int maxLength, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<BioSequences.RnaStructure> trainSet = new ArrayList<>();
        List<BioSequences.RnaStructure> testSet = new ArrayList<>();
        for (int i = 0; i < train + test; i++) {
            BioSequences.RnaStructure s = BioSequences.rna(minLength + rnd.nextInt(maxLength - minLength + 1), rnd);
            (i < train ? trainSet : testSet).add(s);
        }
        double[] energy = RnaFolding.energyWeights();
        double[] crf = RnaFolding.trainCrf(trainSet, energy, 12, 0.05, 1e-3);
        RnaFolding.PairCnn cnn = new RnaFolding.PairCnn(16, seed);
        cnn.fit(trainSet, 300, 0.01f, seed);
        String[] names = {"Nussinov", "energy model MFE (simple Zuker)", "McCaskill MEA (energy model)",
            "CRF-trained weights (MEA)", "CRF-trained weights (Viterbi)", "pairing CNN (greedy)"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            out.put(n, new double[2]);
        }
        for (BioSequences.RnaStructure s : testSet) {
            int[] seq = s.sequence();
            int[][] preds = {RnaFolding.nussinov(seq), RnaFolding.fold(seq, energy),
                RnaFolding.meaFold(RnaFolding.pairProbabilities(seq, energy), 1),
                RnaFolding.meaFold(RnaFolding.pairProbabilities(seq, crf), 1), RnaFolding.fold(seq, crf),
                cnn.predict(seq)};
            for (int m = 0; m < names.length; m++) {
                out.get(names[m])[0] += RnaFolding.pairF1(preds[m], s.pairs()) / test;
                out.get(names[m])[1] += (RnaFolding.isValid(seq, preds[m]) ? 1.0 : 0.0) / test;
            }
        }
        return out;
    }
}
