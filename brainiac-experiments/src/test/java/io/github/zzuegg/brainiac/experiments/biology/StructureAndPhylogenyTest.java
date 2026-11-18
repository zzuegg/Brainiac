package io.github.zzuegg.brainiac.experiments.biology;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class StructureAndPhylogenyTest {

    @Test
    void neighbourJoiningAndMlBeatUpgmaWithoutAClock() {
        Map<String, double[]> nc = StructureAndPhylogeny.phylogeny(8, 300, false, 4, 1);
        System.out.printf("12.4.4 non-clock trees {RF to truth, logL - true logL}: %s%n",
                SequenceAnalysisTest.line(nc));
        Map<String, double[]> c = StructureAndPhylogeny.phylogeny(8, 300, true, 2, 2);
        System.out.printf("12.4.4 clock trees {RF to truth, logL - true logL}: %s%n", SequenceAnalysisTest.line(c));
        assertTrue(nc.get("ML + NNI (from NJ)")[0] <= nc.get("UPGMA (JC)")[0], SequenceAnalysisTest.line(nc));
        assertTrue(nc.get("neighbour joining (K2P)")[0] < nc.get("random tree")[0], SequenceAnalysisTest.line(nc));
        double[] g = StructureAndPhylogeny.phyloGfn(7, 300, 200, 3);
        System.out.printf("12.4.4 PhyloGFN parsimony: NJ %.0f, ML %.0f, true %.0f, GFN best %.0f, GFN mean %.1f, "
                + "distinct near-optimal topologies %.0f of %.0f samples, best-sample RF %.2f%n", g[0], g[1], g[2], g[3],
                g[4], g[5], g[6], g[7]);
        assertTrue(g[3] <= g[0] + 2, "PhyloGFN best " + g[3] + " vs NJ " + g[0]);
    }

    @Test
    void plmDcaWithApcFindsContacts() {
        Map<String, double[]> r = StructureAndPhylogeny.contacts(24, 8, 200, 1.0, false, 4);
        System.out.printf("12.4.5 top-K contact precision (L 24, q 8, 200 seqs, strength 1.0): %s%n",
                SequenceAnalysisTest.line(r));
        Map<String, double[]> p = StructureAndPhylogeny.contacts(24, 8, 200, 1.0, true, 5);
        System.out.printf("12.4.5 with phylogenetic correlation: %s%n", SequenceAnalysisTest.line(p));
        // at this size MI ties plmDCA (a negative result for the hypothesis); both must find contacts
        assertTrue(r.get("plmDCA + APC")[0] > 0.6, SequenceAnalysisTest.line(r));
        assertTrue(r.get("MI + APC")[0] > 0.6, SequenceAnalysisTest.line(r));
    }

    @Test
    void energyModelsBeatNussinov() {
        Map<String, double[]> f = StructureAndPhylogeny.folding(24, 12, 36, 60, 6);
        System.out.printf("12.4.6 RNA folding {pair F1, valid share}: %s%n", SequenceAnalysisTest.line(f));
        assertTrue(f.get("energy model MFE (simple Zuker)")[0] > f.get("Nussinov")[0], SequenceAnalysisTest.line(f));
        assertTrue(f.get("CRF-trained weights (MEA)")[1] == 1.0, SequenceAnalysisTest.line(f));
    }
}
