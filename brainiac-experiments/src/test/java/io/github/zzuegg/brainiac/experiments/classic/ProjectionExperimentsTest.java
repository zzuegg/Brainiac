package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ProjectionExperimentsTest {

    @Test
    void icaUnmixesTheCocktailPartyWherePcaCannot() {
        Map<String, Double> r = ProjectionExperiments.cocktailParty(2000, 1);
        System.out.printf("source recovery |corr|: ICA %.3f vs PCA %.3f%n", r.get("ICA"), r.get("PCA"));
        assertTrue(r.get("ICA") > 0.95, r.toString());
        assertTrue(r.get("ICA") > r.get("PCA") + 0.1, r.toString());
    }

    @Test
    void kernelPcaMakesCirclesLinearlySeparable() {
        Map<String, Double> r = ProjectionExperiments.kernelPcaCircles(2);
        System.out.printf("circles, linear SVM: kernel PCA-3 %.3f vs PCA-2 %.3f%n",
                r.get("kernel PCA-3"), r.get("PCA-2"));
        assertTrue(r.get("kernel PCA-3") > 0.95, r.toString());
        assertTrue(r.get("PCA-2") < 0.75, r.toString());
    }

    @Test
    void tsneAndUmapKeepNeighbourhoodsBetterThanPca() {
        Map<String, double[]> r = ProjectionExperiments.embeddings(30, 240, 3);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s trust %.3f 5nn %.3f; ", k, v[0], v[1])));
        System.out.printf("2-D embeddings: %s%n", sb);
        assertTrue(r.get("digits / t-SNE")[0] > r.get("digits / PCA-2")[0] + 0.05, sb.toString());
        assertTrue(r.get("digits / UMAP")[0] > r.get("digits / PCA-2")[0] + 0.05, sb.toString());
        assertTrue(r.get("digits / t-SNE")[1] > r.get("digits / PCA-2")[1] + 0.2, sb.toString());
        assertTrue(r.get("digits / UMAP")[1] > r.get("digits / PCA-2")[1] + 0.2, sb.toString());
    }

    @Test
    void manifoldLearnersUnrollTheSwissRoll() {
        // s-curve and spheres are available too; three data sets keep the test short
        Map<String, double[]> r = ProjectionExperiments.manifolds(300,
                new String[] {"swiss roll", "swiss roll with hole", "digits"}, 4);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s T %.3f C %.3f rho %.3f %.0fms; ", k, v[0], v[1], v[2], v[3])));
        System.out.printf("manifolds: %s%n", sb);
        assertTrue(r.get("swiss roll / ISOMAP")[2] > r.get("swiss roll / PCA")[2] + 0.2, sb.toString());
        assertTrue(r.get("swiss roll / ISOMAP")[2] > 0.9, sb.toString());
        assertTrue(r.get("swiss roll / PACMAP")[2] > r.get("swiss roll / t-SNE")[2], sb.toString());
        assertTrue(r.get("digits / PACMAP")[0] > r.get("digits / PCA")[0], sb.toString());
    }
}
