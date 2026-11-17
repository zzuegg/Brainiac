package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

class RecsysTabularTest {

    /** Trains a logit model with binary cross-entropy and returns test accuracy. */
    static double accuracy(Layer model, SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(1);
        int n = train.rows();
        int cols = train.features().shape().dim(1);
        float[] x = train.features().toArray();
        float[] y = train.labels().toArray();
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[64 * cols];
            float[] by = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(n);
                System.arraycopy(x, j * cols, bx, i * cols, cols);
                by[i] = y[j];
            }
            Var logit = model.forward(Var.of(Tensor.of(Shape.of(64, cols), bx)), true);
            Tensor t = Tensor.of(Shape.of(64, 1), by);
            // log(1 + e^z) - y z
            Var loss = logit.softplus().sub(logit.mul(t)).mean();
            loss.backward();
            opt.step(model.parameters());
        }
        float[] p = model.forward(Var.of(test.features()), false).value().toArray();
        float[] ty = test.labels().toArray();
        int right = 0;
        for (int i = 0; i < p.length; i++) {
            right += (p[i] > 0) == (ty[i] > 0.5f) ? 1 : 0;
        }
        return right / (double) p.length;
    }

    static Layer logistic(int cols) {
        return new Dense(cols, 1, Activation.IDENTITY, Init.XAVIER, new SplittableRandom(1));
    }

    @Test
    void sparsemaxProjectsOntoTheSimplexSparsely() {
        Var p = Sparsemax.apply(Var.of(Tensor.of(Shape.of(2, 4), 2f, 1.5f, 0f, -1f, 0.1f, 0.1f, 0.1f, 0.1f)));
        float[] v = p.value().toArray();
        assertEquals(0.75f, v[0], 1e-6);
        assertEquals(0.25f, v[1], 1e-6);
        assertEquals(0f, v[2]);
        assertEquals(0.25f, v[5], 1e-6);
        GradCheck.Result g = GradCheck.check(xs -> Sparsemax.apply(xs.get(0)).mul(xs.get(1)).sum(),
                List.of(Tensor.of(Shape.of(2, 4), 1.1f, 0.4f, -0.7f, 0.9f, 0.3f, -0.2f, 0.5f, 0.05f),
                        Tensor.of(Shape.of(2, 4), 0.5f, -1f, 2f, 0.3f, 1f, 0.2f, -0.4f, 0.7f)));
        assertTrue(g.passed(), g.toString());
    }

    @Test
    void factorizationMachineCapturesFieldInteractions() {
        SyntheticRecsys.Table all = SyntheticRecsys.clicks(6000, 3, 8, 1);
        SyntheticRecsys.Table train = all.rows(0, 4000);
        SyntheticRecsys.Table test = all.rows(4000, 6000);
        double lr = accuracy(logistic(24), train, test, 600, 0.03f);
        double fm = accuracy(new FactorizationMachine(24, 8, new SplittableRandom(1)), train, test, 600, 0.03f);
        System.out.printf("clicks accuracy: logistic %.3f FM %.3f%n", lr, fm);
        assertTrue(fm > lr + 0.05, "FM " + fm + " LR " + lr);
    }

    @Test
    void tabNetAndCrossNetworkLearnFeatureInteractions() {
        SyntheticRecsys.Table train = SyntheticRecsys.tabular(3000, 10, 1);
        SyntheticRecsys.Table test = SyntheticRecsys.tabular(1500, 10, 2);
        double lr = accuracy(logistic(10), train, test, 500, 0.03f);
        TabNetLite tabnet = new TabNetLite(10, 1, 3, 16, 1.5f, new SplittableRandom(1));
        double tab = accuracy(tabnet, train, test, 800, 0.01f);
        CrossNetwork cross = new CrossNetwork(10, 2, new SplittableRandom(1));
        Dense crossHead = new Dense(10, 1, Activation.IDENTITY, Init.XAVIER, new SplittableRandom(2));
        Layer dcn = new Layer() {
            @Override
            public Var forward(Var input) {
                return crossHead.forward(cross.forward(input));
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new java.util.ArrayList<>(cross.parameters());
                p.addAll(crossHead.parameters());
                return p;
            }
        };
        double dc = accuracy(dcn, train, test, 800, 0.01f);
        Model mlp = Model.sequential().input(10).dense(32, Activation.RELU).dense(32, Activation.RELU).dense(1)
                .seed(1).build();
        Layer mlpLayer = new Layer() {
            @Override
            public Var forward(Var input) {
                return mlp.forward(input);
            }

            @Override
            public Var forward(Var input, boolean training) {
                return mlp.forward(input, training);
            }

            @Override
            public List<Var> parameters() {
                return mlp.parameters();
            }
        };
        double mlpAcc = accuracy(mlpLayer, train, test, 800, 0.01f);
        float[] maskUse = tabnet.lastMasks().sum(0, false).sum(0, false).toArray();
        double informative = 0;
        double total = 0;
        for (int c = 0; c < maskUse.length; c++) {
            total += maskUse[c];
            informative += c < 5 ? maskUse[c] : 0;
        }
        System.out.printf("tabular accuracy: logistic %.3f TabNet %.3f DCN %.3f MLP %.3f; TabNet mask mass on the 5 "
                + "informative columns %.2f%n", lr, tab, dc, mlpAcc, informative / total);
        assertTrue(tab > lr + 0.1, "TabNet " + tab);
        assertTrue(dc > lr + 0.1, "DCN " + dc);
        assertTrue(informative / total > 0.6, "masks should focus on informative columns");
    }

    @Test
    void twoTowerRetrievesRelevantItems() {
        SyntheticRecsys.Preferences data = SyntheticRecsys.preferences(200, 300, 6, 12, 0.1, 10, 1);
        Model user = Model.sequential().input(12).dense(32, Activation.RELU).dense(16).seed(1).build();
        Model item = Model.sequential().input(12).dense(32, Activation.RELU).dense(16).seed(2).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(user, item);
        float[] uf = data.userFeatures().toArray();
        float[] itf = data.itemFeatures().toArray();
        SplittableRandom r = new SplittableRandom(3);
        int trainUsers = 150;
        for (int s = 0; s < 400; s++) {
            float[] bu = new float[64 * 12];
            float[] bi = new float[64 * 12];
            for (int i = 0; i < 64; i++) {
                int u = r.nextInt(trainUsers);
                int it = data.relevant()[u][r.nextInt(10)];
                System.arraycopy(uf, u * 12, bu, i * 12, 12);
                System.arraycopy(itf, it * 12, bi, i * 12, 12);
            }
            tt.trainStep(Tensor.of(Shape.of(64, 12), bu), Tensor.of(Shape.of(64, 12), bi));
        }
        // held-out (cold-start) users
        Tensor testUsers = data.userFeatures().slice(0, trainUsers, 200);
        int[][] top = tt.topK(testUsers, data.itemFeatures(), 10);
        double hits = 0;
        for (int i = 0; i < top.length; i++) {
            java.util.Set<Integer> rel = new java.util.HashSet<>();
            for (int it : data.relevant()[trainUsers + i]) {
                rel.add(it);
            }
            for (int it : top[i]) {
                hits += rel.contains(it) ? 1 : 0;
            }
        }
        double recall = hits / (top.length * 10.0);
        System.out.printf("two-tower recall@10 on new users %.3f (random %.3f)%n", recall, 10.0 / 300);
        assertTrue(recall > 5 * 10.0 / 300, "recall " + recall);
    }
}
