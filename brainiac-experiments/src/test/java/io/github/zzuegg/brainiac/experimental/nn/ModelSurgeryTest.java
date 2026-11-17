package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ModelSurgeryTest {

    static int argmax(float[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }

    /** Two Gaussian blobs, class = which blob. */
    static Model blobClassifier(float[][] x0, float[][] x1) {
        SplittableRandom r = new SplittableRandom(1);
        int n = x0.length;
        float[][] x = new float[2 * n][];
        float[][] y = new float[2 * n][2];
        for (int i = 0; i < n; i++) {
            x0[i] = new float[] {(float) r.nextGaussian() * 0.5f - 1.5f, (float) r.nextGaussian() * 0.5f};
            x1[i] = new float[] {(float) r.nextGaussian() * 0.5f + 1.5f, (float) r.nextGaussian() * 0.5f};
            x[2 * i] = x0[i];
            y[2 * i][0] = 1;
            x[2 * i + 1] = x1[i];
            y[2 * i + 1][1] = 1;
        }
        Model m = Model.sequential().input(2).dense(8, Activation.TANH).dense(8, Activation.TANH)
                .dense(2, Activation.SOFTMAX).seed(3).build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.03f)).batchSize(32).epochs(40)
                .fit(Dataset.of(x, y));
        return m;
    }

    @Test
    void steeringVectorFlipsTheBehaviour() {
        float[][] x0 = new float[100][];
        float[][] x1 = new float[100][];
        Model m = blobClassifier(x0, x1);
        SteeringVector v = SteeringVector.fromContrast(m, 1, Tensor.of(x1), Tensor.of(x0));
        Model steered = v.apply(m, 1.0f);
        int flipped = 0;
        for (float[] x : x0) {
            assertEquals(0, argmax(m.predict(x)));
            flipped += argmax(steered.predict(x)) == 1 ? 1 : 0;
        }
        assertTrue(flipped > 90, "flipped " + flipped);
        // the steered model shares weights; the original is unchanged
        assertEquals(0, argmax(m.predict(x0[0])));
        assertEquals(m.parameters().size(), steered.parameters().size());
    }

    @Test
    void patchingTheWholeLayerRestoresTheCleanOutput() {
        float[][] x0 = new float[50][];
        float[][] x1 = new float[50][];
        Model m = blobClassifier(x0, x1);
        Tensor clean = Tensor.of(new float[][] {x1[0]});
        Tensor corrupt = Tensor.of(new float[][] {x0[0]});
        double[] effect = ActivationPatching.traceLayers(m, clean, corrupt, ActivationPatching.output(1));
        for (double e : effect) {
            assertEquals(1, e, 1e-4);   // a chain: every full layer carries everything
        }
        Tensor partial = ActivationPatching.patch(m, 0, clean, corrupt, -1, new int[] {0});
        assertTrue(partial.toArray()[1] < m.predict(x1[0])[1] + 1e-6);
    }

    /** A layer that keeps only time step 0 of a sequence {@code [batch, time, features]}. */
    private static final class FirstStep implements Layer {
        @Override
        public Var forward(Var x) {
            return x.slice(1, 0, 1).reshape(x.shape().dim(0), x.shape().dim(2));
        }

        @Override
        public Shape outputShape(Shape in) {
            return Shape.of(in.dim(-1));
        }
    }

    @Test
    void causalTracingFindsThePositionThatMatters() {
        Model m = Model.sequential().input(4, 3).dense(3, Activation.TANH).layer(new FirstStep()).dense(1).seed(1)
                .build();
        float[] c = new float[12];
        float[] x = new float[12];
        for (int i = 0; i < 12; i++) {
            c[i] = 1;
            x[i] = -1;
        }
        double[][] t = ActivationPatching.tracePositions(m, Tensor.of(Shape.of(1, 4, 3), c),
                Tensor.of(Shape.of(1, 4, 3), x), ActivationPatching.output(0));
        assertEquals(1, t[0][0], 1e-5);   // position 0 of the first layer restores everything
        for (int p = 1; p < 4; p++) {
            assertEquals(0, t[0][p], 1e-5);   // later positions are never read
        }
    }

    @Test
    void romeRewritesOneFactAndLeavesTheOthers() {
        int subjects = 10;
        int classes = 4;
        float[][] x = new float[subjects][subjects];
        float[][] y = new float[subjects][classes];
        int[] fact = new int[subjects];
        SplittableRandom r = new SplittableRandom(5);
        for (int s = 0; s < subjects; s++) {
            x[s][s] = 1;
            fact[s] = r.nextInt(classes);
            y[s][fact[s]] = 1;
        }
        Model m = Model.sequential().input(subjects).dense(16, Activation.TANH).dense(16, Activation.RELU)
                .dense(classes, Activation.SOFTMAX).seed(2).build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.03f)).batchSize(10).epochs(300)
                .fit(Dataset.of(x, y));
        for (int s = 0; s < subjects; s++) {
            assertEquals(fact[s], argmax(m.predict(x[s])));
        }
        int subject = 3;
        int newFact = (fact[subject] + 1) % classes;
        double[][] cov = RomeEditor.keyCovariance(m, 1, Tensor.of(x), 1e-2);
        RomeEditor.edit(m, 1, Tensor.of(new float[][] {x[subject]}), newFact, cov, 200);
        assertEquals(newFact, argmax(m.predict(x[subject])));
        int kept = 0;
        for (int s = 0; s < subjects; s++) {
            if (s != subject) {
                kept += argmax(m.predict(x[s])) == fact[s] ? 1 : 0;
            }
        }
        assertTrue(kept >= 8, "kept " + kept + "/9 other facts");
    }
}
