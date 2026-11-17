package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.PrototypicalNetworks;
import io.github.zzuegg.brainiac.experimental.train.Maml;
import io.github.zzuegg.brainiac.experimental.train.Reptile;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Meta-learning experiments.
 *
 * <p><b>Few-shot sine regression</b> (the MAML paper's benchmark): every task is a sine wave
 * {@code a sin(x + phase)} with its own amplitude and phase; the learner sees 10 points of a
 * new wave, takes 5 SGD steps, and is scored on 50 others. Compared: first-order MAML,
 * second-order MAML, Reptile and an untrained initialization.
 *
 * <p><b>Few-shot classification with prototypical networks</b>: an embedding is trained on
 * 4-way 5-shot episodes of four {@link ShapeImages} classes and tested on 2-way 3-shot
 * episodes of the two classes it never saw, against an untrained embedding.
 */
public final class MetaLearningExperiment {

    /** The sine-regression learners. */
    public enum Learner {
        /** No meta-training: adapt from the random initialization. */
        SCRATCH,
        /** First-order MAML. */
        FOMAML,
        /** MAML through the inner steps (exact Hessian-vector products by nested autograd). */
        MAML,
        /** Reptile. */
        REPTILE
    }

    private static final float INNER_RATE = 0.01f;

    private MetaLearningExperiment() {
    }

    /** One sine task with fixed support and query points. */
    private record Sine(Model model, Tensor xs, Tensor ys, Tensor xq, Tensor yq) implements Maml.Task {

        static Sine sample(Model model, SplittableRandom r, int support, int query) {
            double amp = 0.5 + 4.5 * r.nextDouble();
            double phase = Math.PI * r.nextDouble();
            Tensor[] s = points(r, support, amp, phase);
            Tensor[] q = points(r, query, amp, phase);
            return new Sine(model, s[0], s[1], q[0], q[1]);
        }

        static Tensor[] points(SplittableRandom r, int n, double amp, double phase) {
            float[] x = new float[n];
            float[] y = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = (float) (-5 + 10 * r.nextDouble());
                y[i] = (float) (amp * Math.sin(x[i] + phase));
            }
            return new Tensor[] {Tensor.of(Shape.of(n, 1), x), Tensor.of(Shape.of(n, 1), y)};
        }

        @Override
        public Var supportLoss() {
            return model.forward(Var.of(xs)).sub(ys).square().mean();
        }

        @Override
        public Var queryLoss() {
            return model.forward(Var.of(xq)).sub(yq).square().mean();
        }
    }

    /**
     * Meta-trains one learner and returns its mean query error after adapting to new waves.
     *
     * @param learner the learner
     * @param iterations meta iterations of 5 tasks each
     * @param seed the seed
     * @return the mean squared error on 20 new tasks after 5 adaptation steps
     */
    public static float sine(Learner learner, int iterations, long seed) {
        Model model = Model.sequential().input(1).dense(40, Activation.RELU, Init.XAVIER)
                .dense(40, Activation.RELU, Init.XAVIER)
                .dense(1, Activation.IDENTITY).seed(seed).build();
        SplittableRandom r = new SplittableRandom(seed);
        Maml maml = new Maml(model.parameters(), Optimizer.adam(3e-3f), INNER_RATE, 1, learner == Learner.MAML);
        Reptile reptile = new Reptile(model.parameters(), INNER_RATE, 5, 0.3f);
        for (int it = 0; it < iterations && learner != Learner.SCRATCH; it++) {
            List<Sine> tasks = new ArrayList<>();
            for (int k = 0; k < 5; k++) {
                tasks.add(Sine.sample(model, r, 10, 10));
            }
            if (learner == Learner.REPTILE) {
                List<Supplier<Var>> losses = new ArrayList<>();
                for (Sine t : tasks) {
                    losses.add(t::supportLoss);
                }
                reptile.metaStep(losses);
            } else {
                maml.metaStep(tasks);
            }
        }
        SplittableRandom test = new SplittableRandom(seed + 1);
        double total = 0;
        for (int k = 0; k < 20; k++) {
            Sine t = Sine.sample(model, test, 10, 50);
            List<Tensor> start = maml.snapshot();
            maml.adapt(t, 5);
            total += t.queryLoss().value().item();
            maml.restore(start);
        }
        return (float) (total / 20);
    }

    /**
     * Trains a prototypical-network embedding and tests it on unseen classes.
     *
     * @param episodes training episodes; 0 for the untrained embedding
     * @param seed the seed
     * @return the accuracy on 2-way 3-shot episodes of the two held-out classes
     */
    public static float prototypical(int episodes, long seed) {
        Model embed = Model.sequential().input(64).dense(64, Activation.RELU).dense(32, Activation.IDENTITY)
                .seed(seed).build();
        SplittableRandom r = new SplittableRandom(seed);
        ShapeImages images = new ShapeImages(8, 0.1f);
        Tensor[] pool = images.generate(2000, seed + 5);
        List<List<float[]>> byClass = byClass(pool);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int e = 0; e < episodes; e++) {
            Tensor[] ep = episode(byClass, new int[] {0, 1, 2, 3}, 5, 5, r);
            int[][] labels = labels(4, 5, 5);
            PrototypicalNetworks.loss(embed.forward(Var.of(ep[0])), labels[0], embed.forward(Var.of(ep[1])), labels[1],
                    4).backward();
            opt.step(embed.parameters());
        }
        SplittableRandom test = new SplittableRandom(seed + 9);
        int correct = 0;
        int total = 0;
        for (int e = 0; e < 50; e++) {
            Tensor[] ep = episode(byClass, new int[] {4, 5}, 3, 10, test);
            int[][] labels = labels(2, 3, 10);
            Var protos = PrototypicalNetworks.prototypes(embed.forward(Var.of(ep[0])), labels[0], 2);
            int[] pred = PrototypicalNetworks.classify(embed.forward(Var.of(ep[1])), protos);
            for (int i = 0; i < pred.length; i++) {
                correct += pred[i] == labels[1][i] ? 1 : 0;
                total++;
            }
        }
        return correct / (float) total;
    }

    private static List<List<float[]>> byClass(Tensor[] pool) {
        List<List<float[]>> out = new ArrayList<>();
        for (int c = 0; c < ShapeImages.CLASSES; c++) {
            out.add(new ArrayList<>());
        }
        float[] x = pool[0].toArray();
        float[] y = pool[1].toArray();
        for (int i = 0; i < y.length; i++) {
            float[] img = new float[64];
            System.arraycopy(x, i * 64, img, 0, 64);
            out.get((int) y[i]).add(img);
        }
        return out;
    }

    /** {@code {support [ways * shots, 64], query [ways * queries, 64]}}, class-major. */
    private static Tensor[] episode(List<List<float[]>> byClass, int[] classes, int shots, int queries,
            SplittableRandom r) {
        float[] s = new float[classes.length * shots * 64];
        float[] q = new float[classes.length * queries * 64];
        for (int k = 0; k < classes.length; k++) {
            List<float[]> imgs = byClass.get(classes[k]);
            for (int i = 0; i < shots; i++) {
                System.arraycopy(imgs.get(r.nextInt(imgs.size())), 0, s, (k * shots + i) * 64, 64);
            }
            for (int i = 0; i < queries; i++) {
                System.arraycopy(imgs.get(r.nextInt(imgs.size())), 0, q, (k * queries + i) * 64, 64);
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(classes.length * shots, 64), s),
            Tensor.of(Shape.of(classes.length * queries, 64), q)};
    }

    private static int[][] labels(int ways, int shots, int queries) {
        int[] s = new int[ways * shots];
        int[] q = new int[ways * queries];
        for (int i = 0; i < s.length; i++) {
            s[i] = i / shots;
        }
        for (int i = 0; i < q.length; i++) {
            q[i] = i / queries;
        }
        return new int[][] {s, q};
    }
}
