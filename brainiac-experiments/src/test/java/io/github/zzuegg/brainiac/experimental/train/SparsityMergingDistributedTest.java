package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.ScheduledDropout;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class SparsityMergingDistributedTest {

    static final Tensor[] DATA = Fixtures.spiral(60, 1);

    private static List<Var> matrices(SimpleMlp net) {
        List<Var> w = new ArrayList<>();
        for (int l = 0; l < net.depth(); l++) {
            w.add(net.weight(l));
        }
        return w;
    }

    @Test
    void lotteryTicketRewindsASparseSubnetworkThatStillTrains() {
        SimpleMlp net = SimpleMlp.tanh(1, 2, 64, 64, 3);
        LotteryTicket ticket = new LotteryTicket(matrices(net));
        for (int round = 0; round < 3; round++) {
            Fixtures.train(net, ticket.masked(Optimizer.adam(0.02f)), DATA, 150);
            ticket.prune(0.5);
            ticket.rewind();
        }
        assertEquals(0.875, ticket.sparsity(), 0.01);
        float loss = Fixtures.train(net, ticket.masked(Optimizer.adam(0.02f)), DATA, 200);
        double acc = LearningMath.accuracy(net.predict(DATA[0]), DATA[1]);
        assertTrue(acc > 0.9, "the 12.5% ticket trains: " + acc + " (loss " + loss + ")");
        float[] w = net.weight(1).value().toArray();
        float[] m = ticket.mask(net.weight(1)).toArray();
        for (int i = 0; i < w.length; i++) {
            if (m[i] == 0) {
                assertEquals(0, w[i], 0f);
            }
        }
    }

    @Test
    void riglKeepsItsDensityWhileRewiringAndTrains() {
        SimpleMlp net = SimpleMlp.tanh(2, 2, 64, 64, 3);
        RigL rigl = new RigL(Optimizer.adam(0.02f), matrices(net), 0.2, 0.3, 20, 300, 1);
        float[] before = rigl.mask(net.weight(1)).toArray();
        Fixtures.train(net, rigl, DATA, 400);
        assertEquals(0.2, rigl.density(), 0.01);
        assertTrue(rigl.updates() >= 10);
        float[] after = rigl.mask(net.weight(1)).toArray();
        int changed = 0;
        for (int i = 0; i < after.length; i++) {
            changed += before[i] != after[i] ? 1 : 0;
        }
        assertTrue(changed > 50, "connections moved: " + changed);
        double acc = LearningMath.accuracy(net.predict(DATA[0]), DATA[1]);
        assertTrue(acc > 0.85, "20%-dense RigL network trains: " + acc);
    }

    @Test
    void taskArithmeticTiesDareAndSoups() {
        List<Tensor> base = List.of(Tensor.of(0, 0, 0, 0));
        List<Tensor> a = List.of(Tensor.of(1, 0, 0.01f, -2));
        List<Tensor> b = List.of(Tensor.of(0, 1, -0.01f, 2));
        List<Tensor> ta = ModelMerging.taskVector(base, a);
        List<Tensor> sum = ModelMerging.add(base, List.of(ta, ModelMerging.taskVector(base, b)), 1);
        assertArrayEquals(new float[] {1, 1, 0, 0}, sum.get(0).toArray(), 1e-6f);
        assertArrayEquals(new float[] {-1, 0, -0.01f, 2}, ModelMerging.add(base, List.of(ta), -1).get(0).toArray(),
                1e-6f, "negating a task vector removes the skill");
        // TIES: trimming drops the tiny entries; the sign conflict at index 3 elects one side
        List<Tensor> ties = ModelMerging.ties(base, List.of(a, List.of(Tensor.of(0, 1, -0.01f, 1))), 0.5, 1);
        assertArrayEquals(new float[] {1, 1, 0, -2}, ties.get(0).toArray(), 1e-6f);
        List<Tensor> big = List.of(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(10_000), 1f));
        float[] dared = ModelMerging.dare(big, 0.9, 1).get(0).toArray();
        int zeros = 0;
        double mean = 0;
        for (float v : dared) {
            zeros += v == 0 ? 1 : 0;
            mean += v / dared.length;
        }
        assertEquals(0.9, zeros / 10_000.0, 0.02);
        assertEquals(1, mean, 0.1, "rescaling keeps the expected delta");
        assertArrayEquals(new float[] {0.5f, 0.5f, 0, 0}, ModelMerging.uniformSoup(List.of(a, b)).get(0).toArray(), 1e-6f);
        // greedy soup: a bad ingredient is left out
        List<Tensor> good1 = List.of(Tensor.of(1.2f, 0.8f, 1, 1));
        List<Tensor> good2 = List.of(Tensor.of(0.8f, 1.2f, 1, 1));
        List<Tensor> bad = List.of(Tensor.of(-5, -5, -5, -5));
        List<Tensor> soup = ModelMerging.greedySoup(List.of(good1, bad, good2),
                w -> -w.get(0).sub(1).square().sum().item());
        assertArrayEquals(new float[] {1, 1, 1, 1}, soup.get(0).toArray(), 1e-5f);
    }

    @Test
    void swaAveragesLateWeightsAndDropoutScheduleDrivesTheLayers() {
        Model model = Model.sequential().input(2).dense(32, Activation.TANH).add(ScheduledDropout.spec(0.3f))
                .dense(3, Activation.SOFTMAX).seed(1).build();
        StochasticWeightAveraging swa = new StochasticWeightAveraging(model.parameters(), 20, 5);
        DropoutSchedule early = DropoutSchedule.early(model, 0.3f, 10);
        float[] ids = DATA[1].toArray();
        float[][] x = new float[ids.length][];
        float[][] y = new float[ids.length][3];
        float[] xv = DATA[0].toArray();
        for (int i = 0; i < ids.length; i++) {
            x[i] = new float[] {xv[2 * i], xv[2 * i + 1]};
            y[i][Math.round(ids[i])] = 1;
        }
        Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(30).epochs(10)
                .listener(swa).listener(early).fit(Dataset.of(x, y));
        assertEquals(0, DropoutSchedule.layersOf(model).get(0).rate(), "early dropout switched off");
        assertTrue(swa.samples() > 5);
        float[] current = model.parameters().get(0).value().toArray();
        swa.swap();
        float[] averaged = model.parameters().get(0).value().toArray();
        assertTrue(!java.util.Arrays.equals(current, averaged));
        swa.swap();
        assertArrayEquals(current, model.parameters().get(0).value().toArray());
    }

    @Test
    void selfPacedWeightsSkipHardSamplesUntilTheThresholdGrows() {
        SelfPacedLearning spl = new SelfPacedLearning(1f, 2f, false);
        Tensor losses = Tensor.of(0.2f, 0.9f, 1.5f, 3f);
        assertArrayEquals(new float[] {1, 1, 0, 0}, spl.weights(losses).toArray());
        spl.nextEpoch();
        assertArrayEquals(new float[] {1, 1, 1, 0}, spl.weights(losses).toArray());
        SelfPacedLearning soft = new SelfPacedLearning(2f, 1.5f, true);
        assertArrayEquals(new float[] {0.9f, 0.55f, 0.25f, 0}, soft.weights(losses).toArray(), 1e-6f);
        SimpleMlp net = SimpleMlp.tanh(3, 2, 32, 3);
        Var per = SelfPacedLearning.perSampleCrossEntropy(net.forward(Var.of(DATA[0])), DATA[1]);
        assertEquals(Functional.crossEntropy(net.forward(Var.of(DATA[0])), DATA[1]).value().item(),
                per.value().mean().item(), 1e-5f);
    }

    /** A worker training a SimpleMlp replica on one shard. */
    private static LocalWorker worker(Tensor x, Tensor ids, long seed, float lr) {
        SimpleMlp net = SimpleMlp.tanh(seed, 2, 32, 32, 3);
        Optimizer opt = Optimizer.adamW(lr, 0.01f);
        return new LocalWorker() {
            @Override
            public List<Var> parameters() {
                return net.parameters();
            }

            @Override
            public int samples() {
                return x.shape().dim(0);
            }

            @Override
            public float train(int steps) {
                float sum = 0;
                for (int i = 0; i < steps; i++) {
                    Var l = Functional.crossEntropy(net.forward(Var.of(x)), ids);
                    l.backward();
                    sum += l.value().item();
                    opt.step(net.parameters());
                }
                return sum / steps;
            }
        };
    }

    @Test
    void fedAvgAndDiLoCoTrainAGlobalModelFromShards() {
        Tensor[] shuffled = Fixtures.spiral(60, 7);
        // shuffle rows so that shards are mixed
        int n = shuffled[1].shape().dim(0);
        float[] perm = new float[n];
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            idx.add(i);
        }
        java.util.Collections.shuffle(idx, new java.util.Random(1));
        for (int i = 0; i < n; i++) {
            perm[i] = idx.get(i);
        }
        Tensor p = Tensor.of(perm);
        Tensor x = shuffled[0].gather(0, p);
        Tensor ids = shuffled[1].gather(0, p);
        List<LocalWorker> workers = new ArrayList<>();
        for (int k = 0; k < 4; k++) {
            workers.add(worker(x.slice(0, k * n / 4, (k + 1) * n / 4), ids.slice(0, k * n / 4, (k + 1) * n / 4), 10 + k,
                    0.01f));
        }
        SimpleMlp global = SimpleMlp.tanh(5, 2, 32, 32, 3);
        DiLoCo diloco = new DiLoCo(global.parameters(), workers, DiLoCo.defaultOuter());
        for (int round = 0; round < 15; round++) {
            diloco.round(20);
        }
        double acc = LearningMath.accuracy(global.predict(DATA[0]), DATA[1]);
        System.out.println("DiLoCo global accuracy " + acc + " with " + diloco.transfers() + " transfers");
        assertTrue(acc > 0.85, "DiLoCo: " + acc);
        assertEquals(15 * 4 * 2, diloco.transfers(), "communication only once per 20 steps");

        SimpleMlp fedGlobal = SimpleMlp.tanh(5, 2, 32, 32, 3);
        FederatedAveraging fed = new FederatedAveraging(fedGlobal.parameters(), workers, 3);
        for (int round = 0; round < 15; round++) {
            fed.round(20, 1.0);
        }
        double fedAcc = LearningMath.accuracy(fedGlobal.predict(DATA[0]), DATA[1]);
        System.out.println("FedAvg global accuracy " + fedAcc);
        assertTrue(fedAcc > 0.8, "FedAvg: " + fedAcc);
    }
}
