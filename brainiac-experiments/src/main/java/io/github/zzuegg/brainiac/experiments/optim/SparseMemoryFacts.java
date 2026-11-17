package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.experimental.nn.ProductKeyMemory;
import io.github.zzuegg.brainiac.experimental.train.ElasticWeightConsolidation;
import io.github.zzuegg.brainiac.experimental.train.SparseMemoryFinetuning;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Continual learning of a fact stream, the setting of sparse memory fine-tuning
 * (<a href="https://arxiv.org/abs/2510.15103">arXiv:2510.15103</a>, a companion of
 * {@link ContinualLearningExperiment}). Facts are random {@code (subject, relation) ->
 * object} triples. A network {@code dense -> product-key memory (residual) -> dense} is
 * pretrained on background facts, then new facts arrive in small batches and are learned
 * one batch at a time by full fine-tuning, LoRA on the dense layers, full fine-tuning with
 * EWC, or {@link SparseMemoryFinetuning} of the top TF-IDF memory slots. Reported: accuracy
 * on all new facts (learning) and on the background facts (forgetting).
 */
public final class SparseMemoryFacts {
    private static final int SUBJECTS = 50;
    private static final int RELATIONS = 4;
    private static final int OBJECTS = 16;
    private static final int WIDTH = 64;

    private SparseMemoryFacts() {
    }

    /** The fine-tuning methods. */
    public enum Method {
        /** Adam on every parameter. */
        FULL,
        /** Rank-4 LoRA adapters on both dense layers; memory frozen. */
        LORA,
        /** Adam on every parameter plus an EWC penalty anchored after pretraining. */
        EWC,
        /** Only the top TF-IDF memory value rows, plain SGD. */
        SPARSE_MEMORY
    }

    /**
     * One method's outcome after the whole stream.
     *
     * @param newFacts accuracy on all new facts
     * @param background accuracy on the background facts, 1 before fine-tuning
     * @param backgroundBefore background accuracy right after pretraining
     */
    public record Outcome(double newFacts, double background, double backgroundBefore) {
    }

    private static final class FactNet {
        private final Dense in;
        private final ProductKeyMemory memory;
        private final Dense out;
        private Layer inLayer;
        private Layer outLayer;

        FactNet(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            in = new Dense(SUBJECTS + RELATIONS, WIDTH, Activation.TANH, Init.XAVIER, r);
            memory = new ProductKeyMemory(WIDTH, 32, 16, WIDTH, 2, 8, r);
            out = new Dense(WIDTH, OBJECTS, Activation.IDENTITY, Init.XAVIER, r);
            inLayer = in;
            outLayer = out;
        }

        Var logits(Tensor x) {
            Var h = inLayer.forward(Var.of(x));
            return outLayer.forward(h.add(memory.forward(h)));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(inLayer.parameters());
            p.addAll(memory.parameters());
            p.addAll(outLayer.parameters());
            return p;
        }
    }

    private record Facts(Tensor x, Tensor y, int[] objects) {
        int size() {
            return objects.length;
        }

        Facts slice(int from, int to) {
            return new Facts(x.slice(0, from, to), y.slice(0, from, to),
                    java.util.Arrays.copyOfRange(objects, from, to));
        }
    }

    private static Facts[] facts(int background, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int pairs = SUBJECTS * RELATIONS;
        int[] order = new int[pairs];
        for (int i = 0; i < pairs; i++) {
            order[i] = i;
        }
        for (int i = pairs - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        float[] x = new float[pairs * (SUBJECTS + RELATIONS)];
        float[] y = new float[pairs];
        int[] objects = new int[pairs];
        for (int i = 0; i < pairs; i++) {
            int subject = order[i] / RELATIONS;
            int relation = order[i] % RELATIONS;
            x[i * (SUBJECTS + RELATIONS) + subject] = 1;
            x[i * (SUBJECTS + RELATIONS) + SUBJECTS + relation] = 1;
            objects[i] = r.nextInt(OBJECTS);
            y[i] = objects[i];
        }
        Facts all = new Facts(Tensor.of(Shape.of(pairs, SUBJECTS + RELATIONS), x), Tensor.of(Shape.of(pairs), y),
                objects);
        return new Facts[] {all.slice(0, background), all.slice(background, pairs)};
    }

    /**
     * Pretrains, streams the new facts, and measures.
     *
     * @param method the fine-tuning method
     * @param batch new facts per batch
     * @param steps optimizer steps per batch
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(Method method, int batch, int steps, long seed) {
        Facts[] split = facts(100, seed);
        Facts background = split[0];
        Facts fresh = split[1];
        FactNet net = new FactNet(seed);
        Optimizer pre = Optimizer.adam(0.01f);
        for (int s = 0; s < 400; s++) {
            Functional.crossEntropy(net.logits(background.x()), background.y()).backward();
            pre.step(net.parameters());
        }
        double before = accuracy(net, background);
        ElasticWeightConsolidation ewc = null;
        SparseMemoryFinetuning sparse = null;
        Optimizer opt;
        switch (method) {
            case LORA -> {
                SplittableRandom r = new SplittableRandom(seed + 3);
                net.inLayer = LoraAdapterDense.lora(net.in, 4, 8, r);
                net.outLayer = LoraAdapterDense.lora(net.out, 4, 8, r);
                opt = Optimizer.adam(0.01f);
            }
            case EWC -> {
                ewc = new ElasticWeightConsolidation(net.parameters(), 2000f);
                List<Supplier<Var>> losses = new ArrayList<>();
                for (int i = 0; i < background.size(); i += 10) {
                    Facts b = background.slice(i, Math.min(background.size(), i + 10));
                    losses.add(() -> Functional.crossEntropy(net.logits(b.x()), b.y()));
                }
                ewc.consolidate(losses);
                opt = Optimizer.adam(0.01f);
            }
            case SPARSE_MEMORY -> {
                sparse = new SparseMemoryFinetuning(net.memory.values(), 64, Optimizer.sgd(4f));
                for (int i = 0; i < background.size(); i++) {
                    sparse.recordBackground(accessed(net, background.slice(i, i + 1)));
                }
                opt = sparse;
            }
            default -> opt = Optimizer.adam(0.01f);
        }
        List<Var> trained = method == Method.LORA ? loraParameters(net) : net.parameters();
        for (int from = 0; from < fresh.size(); from += batch) {
            Facts b = fresh.slice(from, Math.min(fresh.size(), from + batch));
            if (sparse != null) {
                List<int[]> rows = new ArrayList<>();
                for (int i = 0; i < b.size(); i++) {
                    rows.add(accessed(net, b.slice(i, i + 1)));
                }
                sparse.select(rows);
            }
            for (int s = 0; s < steps; s++) {
                Var loss = Functional.crossEntropy(net.logits(b.x()), b.y());
                if (ewc != null) {
                    loss = loss.add(ewc.penalty());
                }
                loss.backward();
                opt.step(trained);
                for (Var p : net.parameters()) {
                    p.zeroGrad();
                }
            }
        }
        return new Outcome(accuracy(net, fresh), accuracy(net, background), before);
    }

    private static List<Var> loraParameters(FactNet net) {
        List<Var> p = new ArrayList<>(net.inLayer.parameters());
        p.addAll(net.outLayer.parameters());
        return p;
    }

    private static int[] accessed(FactNet net, Facts one) {
        for (Var p : net.parameters()) {
            p.zeroGrad();
        }
        Functional.crossEntropy(net.logits(one.x()), one.y()).backward();
        int[] rows = SparseMemoryFinetuning.accessedRows(net.memory.values());
        for (Var p : net.parameters()) {
            p.zeroGrad();
        }
        return rows;
    }

    private static double accuracy(FactNet net, Facts facts) {
        float[] l = net.logits(facts.x()).value().toArray();
        int correct = 0;
        for (int i = 0; i < facts.size(); i++) {
            int best = 0;
            for (int c = 1; c < OBJECTS; c++) {
                best = l[i * OBJECTS + c] > l[i * OBJECTS + best] ? c : best;
            }
            correct += best == facts.objects()[i] ? 1 : 0;
        }
        return (double) correct / facts.size();
    }
}
