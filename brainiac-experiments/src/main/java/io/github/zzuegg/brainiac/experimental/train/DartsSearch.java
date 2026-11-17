package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * The bilevel loop of differentiable architecture search: alternately take a step on the
 * network weights with the training loss and a step on the architecture weights
 * ({@code alpha}, such as {@code DartsCell.architecture()}) with the validation loss. This
 * is first-order DARTS, which treats the weights as constant in the architecture step; it
 * is cheaper than the second-order version and in practice nearly as good.
 *
 * <p>Liu, Simonyan and Yang 2019, "DARTS: Differentiable Architecture Search",
 * <a href="https://arxiv.org/abs/1806.09055">arXiv:1806.09055</a>.
 */
public final class DartsSearch {

    /**
     * One epoch's losses.
     *
     * @param epoch the epoch
     * @param trainLoss the mean training loss
     * @param validLoss the mean validation loss
     */
    public record Epoch(int epoch, double trainLoss, double validLoss) {
    }

    private DartsSearch() {
    }

    /**
     * Runs the search.
     *
     * @param model the network, containing the search cells; its parameters are the weights
     * @param architecture the architecture weights
     * @param train data for the weight steps
     * @param valid data for the architecture steps
     * @param loss the loss
     * @param weightOptimizer the optimizer for the weights
     * @param architectureOptimizer the optimizer for the architecture weights
     * @param epochs passes over the training data
     * @param batchSize the batch size
     * @return one entry per epoch
     */
    public static List<Epoch> search(Model model, List<Var> architecture, Dataset train, Dataset valid, Loss loss,
            Optimizer weightOptimizer, Optimizer architectureOptimizer, int epochs, int batchSize) {
        List<Epoch> history = new ArrayList<>();
        List<Var> weights = model.parameters();
        for (int e = 0; e < epochs; e++) {
            List<Batch> tb = train.shuffle(e).batches(batchSize);
            List<Batch> vb = valid.shuffle(e + 7919).batches(batchSize);
            double tl = 0;
            double vl = 0;
            for (int i = 0; i < tb.size(); i++) {
                Batch b = tb.get(i);
                Var l = loss.compute(model.forward(Var.of(b.features()), true), b.targets());
                l.backward();
                tl += l.value().item();
                architecture.forEach(Var::zeroGrad);
                weightOptimizer.step(weights);

                Batch v = vb.get(i % vb.size());
                Var lv = loss.compute(model.forward(Var.of(v.features()), true), v.targets());
                lv.backward();
                vl += lv.value().item();
                weights.forEach(Var::zeroGrad);
                architectureOptimizer.step(architecture);
            }
            history.add(new Epoch(e, tl / tb.size(), vl / tb.size()));
        }
        return history;
    }
}
