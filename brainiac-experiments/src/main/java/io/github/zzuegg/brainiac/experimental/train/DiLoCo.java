package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * DiLoCo, distributed low-communication training (Douillard et al. 2023,
 * <a href="https://arxiv.org/abs/2311.08105">arXiv:2311.08105</a>), simulated in-process.
 * Each round every worker starts from the global weights and takes {@code H} inner
 * optimizer steps (AdamW in the paper) on its own shard; the <em>pseudo-gradient</em>
 * {@code global - mean(worker weights)} is then applied to the global weights by an outer
 * optimizer, Nesterov momentum ({@code lr 0.7, momentum 0.9}). Workers communicate once
 * every {@code H} steps instead of every step, hundreds of times less, yet match fully
 * synchronous training: ideal for a home cluster of mixed machines.
 */
public final class DiLoCo {

    private final List<Var> global;
    private final List<LocalWorker> workers;
    private final Optimizer outer;
    private long transfers;

    /**
     * @param global the global model's parameters
     * @param workers the workers
     * @param outer the outer optimizer, for example {@code new Nesterov(0.7f, 0.9f)}
     */
    public DiLoCo(List<Var> global, List<LocalWorker> workers, Optimizer outer) {
        this.global = List.copyOf(global);
        this.workers = List.copyOf(workers);
        this.outer = outer;
    }

    /** @return the DiLoCo paper's outer optimizer: Nesterov with lr 0.7 and momentum 0.9 */
    public static Optimizer defaultOuter() {
        return new Nesterov(0.7f, 0.9f);
    }

    /** @return how many model copies were sent or received so far */
    public long transfers() {
        return transfers;
    }

    /**
     * One outer step.
     *
     * @param innerSteps the steps every worker trains, {@code H}
     * @return the workers' mean inner loss
     */
    public float round(int innerSteps) {
        double loss = 0;
        Tensor[] mean = new Tensor[global.size()];
        for (LocalWorker w : workers) {
            FederatedAveraging.broadcast(global, w.parameters());
            loss += w.train(innerSteps);
            transfers += 2;
            for (int i = 0; i < mean.length; i++) {
                Tensor v = w.parameters().get(i).value().div(workers.size());
                mean[i] = mean[i] == null ? v : mean[i].add(v);
            }
        }
        for (int i = 0; i < mean.length; i++) {
            Var g = global.get(i);
            g.setGrad(g.value().sub(mean[i]));   // the pseudo-gradient
        }
        outer.step(global);
        return (float) (loss / workers.size());
    }
}
