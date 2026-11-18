package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ReservoirReplay;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Replay with asymmetric losses for online class-incremental learning (one pass over the stream,
 * a small replay buffer).
 *
 * <p>When new classes arrive, plain experience replay trains the incoming samples with a softmax
 * over all classes, whose gradient pushes every old class's logit down; old-class prototypes then
 * drift abruptly. ER-ACE (asymmetric cross-entropy) computes the incoming batch's loss with a
 * softmax over only the classes present in that batch, so new data cannot push old logits down,
 * while the replayed samples keep the full softmax:
 * <pre>
 * L = CE_{classes in B_in}(f(x_in), y_in) + CE_{all}(f(x_buf), y_buf)
 * </pre>
 * ER-AML (asymmetric metric learning) instead trains the incoming samples with a contrastive
 * loss against buffered samples (a positive of the same class, negatives of other classes), and
 * the replayed samples with cross-entropy. The buffer is core-experimental
 * {@link ReservoirReplay} (reservoir sampling over the stream).
 *
 * <p>Paper: Caccia et al., "New Insights on Reducing Abrupt Representation Change in Online
 * Continual Learning", ICLR 2022, <a href="https://arxiv.org/abs/2104.05025">arXiv:2104.05025</a>.
 */
public final class AsymmetricReplay {

    private final ReservoirReplay buffer;
    private final int classes;

    /**
     * Creates the replay memory.
     *
     * @param capacity buffer size in samples
     * @param classes the total number of classes
     * @param seed the seed
     */
    public AsymmetricReplay(int capacity, int classes, long seed) {
        this.buffer = new ReservoirReplay(capacity, seed);
        this.classes = classes;
    }

    /** @return the buffer */
    public ReservoirReplay buffer() {
        return buffer;
    }

    /**
     * Cross-entropy with a softmax restricted to some classes: logits of the other classes are
     * pushed to {@code -1e4} so they receive no gradient.
     *
     * @param logits {@code [rows, classes]}
     * @param ids {@code [rows]} labels
     * @param allowed which classes take part
     * @return the mean loss
     */
    public static Var maskedCrossEntropy(Var logits, Tensor ids, boolean[] allowed) {
        int c = logits.shape().dim(1);
        float[] mask = new float[c];
        float[] off = new float[c];
        for (int i = 0; i < c; i++) {
            mask[i] = allowed[i] ? 1 : 0;
            off[i] = allowed[i] ? 0 : -1e4f;
        }
        Var masked = logits.mul(Tensor.of(Shape.of(c), mask)).add(Tensor.of(Shape.of(c), off));
        return Functional.crossEntropy(masked, ids);
    }

    /**
     * The classes present in a batch.
     *
     * @param ids labels
     * @param classes the number of classes
     * @return a mask
     */
    public static boolean[] present(Tensor ids, int classes) {
        boolean[] p = new boolean[classes];
        for (float v : ids.toArray()) {
            p[(int) v] = true;
        }
        return p;
    }

    /**
     * ER-ACE's loss for one step: asymmetric cross-entropy on the incoming batch, full
     * cross-entropy on a replayed batch of the same size (none while the buffer is empty).
     *
     * @param model maps inputs {@code [rows, features]} to logits
     * @param x incoming inputs
     * @param ids incoming labels
     * @return the loss
     */
    public Var aceLoss(java.util.function.Function<Var, Var> model, Tensor x, Tensor ids) {
        Var loss = maskedCrossEntropy(model.apply(Var.of(x)), ids, present(ids, classes));
        if (buffer.size() > 0) {
            Tensor[] old = buffer.sample(ids.shape().dim(0));
            loss = loss.add(Functional.crossEntropy(model.apply(Var.of(old[0])), old[1].reshape(old[1].shape().dim(0))));
        }
        return loss;
    }

    /**
     * ER-AML's loss for one step: for every incoming sample, a softmax contrastive loss over
     * cosine similarities (temperature 0.2) between its features and buffered samples' features,
     * the same-class ones positive; plus cross-entropy on a replayed batch. Incoming samples whose
     * class is not in the buffer sample use cross-entropy over the present classes instead.
     *
     * @param features maps inputs to feature vectors
     * @param head maps features to logits
     * @param x incoming inputs
     * @param ids incoming labels
     * @return the loss
     */
    public Var amlLoss(java.util.function.Function<Var, Var> features, java.util.function.Function<Var, Var> head,
            Tensor x, Tensor ids) {
        int rows = ids.shape().dim(0);
        Var hIn = features.apply(Var.of(x));
        if (buffer.size() == 0) {
            return maskedCrossEntropy(head.apply(hIn), ids, present(ids, classes));
        }
        int pool = Math.min(64, Math.max(rows * 4, 16));
        Tensor[] mem = buffer.sample(pool);
        float[] memY = mem[1].toArray();
        Var hMem = features.apply(Var.of(mem[0]));
        Var a = normalise(hIn);
        Var b = normalise(hMem);
        Var sim = a.matmul(b.transpose()).mul(1 / 0.2f);   // [rows, pool]
        float[] y = ids.toArray();
        float[] pos = new float[rows * pool];
        float[] has = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < pool; j++) {
                if (memY[j] == y[i]) {
                    pos[i * pool + j] = 1;
                    has[i] = 1;
                }
            }
        }
        float count = 0;
        for (float h : has) {
            count += h;
        }
        Var loss;
        if (count > 0) {
            Var logp = Activation.LOG_SOFTMAX.forward(sim);
            // mean log-probability of the positives per row, averaged over rows that have one
            Tensor posT = Tensor.of(Shape.of(rows, pool), pos);
            Var perRow = logp.mul(posT).sum(1, false).div(posT.sum(1).add(1e-6f));
            loss = perRow.mul(Tensor.of(Shape.of(rows), has)).sum().mul(-1f / count);
        } else {
            loss = maskedCrossEntropy(head.apply(hIn), ids, present(ids, classes));
        }
        Tensor[] old = buffer.sample(rows);
        return loss.add(Functional.crossEntropy(head.apply(features.apply(Var.of(old[0]))),
                old[1].reshape(old[1].shape().dim(0))));
    }

    private static Var normalise(Var h) {
        return h.div(h.square().sum(1, true).add(1e-8f).sqrt());
    }

    /**
     * Adds the incoming batch to the buffer (reservoir sampling).
     *
     * @param x inputs
     * @param ids labels
     */
    public void remember(Tensor x, Tensor ids) {
        buffer.add(x, ids.reshape(ids.shape().dim(0), 1));
    }
}
