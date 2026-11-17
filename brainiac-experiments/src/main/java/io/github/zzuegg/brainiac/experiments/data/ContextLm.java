package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The toy language model of the data experiments: an MLP that predicts each token from the
 * two before it (one-hot), so a document of length {@code L} gives {@code L - 2} training
 * examples. Small enough to train in a fraction of a second, yet a real autoregressive
 * model that can be sampled from and scored token by token.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ContextLm {

    private final int vocab;
    private final SimpleMlp net;

    /**
     * @param vocab the vocabulary size
     * @param hidden the hidden width
     * @param seed the seed of the weights
     */
    public ContextLm(int vocab, int hidden, long seed) {
        this.vocab = vocab;
        this.net = SimpleMlp.relu(seed, 2 * vocab, hidden, vocab);
    }

    /** @return the vocabulary size */
    public int vocab() {
        return vocab;
    }

    /** @return the network */
    public SimpleMlp net() {
        return net;
    }

    /**
     * The next-token examples of documents.
     *
     * @param vocab the vocabulary size
     * @param docs the documents
     * @return {@code {one-hot contexts [n, 2 vocab], next ids [n]}}
     */
    public static Tensor[] examples(int vocab, List<int[]> docs) {
        int n = 0;
        for (int[] d : docs) {
            n += Math.max(0, d.length - 2);
        }
        float[] x = new float[n * 2 * vocab];
        float[] y = new float[n];
        int i = 0;
        for (int[] d : docs) {
            for (int t = 2; t < d.length; t++) {
                x[i * 2 * vocab + d[t - 2]] = 1;
                x[i * 2 * vocab + vocab + d[t - 1]] = 1;
                y[i++] = d[t];
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2 * vocab), x), Tensor.of(Shape.of(n), y)};
    }

    /**
     * Rows of an example set.
     *
     * @param data {@code [n, ...]}
     * @param rows the rows
     * @return {@code [rows.length, ...]}
     */
    public static Tensor rows(Tensor data, int[] rows) {
        int n = data.shape().dim(0);
        int width = Math.toIntExact(data.size() / n);
        float[] all = data.toArray();
        float[] out = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(all, rows[i] * width, out, i * width, width);
        }
        int[] dims = data.shape().toArray();
        dims[0] = rows.length;
        return Tensor.of(Shape.of(dims), out);
    }

    /**
     * Trains on random mini-batches of examples with Adam.
     *
     * @param data the examples, as {@link #examples} returns them
     * @param steps the steps
     * @param batch the batch size
     * @param learningRate Adam's rate
     * @param seed the seed of the batches
     * @return this model
     */
    public ContextLm train(Tensor[] data, int steps, int batch, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int n = data[1].shape().dim(0);
        float[] x = data[0].toArray();
        float[] y = data[1].toArray();
        int w = 2 * vocab;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * w];
            float[] by = new float[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(n);
                System.arraycopy(x, k * w, bx, i * w, w);
                by[i] = y[k];
            }
            Functional.crossEntropy(net.forward(Var.of(Tensor.of(Shape.of(batch, w), bx))), Tensor.of(Shape.of(batch),
                    by)).backward();
            opt.step(net.parameters());
        }
        return this;
    }

    /**
     * Trains on documents.
     *
     * @param docs the documents
     * @param steps the steps
     * @param batch the batch size
     * @param learningRate Adam's rate
     * @param seed the seed of the batches
     * @return this model
     */
    public ContextLm train(List<int[]> docs, int steps, int batch, float learningRate, long seed) {
        return train(examples(vocab, docs), steps, batch, learningRate, seed);
    }

    /**
     * @param x one-hot contexts
     * @return the logits
     */
    public Tensor logits(Tensor x) {
        return net.predict(x);
    }

    /**
     * @param doc a document
     * @return the logits at every predicted position, {@code [L - 2, vocab]}
     */
    public Tensor logits(int[] doc) {
        return logits(examples(vocab, List.of(doc))[0]);
    }

    /**
     * @param docs documents
     * @return the mean next-token cross-entropy over all their predicted tokens
     */
    public float loss(List<int[]> docs) {
        Tensor[] e = examples(vocab, docs);
        return Functional.crossEntropy(Var.of(logits(e[0])), e[1]).value().item();
    }

    /**
     * Samples documents.
     *
     * @param n the documents
     * @param length their length
     * @param starts the first two tokens of each document are drawn from these documents
     * @param random the source of randomness
     * @return the samples
     */
    public List<int[]> sample(int n, int length, List<int[]> starts, SplittableRandom random) {
        List<int[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int[] s = starts.get(random.nextInt(starts.size()));
            int[] d = new int[length];
            d[0] = s[0];
            d[1] = s[1];
            out.add(d);
        }
        for (int t = 2; t < length; t++) {
            float[] x = new float[n * 2 * vocab];
            for (int i = 0; i < n; i++) {
                x[i * 2 * vocab + out.get(i)[t - 2]] = 1;
                x[i * 2 * vocab + vocab + out.get(i)[t - 1]] = 1;
            }
            float[] z = logits(Tensor.of(Shape.of(n, 2 * vocab), x)).toArray();
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < vocab; k++) {
                    max = Math.max(max, z[i * vocab + k]);
                }
                double[] p = new double[vocab];
                double sum = 0;
                for (int k = 0; k < vocab; k++) {
                    p[k] = Math.exp(z[i * vocab + k] - max);
                    sum += p[k];
                }
                double u = random.nextDouble() * sum;
                int k = 0;
                while (k < vocab - 1 && (u -= p[k]) > 0) {
                    k++;
                }
                out.get(i)[t] = k;
            }
        }
        return out;
    }

    /** @return copies of the weights, for a checkpoint */
    public List<Tensor> snapshot() {
        List<Tensor> s = new ArrayList<>();
        for (Var p : net.parameters()) {
            s.add(p.value().add(0f));
        }
        return s;
    }

    /**
     * Restores weights a {@link #snapshot} returned.
     *
     * @param weights the weights
     */
    public void load(List<Tensor> weights) {
        List<Var> p = net.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(weights.get(i));
        }
    }
}
