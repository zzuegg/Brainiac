package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * kNN-LM (Khandelwal et al. 2020, <a href="https://arxiv.org/abs/1911.00172">arXiv:1911.00172</a>):
 * a datastore maps the model's own hidden state before the output layer to the token that
 * followed it, and at generation time the model's next-token distribution is mixed with a
 * distribution over the tokens of the {@code k} nearest stored states:
 * {@code p = lambda p_kNN + (1 - lambda) p_LM}, {@code p_kNN(w) ~ sum_(neighbours with w) exp(-d / T)}.
 * Knowledge is added by adding rows, not by training: retrieval as attention over an
 * external memory, queried by the hidden state.
 *
 * <p>{@link #hiddenStates} works with a core sequential language model ({@code Model} with
 * token ids in and probabilities out): it runs every layer but the last.
 */
public final class KnnLm {

    private final KnnIndex index;
    private final int vocabulary;
    private int k = 8;
    private float temperature = 1;

    private KnnLm(KnnIndex index, int vocabulary) {
        this.index = index;
        this.vocabulary = vocabulary;
    }

    /**
     * An empty datastore.
     *
     * @param hiddenSize the hidden-state size
     * @param vocabulary the vocabulary size
     * @return the datastore
     */
    public static KnnLm create(int hiddenSize, int vocabulary) {
        return new KnnLm(KnnIndex.create(hiddenSize, 1, KnnIndex.Metric.L2), vocabulary);
    }

    /**
     * @param neighbours the neighbours retrieved
     * @param temp the softmax temperature over negative squared distances
     * @return this datastore
     */
    public KnnLm search(int neighbours, float temp) {
        this.k = neighbours;
        this.temperature = temp;
        return this;
    }

    /** @return the stored rows */
    public int size() {
        return index.size();
    }

    /**
     * Stores one hidden state and the token that followed.
     *
     * @param hidden the hidden state
     * @param next the next token
     */
    public void add(float[] hidden, int next) {
        index.add(hidden, new float[] {next});
    }

    /**
     * Stores every position of a text: its hidden state and the token after it.
     *
     * @param model the language model
     * @param ids the text
     */
    public void addText(Model model, int[] ids) {
        float[][] h = hiddenStates(model, ids);
        for (int t = 0; t + 1 < ids.length; t++) {
            add(h[t], ids[t + 1]);
        }
    }

    /**
     * The hidden states of a sequential language model before its output layer, each
     * position seeing at most the model's window of tokens before it.
     *
     * @param model a chain whose last layer maps hidden states to the vocabulary
     * @param ids the tokens
     * @return {@code [ids.length][hidden]}
     */
    public static float[][] hiddenStates(Model model, int[] ids) {
        int window = model.inputShape().dim(0);
        List<Layer> layers = model.layers();
        float[][] out = new float[ids.length][];
        for (int t = 0; t < ids.length; t++) {
            int from = Math.max(0, t + 1 - window);
            int n = t + 1 - from;
            float[] x = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = ids[from + i];
            }
            Var h = Var.of(Tensor.of(Shape.of(1, n), x));
            for (int l = 0; l < layers.size() - 1; l++) {
                h = layers.get(l).forward(h, false);
            }
            int size = h.shape().dim(-1);
            out[t] = h.value().slice(1, n - 1, n).reshape(size).toArray();
        }
        return out;
    }

    /**
     * The model's own next-token distribution at the last position of a context.
     *
     * @param model the language model
     * @param ids the context
     * @return the probabilities
     */
    public static float[] modelProbabilities(Model model, int[] ids) {
        int window = model.inputShape().dim(0);
        int from = Math.max(0, ids.length - window);
        int n = ids.length - from;
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = ids[from + i];
        }
        Tensor p = model.forward(Var.of(Tensor.of(Shape.of(1, n), x)), false).value();
        int v = p.shape().dim(-1);
        return p.slice(1, n - 1, n).reshape(v).toArray();
    }

    /**
     * The kNN distribution for a hidden state.
     *
     * @param hidden the query
     * @return probabilities over the vocabulary; all zero for an empty datastore
     */
    public float[] knnProbabilities(float[] hidden) {
        float[] p = new float[vocabulary];
        KnnIndex.Hit[] hits = index.search(hidden, k);
        if (hits.length == 0) {
            return p;
        }
        double max = hits[0].score() / temperature;
        double sum = 0;
        for (KnnIndex.Hit h : hits) {
            double w = Math.exp(h.score() / temperature - max);
            p[(int) index.value(h.row())[0]] += (float) w;
            sum += w;
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= (float) sum;
        }
        return p;
    }

    /**
     * Mixes the model's distribution with the kNN one.
     *
     * @param model the model's probabilities
     * @param hidden the hidden state that produced them
     * @param lambda the kNN share
     * @return the mixed probabilities
     */
    public float[] interpolate(float[] model, float[] hidden, float lambda) {
        if (index.size() == 0) {
            return model.clone();
        }
        float[] knn = knnProbabilities(hidden);
        float[] p = new float[model.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = lambda * knn[i] + (1 - lambda) * model[i];
        }
        return p;
    }

    /**
     * The perplexity of a text under the model with and without retrieval.
     *
     * @param model the language model
     * @param ids the text
     * @param lambda the kNN share
     * @return {@code {model perplexity, kNN-LM perplexity}}
     */
    public double[] perplexity(Model model, int[] ids, float lambda) {
        float[][] h = hiddenStates(model, ids);
        double nllModel = 0;
        double nllMixed = 0;
        for (int t = 0; t + 1 < ids.length; t++) {
            float[] pm = modelProbabilities(model, Arrays.copyOf(ids, t + 1));
            float[] px = interpolate(pm, h[t], lambda);
            nllModel -= Math.log(Math.max(1e-12, pm[ids[t + 1]]));
            nllMixed -= Math.log(Math.max(1e-12, px[ids[t + 1]]));
        }
        int n = ids.length - 1;
        return new double[] {Math.exp(nllModel / n), Math.exp(nllMixed / n)};
    }

    /**
     * Greedy generation with retrieval.
     *
     * @param model the language model
     * @param prompt the prompt
     * @param tokens the tokens to generate
     * @param lambda the kNN share
     * @return the prompt followed by the generated tokens
     */
    public int[] generate(Model model, int[] prompt, int tokens, float lambda) {
        int[] ids = Arrays.copyOf(prompt, prompt.length + tokens);
        for (int t = prompt.length; t < ids.length; t++) {
            int[] ctx = Arrays.copyOf(ids, t);
            float[][] h = hiddenStates(model, ctx);
            float[] p = interpolate(modelProbabilities(model, ctx), h[t - 1], lambda);
            int best = 0;
            for (int i = 1; i < p.length; i++) {
                if (p[i] > p[best]) {
                    best = i;
                }
            }
            ids[t] = best;
        }
        return ids;
    }

    /**
     * Saves the datastore.
     *
     * @param path the file
     * @throws IOException if writing fails
     */
    public void save(Path path) throws IOException {
        index.save(path);
    }

    /**
     * Loads a datastore {@link #save} wrote.
     *
     * @param path the file
     * @param vocabulary the vocabulary size
     * @return the datastore
     * @throws IOException if reading fails
     */
    public static KnnLm load(Path path, int vocabulary) throws IOException {
        return new KnnLm(KnnIndex.load(path), vocabulary);
    }

    @Override
    public String toString() {
        return "KnnLm(" + index.size() + " rows, k " + k + ")";
    }
}
