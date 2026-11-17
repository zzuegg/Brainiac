package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FastActionTokenizer;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * DeMo, decoupled momentum optimization (Peng, Quesnelle and Kingma 2024,
 * <a href="https://arxiv.org/abs/2411.19870">arXiv:2411.19870</a>): data-parallel training
 * that never all-reduces gradients. Every replica keeps its <em>own</em> momentum
 * {@code m = beta m + g}, splits it into chunks, takes each chunk's DCT, and extracts only
 * the {@code k} largest coefficients, the fast-moving components; these are removed from
 * the local momentum (the rest keeps accumulating) and are all that is exchanged. The
 * replicas average the decoded components and step with their sign, like signSGD, so all
 * replicas apply the same update and stay identical.
 *
 * <p>{@link #step(List)} is the single-replica optimizer (no exchange);
 * {@link #stepReplicas(List)} simulates the cluster. Bytes count every replica sending its
 * {@code k} coefficients per chunk (a float and a 16-bit index each) and receiving everybody
 * else's.
 */
public final class DecoupledMomentum implements Optimizer {

    private final float beta;
    private final int chunk;
    private final int topK;
    private final Map<Var, double[]> momentum = new IdentityHashMap<>();
    private float learningRate;
    private long bytes;

    /**
     * @param learningRate the sign-step size
     * @param beta the momentum decay, for example 0.9
     * @param chunk the DCT chunk length, for example 32
     * @param topK coefficients kept per chunk, for example 4
     */
    public DecoupledMomentum(float learningRate, float beta, int chunk, int topK) {
        if (topK < 1 || topK > chunk) {
            throw new IllegalArgumentException("need 1 <= topK <= chunk");
        }
        this.learningRate = learningRate;
        this.beta = beta;
        this.chunk = chunk;
        this.topK = topK;
    }

    @Override
    public void step(List<Var> parameters) {
        stepReplicas(List.of(parameters));
    }

    /**
     * One DeMo step over replicas whose gradients have been computed on their own batches.
     * Every replica receives the same update; gradients are cleared.
     *
     * @param replicas each replica's parameters, all in the same order with equal values
     */
    public void stepReplicas(List<List<Var>> replicas) {
        int count = replicas.size();
        for (int i = 0; i < replicas.get(0).size(); i++) {
            int size = Math.toIntExact(replicas.get(0).get(i).value().size());
            double[] shared = new double[size];
            for (List<Var> replica : replicas) {
                Var p = replica.get(i);
                double[] m = momentum.computeIfAbsent(p, k -> new double[size]);
                float[] g = p.grad() == null ? new float[size] : p.grad().toArray();
                for (int j = 0; j < size; j++) {
                    m[j] = beta * m[j] + g[j];
                }
                double[] fast = extract(m);
                for (int j = 0; j < size; j++) {
                    m[j] -= fast[j];
                    shared[j] += fast[j] / count;
                }
                long chunks = (size + chunk - 1) / chunk;
                bytes += chunks * topK * 6L * count;   // send own, receive the others' (count - 1)
            }
            for (List<Var> replica : replicas) {
                Var p = replica.get(i);
                float[] w = p.value().toArray();
                for (int j = 0; j < size; j++) {
                    w[j] -= learningRate * (float) Math.signum(shared[j]);
                }
                p.assign(Tensor.of(p.value().shape(), w));
                p.setGrad(null);
            }
        }
    }

    /** The k largest DCT coefficients of every chunk, decoded back. */
    private double[] extract(double[] m) {
        double[] out = new double[m.length];
        for (int from = 0; from < m.length; from += chunk) {
            int len = Math.min(chunk, m.length - from);
            double[] c = FastActionTokenizer.dct(java.util.Arrays.copyOfRange(m, from, from + len));
            int keep = Math.min(topK, len);
            boolean[] kept = new boolean[len];
            for (int r = 0; r < keep; r++) {
                int best = -1;
                for (int j = 0; j < len; j++) {
                    if (!kept[j] && (best < 0 || Math.abs(c[j]) > Math.abs(c[best]))) {
                        best = j;
                    }
                }
                kept[best] = true;
            }
            for (int j = 0; j < len; j++) {
                c[j] = kept[j] ? c[j] : 0;
            }
            double[] back = FastActionTokenizer.idct(c);
            System.arraycopy(back, 0, out, from, len);
        }
        return out;
    }

    /** @return bytes exchanged by all replicas so far */
    public long bytes() {
        return bytes;
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "DeMo(lr=" + learningRate + ", beta=" + beta + ", chunk=" + chunk + ", topK=" + topK + ")";
    }
}
