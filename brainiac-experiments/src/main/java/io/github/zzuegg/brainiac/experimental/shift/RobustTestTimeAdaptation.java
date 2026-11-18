package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.robust.TestTimeAdaptation;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
import io.github.zzuegg.brainiac.experimental.train.SharpnessAware;
import io.github.zzuegg.brainiac.nn.BatchNorm;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Continual and robust test-time adaptation, for long, non-stationary, small-batch or
 * label-shifted test streams where TENT ({@link TestTimeAdaptation}) drifts or collapses. All
 * three update online, batch after batch, and normalise with test-batch statistics where the
 * model has batch norm.
 *
 * <ul>
 *   <li>{@link Method#EATA}: entropy minimisation on the normalisation affine parameters, but
 *       only on reliable samples (entropy below {@code 0.4 ln K}) whose prediction differs from
 *       the moving average of recent ones (cosine below {@code epsilon}), each weighted by
 *       {@code exp(E0 - H)}, plus a Fisher-weighted pull to the source weights against
 *       forgetting (Niu et al., ICML 2022, <a href="https://arxiv.org/abs/2204.02610">arXiv:2204.02610</a>).</li>
 *   <li>{@link Method#SAR}: reliable-entropy filtering, sharpness-aware entropy minimisation
 *       through {@link SharpnessAware} (the reliable set is filtered again at the perturbed
 *       weights), and a model reset when the moving average of the entropy loss falls below a
 *       threshold, a sign of collapse (Niu et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2302.12400">arXiv:2302.12400</a>).</li>
 *   <li>{@link Method#COTTA}: a weight-averaged teacher ({@link MomentumTeacher}) gives
 *       pseudo-labels, averaged over augmentations when the frozen source model is unsure; the
 *       student (all parameters) learns them by cross-entropy, and a random share of weights
 *       is restored to the source values each step against forgetting (Wang et al., CVPR 2022,
 *       <a href="https://arxiv.org/abs/2203.13591">arXiv:2203.13591</a>). It predicts with the
 *       teacher and costs several forward passes per batch.</li>
 * </ul>
 */
public final class RobustTestTimeAdaptation {

    /** The adaptation. */
    public enum Method {
        /** efficient anti-forgetting TTA */
        EATA,
        /** sharpness-aware reliable entropy minimisation with resets */
        SAR,
        /** continual TTA with a mean teacher and stochastic restore */
        COTTA
    }

    private final Model model;
    private final Method method;
    private final List<Var> all;
    private final List<Var> buffers;
    private final List<Tensor> initial;
    private final List<Tensor> initialBuffers;
    private final List<Var> affine = new ArrayList<>();
    private float learningRate = 1e-3f;
    private double marginFactor = 0.4;
    private double redundancy = 0.4;
    private float fisherWeight = 50f;
    private float rho = 0.05f;
    private double resetFactor = 0.05;
    private float teacherMomentum = 0.99f;
    private double restoreProbability = 0.01;
    private double confidenceThreshold = 0.9;
    private int augmentations = 8;
    private UnaryOperator<Tensor> augment;
    private List<Tensor> fisher;
    private List<Tensor> anchorValues;
    private Optimizer optimizer;
    private SharpnessAware sam;
    private double[] average;
    private double lossAverage = Double.NaN;
    private int resets;
    private long adapted;
    private Model teacher;
    private Model anchor;
    private MomentumTeacher ema;
    private final SplittableRandom random = new SplittableRandom(7);

    private RobustTestTimeAdaptation(Model model, Method method) {
        this.model = Objects.requireNonNull(model);
        this.method = Objects.requireNonNull(method);
        this.all = model.parameters();
        this.buffers = model.buffers();
        this.initial = copies(all);
        this.initialBuffers = copies(buffers);
        for (Layer l : model.layers()) {
            if (l instanceof BatchNorm || l instanceof LayerNorm || l instanceof GroupNorm) {
                affine.addAll(l.parameters());
            }
        }
    }

    /**
     * Adapts a model.
     *
     * @param model the trained model, adapted in place
     * @param method the adaptation
     * @return the adapter
     */
    public static RobustTestTimeAdaptation of(Model model, Method method) {
        return new RobustTestTimeAdaptation(model, method);
    }

    /**
     * @param lr the step size (Adam for EATA and CoTTA, SGD with momentum for SAR)
     * @return this
     */
    public RobustTestTimeAdaptation learningRate(float lr) {
        this.learningRate = lr;
        this.optimizer = null;
        this.sam = null;
        return this;
    }

    /**
     * @param factor the entropy margin {@code E0 = factor ln K}, 0.4 in both papers
     * @param cosine EATA's redundancy threshold on the cosine to the moving-average prediction
     * @return this
     */
    public RobustTestTimeAdaptation reliability(double factor, double cosine) {
        this.marginFactor = factor;
        this.redundancy = cosine;
        return this;
    }

    /**
     * @param weight EATA's Fisher regulariser weight
     * @return this
     */
    public RobustTestTimeAdaptation fisherWeight(float weight) {
        this.fisherWeight = weight;
        return this;
    }

    /**
     * @param radius SAR's SAM radius
     * @param resetFraction SAR resets when the moving-average loss falls below {@code resetFraction ln K}
     * @return this
     */
    public RobustTestTimeAdaptation sar(float radius, double resetFraction) {
        this.rho = radius;
        this.resetFactor = resetFraction;
        this.sam = null;
        return this;
    }

    /**
     * @param momentum the teacher's EMA momentum
     * @param restore the per-weight probability of a restore to the source value
     * @param threshold the source confidence below which pseudo-labels are augmentation-averaged
     * @param copies the number of augmented copies
     * @param augmenter maps a batch to an augmented batch of the same shape
     * @return this
     */
    public RobustTestTimeAdaptation cotta(float momentum, double restore, double threshold, int copies,
            UnaryOperator<Tensor> augmenter) {
        this.teacherMomentum = momentum;
        this.restoreProbability = restore;
        this.confidenceThreshold = threshold;
        this.augmentations = copies;
        this.augment = augmenter;
        return this;
    }

    /**
     * Estimates EATA's diagonal Fisher information on the adapted parameters from unlabelled
     * in-distribution samples, using the model's own predictions as labels.
     *
     * @param samples in-distribution inputs
     * @return this
     */
    public RobustTestTimeAdaptation fisher(Tensor samples) {
        Var logits = model.forward(Var.of(samples), true);
        restoreBuffers();
        Var logP = logits.sub(logits.logsumexp(1, true));
        int n = samples.shape().dim(0);
        int[] arg = logits.value().argmax(1).toIntArray();
        float[] oh = new float[logP.value().toArray().length];
        int k = oh.length / n;
        for (int i = 0; i < n; i++) {
            oh[i * k + arg[i]] = 1;
        }
        Var nll = logP.mul(Tensor.of(Shape.of(n, k), oh)).sum().neg().div(n);
        List<Var> g = Var.gradients(nll, affine, false);
        fisher = new ArrayList<>();
        for (Var v : g) {
            fisher.add(v.value().square());
        }
        anchorValues = copies(affine);
        return this;
    }

    /** @return the number of SAR model resets so far */
    public int resets() {
        return resets;
    }

    /** @return the number of samples that took part in an update */
    public long adaptedSamples() {
        return adapted;
    }

    /**
     * Predicts a test batch, adapting on it.
     *
     * @param x the inputs
     * @return the logits
     */
    public Tensor predict(Tensor x) {
        return switch (method) {
            case EATA -> eata(x);
            case SAR -> sar(x);
            case COTTA -> cotta(x);
        };
    }

    private double margin(int classes) {
        return marginFactor * Math.log(classes);
    }

    private Tensor eata(Tensor x) {
        if (optimizer == null) {
            optimizer = Optimizer.adam(learningRate);
        }
        Var logits = model.forward(Var.of(x), true);
        restoreBuffers();
        int n = x.shape().dim(0);
        int k = logits.shape().dim(1);
        double e0 = margin(k);
        Var ent = TestTimeAdaptation.entropy(logits);
        double[] h = ent.value().toDoubleArray();
        double[] p = logits.value().sub(logits.value().logsumexp(1, true)).exp().toDoubleArray();
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (h[i] < e0 && (average == null || cosine(average, p, i * k, k) < redundancy)) {
                keep.add(i);
            }
        }
        // the moving average follows the reliable samples
        double[] mean = new double[k];
        int reliable = 0;
        for (int i = 0; i < n; i++) {
            if (h[i] < e0) {
                reliable++;
                for (int c = 0; c < k; c++) {
                    mean[c] += p[i * k + c];
                }
            }
        }
        if (reliable > 0) {
            for (int c = 0; c < k; c++) {
                mean[c] /= reliable;
            }
            if (average == null) {
                average = mean;
            } else {
                for (int c = 0; c < k; c++) {
                    average[c] = 0.9 * average[c] + 0.1 * mean[c];
                }
            }
        }
        if (!keep.isEmpty()) {
            float[] rows = new float[keep.size()];
            float[] w = new float[keep.size()];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = keep.get(i);
                w[i] = (float) Math.exp(e0 - h[keep.get(i)]);
            }
            Var loss = ent.gather(0, Tensor.of(Shape.of(rows.length), rows)).mul(Tensor.of(Shape.of(w.length), w))
                    .mean();
            if (fisher != null) {
                for (int i = 0; i < affine.size(); i++) {
                    loss = loss.add(affine.get(i).sub(anchorValues.get(i)).square().mul(fisher.get(i)).sum()
                            .mul(fisherWeight));
                }
            }
            step(loss, affine, optimizer);
            adapted += keep.size();
        }
        return logits.value();
    }

    private Tensor sar(Tensor x) {
        if (sam == null) {
            sam = SharpnessAware.sam(Optimizer.momentum(learningRate, 0.9f), rho);
        }
        Tensor out = Var.noGrad(() -> {
            Tensor o = model.forward(Var.of(x), true).value();
            restoreBuffers();
            return o;
        });
        int k = out.shape().dim(1);
        double e0 = margin(k);
        Var first = reliableEntropy(x, e0);
        if (first == null) {
            return out;
        }
        for (Var p : all) {
            p.zeroGrad();
        }
        boolean[] firstCall = {true};
        float loss = sam.step(affine, () -> {
            Var l = firstCall[0] ? first : reliableEntropy(x, e0);
            firstCall[0] = false;
            if (l == null) {
                return Var.of(Tensor.scalar(0));
            }
            l.backward();
            return l;
        });
        for (Var p : all) {
            p.zeroGrad();
        }
        adapted += x.shape().dim(0);
        lossAverage = Double.isNaN(lossAverage) ? loss : 0.9 * lossAverage + 0.1 * loss;
        if (lossAverage < resetFactor * Math.log(k)) {
            restoreParameters();
            sam = null;
            lossAverage = Double.NaN;
            resets++;
        }
        return out;
    }

    /** The mean entropy over the reliable rows, or {@code null} if none is reliable. */
    private Var reliableEntropy(Tensor x, double e0) {
        Var logits = model.forward(Var.of(x), true);
        restoreBuffers();
        Var ent = TestTimeAdaptation.entropy(logits);
        double[] h = ent.value().toDoubleArray();
        List<Float> rows = new ArrayList<>();
        for (int i = 0; i < h.length; i++) {
            if (h[i] < e0) {
                rows.add((float) i);
            }
        }
        if (rows.isEmpty()) {
            return null;
        }
        float[] r = new float[rows.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = rows.get(i);
        }
        return ent.gather(0, Tensor.of(Shape.of(r.length), r)).mean();
    }

    private Tensor cotta(Tensor x) {
        if (teacher == null) {
            teacher = model.copy(model.device());
            anchor = model.copy(model.device());
            ema = new MomentumTeacher(teacher.parameters(), all, teacherMomentum);
            optimizer = Optimizer.adam(learningRate);
        }
        Tensor anchorLogits = Var.noGrad(() -> anchor.forward(Var.of(x), false).value());
        double meanConfidence = 0;
        double[] conf = anchorLogits.max(1, false).sub(anchorLogits.logsumexp(1, false)).exp().toDoubleArray();
        for (double c : conf) {
            meanConfidence += c / conf.length;
        }
        Tensor teacherProbs = softmax(batchStatistics(teacher, x));
        if (meanConfidence < confidenceThreshold && augment != null) {
            Tensor sum = teacherProbs;
            for (int a = 0; a < augmentations; a++) {
                Tensor aug = augment.apply(x);
                sum = sum.add(softmax(batchStatistics(teacher, aug)));
            }
            teacherProbs = sum.div(augmentations + 1);
        }
        Var logits = model.forward(Var.of(x), true);
        restoreBuffers();
        Var logP = logits.sub(logits.logsumexp(1, true));
        Var loss = logP.mul(teacherProbs).sum(1, false).mean().neg();
        step(loss, all, optimizer);
        ema.update();
        if (restoreProbability > 0) {
            for (int i = 0; i < all.size(); i++) {
                float[] v = all.get(i).value().toArray();
                float[] v0 = initial.get(i).toArray();
                for (int j = 0; j < v.length; j++) {
                    if (random.nextDouble() < restoreProbability) {
                        v[j] = v0[j];
                    }
                }
                all.get(i).assign(Tensor.of(all.get(i).value().shape(), v));
            }
        }
        adapted += x.shape().dim(0);
        return teacherProbs.add(1e-8f).log();
    }

    private void step(Var loss, List<Var> params, Optimizer opt) {
        for (Var p : all) {
            p.zeroGrad();
        }
        List<Var> g = Var.gradients(loss, params, false);
        for (int i = 0; i < g.size(); i++) {
            params.get(i).setGrad(g.get(i).value());
        }
        opt.step(params);
    }

    /** Restores the parameters, buffers and moving statistics the adapter started with. */
    public void reset() {
        restoreParameters();
        optimizer = null;
        sam = null;
        average = null;
        lossAverage = Double.NaN;
        teacher = null;
        anchor = null;
        ema = null;
    }

    private void restoreParameters() {
        for (int i = 0; i < all.size(); i++) {
            all.get(i).assign(initial.get(i).copy());
        }
        restoreBuffers();
    }

    private void restoreBuffers() {
        for (int i = 0; i < buffers.size(); i++) {
            buffers.get(i).assign(initialBuffers.get(i).copy());
        }
    }

    /** A forward pass with test-batch statistics that leaves the running statistics unchanged. */
    private static Tensor batchStatistics(Model m, Tensor x) {
        List<Var> b = m.buffers();
        List<Tensor> saved = copies(b);
        Tensor out = Var.noGrad(() -> m.forward(Var.of(x), true).value());
        for (int i = 0; i < b.size(); i++) {
            b.get(i).assign(saved.get(i));
        }
        return out;
    }

    private static double cosine(double[] a, double[] p, int off, int k) {
        double dot = 0;
        double na = 0;
        double np = 0;
        for (int c = 0; c < k; c++) {
            dot += a[c] * p[off + c];
            na += a[c] * a[c];
            np += p[off + c] * p[off + c];
        }
        return dot / Math.sqrt(na * np + 1e-12);
    }

    private static Tensor softmax(Tensor logits) {
        return logits.sub(logits.logsumexp(1, true)).exp();
    }

    private static List<Tensor> copies(List<Var> vars) {
        List<Tensor> out = new ArrayList<>();
        for (Var v : vars) {
            out.add(v.value().copy());
        }
        return out;
    }
}
