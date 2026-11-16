package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Several designs that share their region order, outputs and ticks, stacked into one layer
 * so they train together: every weight matrix gets a leading population axis, each
 * projection is one batched matrix product for all candidates, and a candidate's missing
 * connections, neurons and projections are masked to zero. Gradients of the candidates
 * never mix, so with an element-wise optimizer (SGD, momentum, Adam) each candidate trains
 * as it would alone, while a GPU runs the whole population in the kernels of one model.
 *
 * <p>Each candidate's learning rate is folded into its weights: they are stored as
 * {@code w = s * u} and the optimizer updates {@code u} at a shared base rate, which moves
 * {@code w} at the candidate's own rate ({@code s = rate / base} for Adam, its square root
 * for SGD and momentum).
 *
 * <p>Takes input {@code [batch, inputs]} and returns {@code [batch, candidates * outputs]},
 * candidate by candidate; {@link #loss} scores that against the plain targets.
 */
final class Population implements Layer {

    private final List<Genome> genomes;
    private final int count;
    private final int inputs;
    private final int outputs;
    private final List<String> regions;
    private final int[] width;
    private final List<String[]> projections = new ArrayList<>();
    private final Var[] weights;
    private final Var[] masks;
    private final Var[] biases;
    private final List<List<Activation>> activations = new ArrayList<>();
    private final List<List<Var>> activationMasks = new ArrayList<>();
    /** Per region with switching neurons (else null): its settings and stacked parameters. */
    private final Brain.Switching[] switching;
    private final Var[] controls;
    private final Var[] controlBias;
    private final List<List<Var>> switchScale = new ArrayList<>();
    private final List<List<Var>> switchOffset = new ArrayList<>();
    private final Var[] switchMasks;
    private final Var headWeight;
    private final Var headBias;
    private final Var scale;
    private final Activation head;
    private final int ticks;

    /**
     * Stacks built models.
     *
     * @param genomes the designs; same region order, outputs and ticks
     * @param models their models, built by {@link Genome#build} without a front end, with
     *     the starting weights to train from
     * @param scales each candidate's learning rate factor {@code s}
     * @param head the output activation
     * @param palette activations every mixed region computes, as in {@link Genome#brainSpec(java.util.Collection)}
     */
    Population(List<Genome> genomes, List<Model> models, float[] scales, Activation head,
            java.util.Collection<Activation> palette) {
        this.genomes = List.copyOf(genomes);
        this.count = genomes.size();
        Genome first = genomes.getFirst();
        this.inputs = first.inputs();
        this.ticks = first.ticks();
        this.head = Objects.requireNonNull(head, "head");
        this.regions = first.regions().stream().map(Genome.Region::name).toList();
        for (Genome g : genomes) {
            if (!g.regions().stream().map(Genome.Region::name).toList().equals(regions)
                    || !g.outputs().equals(first.outputs()) || g.ticks() != ticks || g.inputs() != inputs) {
                throw new IllegalArgumentException("stacked designs need the same regions, outputs and ticks");
            }
        }
        switching = new Brain.Switching[regions.size()];
        for (int r = 0; r < regions.size(); r++) {
            for (Genome g : genomes) {
                Brain.Switching sw = g.regions().get(r).switching();
                if (sw != null && switching[r] != null && !sw.equals(switching[r])) {
                    throw new IllegalArgumentException("stacked designs need the same switching settings per region");
                }
                switching[r] = sw != null ? sw : switching[r];
            }
        }
        width = new int[regions.size()];
        for (int r = 0; r < width.length; r++) {
            for (Genome g : genomes) {
                width[r] = Math.max(width[r], g.regions().get(r).neurons().size());
            }
        }
        // the union of all candidates' projections, in order of first appearance
        for (Genome g : genomes) {
            for (Genome.Projection p : g.projections()) {
                if (p.size() > 0 && projections.stream().noneMatch(q -> q[0].equals(p.from()) && q[1].equals(p.to()))) {
                    projections.add(new String[] {p.from(), p.to()});
                }
            }
        }
        float[] s = scales.clone();
        scale = Var.of(Tensor.of(Shape.of(count, 1, 1), s));

        weights = new Var[projections.size()];
        masks = new Var[projections.size()];
        controls = new Var[projections.size()];
        for (int j = 0; j < projections.size(); j++) {
            String from = projections.get(j)[0];
            String to = projections.get(j)[1];
            int rows = size(from);
            int cols = size(to);
            float[] w = new float[count * rows * cols];
            float[] m = new float[w.length];
            for (int c = 0; c < count; c++) {
                Genome g = genomes.get(c);
                int index = brainIndex(g, from, to);
                if (index < 0) {
                    continue;
                }
                Brain brain = Genome.brainOf(models.get(c));
                boolean[] mask = brain.connectionMask(index);
                float[] values = brain.weightValues(index);
                int gc = own(g, to);
                int gr = mask.length / gc;
                for (int a = 0; a < gr; a++) {
                    for (int b = 0; b < gc; b++) {
                        if (mask[a * gc + b]) {
                            int k = (c * rows + a) * cols + b;
                            m[k] = 1;
                            w[k] = values[a * gc + b] / s[c];
                        }
                    }
                }
            }
            weights[j] = Var.param(Tensor.of(Shape.of(count, rows, cols), w));
            masks[j] = Var.of(Tensor.of(Shape.of(count, rows, cols), m));
            if (switching[regions.indexOf(to)] != null) {
                float[] v = new float[count * rows * cols];
                for (int c = 0; c < count; c++) {
                    Genome g = genomes.get(c);
                    int index = brainIndex(g, from, to);
                    if (index < 0 || g.region(to).switching() == null) {
                        continue;   // no control weights: they stay 0, and so do their gradients' effects
                    }
                    float[] values = Genome.brainOf(models.get(c)).controlValues(index);
                    int gc = own(g, to);
                    int gr = values.length / gc;
                    for (int a = 0; a < gr; a++) {
                        for (int b = 0; b < gc; b++) {
                            v[(c * rows + a) * cols + b] = values[a * gc + b] / s[c];
                        }
                    }
                }
                controls[j] = Var.param(Tensor.of(Shape.of(count, rows, cols), v));
            }
        }
        controlBias = new Var[regions.size()];
        switchMasks = new Var[regions.size()];
        for (int r = 0; r < regions.size(); r++) {
            switchScale.add(null);
            switchOffset.add(null);
            if (switching[r] == null) {
                continue;
            }
            int n = width[r];
            int kk = switching[r].candidates().size();
            float[] cb = new float[count * n];
            float[][] u = new float[kk][count * n];
            float[][] e = new float[kk][count * n];
            float[] on = new float[count * n];
            for (int c = 0; c < count; c++) {
                Genome.Region region = genomes.get(c).regions().get(r);
                if (region.switching() == null) {
                    continue;
                }
                float[][] own = Genome.brainOf(models.get(c)).switchingValues(region.name());
                int gn = region.neurons().size();
                for (int i = 0; i < gn; i++) {
                    on[c * n + i] = region.neurons().get(i).switches() ? 1 : 0;
                    cb[c * n + i] = own[0][i] / s[c];
                    for (int k = 0; k < kk; k++) {
                        u[k][c * n + i] = own[1][k * gn + i] / s[c];
                        e[k][c * n + i] = own[2][k * gn + i] / s[c];
                    }
                }
            }
            controlBias[r] = Var.param(Tensor.of(Shape.of(count, 1, n), cb));
            List<Var> us = new ArrayList<>();
            List<Var> es = new ArrayList<>();
            for (int k = 0; k < kk; k++) {
                us.add(Var.param(Tensor.of(Shape.of(count, 1, n), u[k])));
                es.add(Var.param(Tensor.of(Shape.of(count, 1, n), e[k])));
            }
            switchScale.set(r, us);
            switchOffset.set(r, es);
            switchMasks[r] = Var.of(Tensor.of(Shape.of(count, 1, n), on));
        }

        biases = new Var[regions.size()];
        for (int r = 0; r < regions.size(); r++) {
            int n = width[r];
            float[] b = new float[count * n];
            List<Activation> distinct = new ArrayList<>();
            Map<Activation, float[]> am = new HashMap<>();
            for (int c = 0; c < count; c++) {
                Genome.Region region = genomes.get(c).regions().get(r);
                float[] values = Genome.brainOf(models.get(c)).biasValues(region.name());
                for (int i = 0; i < region.neurons().size(); i++) {
                    b[c * n + i] = values[i] / s[c];
                    if (region.neurons().get(i).switches()) {
                        continue;   // computed by the switching part
                    }
                    Activation a = region.neurons().get(i).activation();
                    if (!am.containsKey(a)) {
                        distinct.add(a);
                        am.put(a, new float[count * n]);
                    }
                    am.get(a)[c * n + i] = 1;
                }
            }
            biases[r] = Var.param(Tensor.of(Shape.of(count, 1, n), b));
            if (distinct.size() > 1) {
                // the palette in a fixed order, as in Genome.brainSpec, so every generation
                // runs the same generated kernels
                for (Activation a : palette) {
                    if (!am.containsKey(a)) {
                        distinct.add(a);
                        am.put(a, new float[count * n]);
                    }
                }
                distinct.sort(java.util.Comparator.comparing(Activation::toString));
            }
            activations.add(distinct);
            List<Var> ms = new ArrayList<>();
            for (Activation a : distinct) {
                ms.add(Var.of(Tensor.of(Shape.of(count, 1, n), am.get(a))));
            }
            activationMasks.add(ms);
        }

        int fed = 0;
        for (String o : first.outputs()) {
            fed += width[regions.indexOf(o)];
        }
        Dense d0 = (Dense) models.getFirst().layers().getLast();
        outputs = Math.toIntExact(d0.parameters().get(1).value().shape().size());
        float[] hw = new float[count * fed * outputs];
        float[] hb = new float[count * outputs];
        for (int c = 0; c < count; c++) {
            Dense d = (Dense) models.get(c).layers().getLast();
            float[] w = d.parameters().get(0).value().to(DType.FLOAT32).toArray();
            float[] b = d.parameters().get(1).value().to(DType.FLOAT32).toArray();
            int row = 0;       // row in the candidate's own head
            int padded = 0;    // row in the stacked head
            for (String o : first.outputs()) {
                int own = genomes.get(c).region(o).neurons().size();
                for (int i = 0; i < own; i++) {
                    System.arraycopy(w, (row + i) * outputs, hw, ((c * fed) + padded + i) * outputs, outputs);
                }
                row += own;
                padded += width[regions.indexOf(o)];
            }
            for (int k = 0; k < outputs; k++) {
                hb[c * outputs + k] = b[k];
            }
            for (int k = c * fed * outputs; k < (c + 1) * fed * outputs; k++) {
                hw[k] /= s[c];
            }
            for (int k = 0; k < outputs; k++) {
                hb[c * outputs + k] /= s[c];
            }
        }
        headWeight = Var.param(Tensor.of(Shape.of(count, fed, outputs), hw));
        headBias = Var.param(Tensor.of(Shape.of(count, 1, outputs), hb));
    }

    private int size(String region) {
        return region.equals(Genome.INPUT) ? inputs : width[regions.indexOf(region)];
    }

    private static int own(Genome g, String region) {
        return region.equals(Genome.INPUT) ? g.inputs() : g.region(region).neurons().size();
    }

    /** The index of a projection in the candidate's own brain, or -1 if it has none. */
    private static int brainIndex(Genome g, String from, String to) {
        int j = 0;
        for (Genome.Projection p : g.projections()) {
            if (p.size() == 0) {
                continue;
            }
            if (p.from().equals(from) && p.to().equals(to)) {
                return j;
            }
            j++;
        }
        return -1;
    }

    /** @return how many candidates are stacked */
    int count() {
        return count;
    }

    /** @return outputs per candidate */
    int outputs() {
        return outputs;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int batch = input.shape().dim(0);
        int r = regions.size();
        Var[] state = new Var[r];
        for (int t = 0; t < ticks; t++) {
            Var[] next = new Var[r];
            for (int i = 0; i < r; i++) {
                String name = regions.get(i);
                Var pre = null;
                Var control = null;
                for (int j = 0; j < projections.size(); j++) {
                    if (!projections.get(j)[1].equals(name)) {
                        continue;
                    }
                    String from = projections.get(j)[0];
                    int s = from.equals(Genome.INPUT) ? -1 : regions.indexOf(from);
                    Var src = s < 0 ? input : s < i ? next[s] : state[s];
                    if (src == null) {
                        continue;
                    }
                    Var term = src.matmul(weights[j].mul(masks[j]).mul(scale));
                    pre = pre == null ? term : pre.add(term);
                    if (controls[j] != null) {
                        Var ct = src.matmul(controls[j].mul(masks[j]).mul(scale));
                        control = control == null ? ct : control.add(ct);
                    }
                }
                Var b = biases[i].mul(scale);
                pre = pre == null ? b.expand(Shape.of(count, batch, width[i])) : pre.add(b);
                Var out = null;
                List<Activation> acts = activations.get(i);
                for (int k = 0; k < acts.size(); k++) {
                    Var term = acts.get(k).forward(pre, training).mul(activationMasks.get(i).get(k));
                    out = out == null ? term : out.add(term);
                }
                if (switching[i] != null) {
                    Var cb = controlBias[i].mul(scale);
                    control = control == null ? cb.expand(Shape.of(count, batch, width[i])) : control.add(cb);
                    List<Var> u = new ArrayList<>();
                    List<Var> e = new ArrayList<>();
                    for (int k = 0; k < switching[i].candidates().size(); k++) {
                        u.add(switchScale.get(i).get(k).mul(scale));
                        e.add(switchOffset.get(i).get(k).mul(scale));
                    }
                    Var sw = io.github.zzuegg.brainiac.nn.SwitchingDense.mix(pre, control, u, e,
                            switching[i].candidates(), switching[i].hard(), training).mul(switchMasks[i]);
                    out = out == null ? sw : out.add(sw);
                }
                next[i] = out;
            }
            state = next;
        }
        List<Var> outs = new ArrayList<>();
        for (String o : genomes.getFirst().outputs()) {
            outs.add(state[regions.indexOf(o)]);
        }
        Var fed = outs.size() == 1 ? outs.getFirst() : Var.concat(2, outs.toArray(Var[]::new));
        Var y = fed.matmul(headWeight.mul(scale)).add(headBias.mul(scale));
        y = head.forward(y, training);                       // [count, batch, outputs]
        return y.permute(1, 0, 2).reshape(batch, count * outputs);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(count * outputs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weights));
        p.addAll(List.of(biases));
        for (int j = 0; j < controls.length; j++) {
            if (controls[j] != null) {
                p.add(controls[j]);
            }
        }
        for (int r = 0; r < regions.size(); r++) {
            if (switching[r] != null) {
                p.add(controlBias[r]);
                p.addAll(switchScale.get(r));
                p.addAll(switchOffset.get(r));
            }
        }
        p.add(headWeight);
        p.add(headBias);
        return p;
    }

    @Override
    public List<Var> buffers() {
        List<Var> b = new ArrayList<>(List.of(masks));
        b.add(scale);
        activationMasks.forEach(b::addAll);
        for (Var m : switchMasks) {
            if (m != null) {
                b.add(m);
            }
        }
        return b;
    }

    /**
     * The loss of all candidates together: the sum of each one's own loss, so each gets the
     * gradient it would get alone.
     *
     * @param loss the task's loss
     * @return the loss for {@code [batch, candidates * outputs]} predictions and targets
     *     {@linkplain #tile tiled} the same way
     */
    io.github.zzuegg.brainiac.nn.Loss loss(io.github.zzuegg.brainiac.nn.Loss loss) {
        return (prediction, target) -> {
            int rows = prediction.shape().dim(0) * count;
            return loss.compute(prediction.reshape(rows, outputs), target.reshape(rows, outputs))
                    .mul((float) count);
        };
    }

    /**
     * Repeats a dataset's targets once per candidate, to match this layer's output.
     *
     * @param data the data
     * @return the same features with targets {@code [samples, candidates * outputs]}
     */
    io.github.zzuegg.brainiac.data.Dataset tile(io.github.zzuegg.brainiac.data.Dataset data) {
        Tensor t = data.targets();
        int n = t.shape().dim(0);
        Tensor tiled = t.reshape(n, 1, outputs)
                .add(Tensor.zeros(Shape.of(1, count, outputs), t.dtype(), t.device()))
                .reshape(n, count * outputs);
        return io.github.zzuegg.brainiac.data.Dataset.of(data.features(), tiled);
    }

    /**
     * Reads one candidate's trained weights.
     *
     * @param c the candidate
     * @return its weights by neuron id
     */
    Weights weights(int c) {
        Genome g = genomes.get(c);
        float s = scale.value().to(DType.FLOAT32).toArray()[c];
        Map<Long, Float> conn = new HashMap<>();
        for (int j = 0; j < projections.size(); j++) {
            String from = projections.get(j)[0];
            String to = projections.get(j)[1];
            if (brainIndex(g, from, to) < 0) {
                continue;
            }
            int rows = size(from);
            int cols = size(to);
            float[] w = weights[j].value().to(DType.FLOAT32).toArray();
            float[] m = masks[j].value().to(DType.FLOAT32).toArray();
            int[] src = Weights.ids(g, from);
            int[] dst = Weights.ids(g, to);
            for (int a = 0; a < src.length; a++) {
                for (int b = 0; b < dst.length; b++) {
                    int k = (c * rows + a) * cols + b;
                    if (m[k] != 0) {
                        conn.put(Genome.key(src[a], dst[b]), s * w[k]);
                    }
                }
            }
        }
        Map<Integer, Float> bias = new HashMap<>();
        for (int r = 0; r < regions.size(); r++) {
            float[] b = biases[r].value().to(DType.FLOAT32).toArray();
            List<Genome.Neuron> ns = g.regions().get(r).neurons();
            for (int i = 0; i < ns.size(); i++) {
                bias.put(ns.get(i).id(), s * b[c * width[r] + i]);
            }
        }
        int fed = headWeight.value().shape().dim(1);
        float[] hwAll = headWeight.value().to(DType.FLOAT32).toArray();
        float[] hbAll = headBias.value().to(DType.FLOAT32).toArray();
        int ownFed = g.outputs().stream().mapToInt(o -> g.region(o).neurons().size()).sum();
        float[] hw = new float[ownFed * outputs];
        int row = 0;
        int padded = 0;
        for (String o : g.outputs()) {
            int own = g.region(o).neurons().size();
            for (int i = 0; i < own; i++) {
                for (int k = 0; k < outputs; k++) {
                    hw[(row + i) * outputs + k] = s * hwAll[((c * fed) + padded + i) * outputs + k];
                }
            }
            row += own;
            padded += width[regions.indexOf(o)];
        }
        float[] hb = new float[outputs];
        for (int k = 0; k < outputs; k++) {
            hb[k] = s * hbAll[c * outputs + k];
        }
        Map<Long, Float> ctl = new HashMap<>();
        Map<String, Float> sw = new HashMap<>();
        for (int r = 0; r < regions.size(); r++) {
            Genome.Region region = g.regions().get(r);
            if (region.switching() == null) {
                continue;
            }
            int n = width[r];
            List<int[]> from = new ArrayList<>();
            List<float[]> control = new ArrayList<>();
            List<boolean[]> mask = new ArrayList<>();
            for (int j = 0; j < projections.size(); j++) {
                if (!projections.get(j)[1].equals(region.name()) || brainIndex(g, projections.get(j)[0],
                        region.name()) < 0) {
                    continue;
                }
                int rows = size(projections.get(j)[0]);
                float[] v = controls[j].value().to(DType.FLOAT32).toArray();
                float[] m = masks[j].value().to(DType.FLOAT32).toArray();
                float[] cv = new float[rows * n];
                boolean[] cm = new boolean[rows * n];
                for (int k = 0; k < rows * n; k++) {
                    cv[k] = s * v[c * rows * n + k];
                    cm[k] = m[c * rows * n + k] != 0;
                }
                from.add(Weights.ids(g, projections.get(j)[0]));
                control.add(cv);
                mask.add(cm);
            }
            int kk = switching[r].candidates().size();
            float[][] own = new float[3][];
            own[0] = slice(controlBias[r], c, n, s);
            own[1] = new float[kk * n];
            own[2] = new float[kk * n];
            for (int k = 0; k < kk; k++) {
                System.arraycopy(slice(switchScale.get(r).get(k), c, n, s), 0, own[1], k * n, n);
                System.arraycopy(slice(switchOffset.get(r).get(k), c, n, s), 0, own[2], k * n, n);
            }
            Weights.readSwitching(region, from, control, mask, own, n, ctl, sw);
        }
        return Weights.fromHead(g, conn, bias, hw, hb, ctl, sw);
    }

    /** Candidate {@code c}'s {@code n} values of a {@code [count, 1, n]} parameter, times {@code s}. */
    private static float[] slice(Var v, int c, int n, float s) {
        float[] all = v.value().to(DType.FLOAT32).toArray();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = s * all[c * n + i];
        }
        return out;
    }
}
