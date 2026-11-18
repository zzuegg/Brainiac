package io.github.zzuegg.brainiac.experimental.edge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Activation-memory planning over a layer graph: tensor lifetimes, the peak memory of an
 * execution order, the order with the lowest peak (operator reordering), in-place operators, and
 * the memory of on-device training.
 *
 * <p>A tensor is alive from the step that produces it until the last step that reads it (the
 * graph's output stays alive). At each step the operator's inputs and output are alive together,
 * except for an in-place operator (a depthwise convolution with stride 1, for example) whose
 * output overwrites an input that nothing else reads later. The best order is found by dynamic
 * programming over the executed sets, whose live memory does not depend on the order that
 * reached them (exact, exponential in the width of the graph, fine for blocks and cells).
 *
 * <p>Papers: Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices", NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2007.10319">arXiv:2007.10319</a>; Lin et al., "MCUNetV2:
 * Memory-Efficient Patch-based Inference for Tiny Deep Learning", NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2110.15352">arXiv:2110.15352</a>.
 */
public final class MemoryPlanner {

    /**
     * An operator and the tensor it produces.
     *
     * @param name a label
     * @param inputs indices of the operators whose outputs it reads (earlier in the list)
     * @param bytes the size of its output
     * @param inPlace whether it may overwrite its first input when that input is not read later
     */
    public record Node(String name, int[] inputs, long bytes, boolean inPlace) {
    }

    /**
     * Training memory, in bytes.
     *
     * @param storedActivations activations and masks kept from the forward pass for the backward pass
     * @param gradients gradient buffers of the updated parameters
     * @param updatedWeights updated weights, which must sit in RAM rather than flash
     * @param workingSet the largest working set of the forward or backward pass
     * @param total the sum
     */
    public record TrainingMemory(long storedActivations, long gradients, long updatedWeights, long workingSet,
            long total) {
    }

    private final List<Node> nodes;

    /**
     * Creates a planner. Node 0 is normally the graph input; the last node is the output.
     *
     * @param nodes the operators in a valid topological order
     */
    public MemoryPlanner(List<Node> nodes) {
        this.nodes = List.copyOf(nodes);
        for (int i = 0; i < nodes.size(); i++) {
            for (int j : nodes.get(i).inputs()) {
                if (j >= i) {
                    throw new IllegalArgumentException("node " + i + " reads a later node " + j);
                }
            }
        }
    }

    /**
     * The chain graph of a network, activations at a given number of bytes per element.
     * Depthwise convolutions with stride 1 are marked in place.
     *
     * @param net the network
     * @param bytesPerElement 1 for int8, 4 for float
     * @return the planner
     */
    public static MemoryPlanner chain(EdgeNet net, int bytesPerElement) {
        List<Node> nodes = new ArrayList<>();
        nodes.add(new Node("input", new int[0], EdgeNet.size(net.inputShape()) * bytesPerElement, false));
        List<int[]> shapes = net.shapes();
        for (int i = 0; i < net.units().size(); i++) {
            EdgeNet.Unit u = net.units().get(i);
            boolean inPlace = u instanceof EdgeNet.Conv c && c.groups() == c.inChannels()
                    && c.inChannels() == c.outChannels() && c.stride() == 1;
            nodes.add(new Node(u.getClass().getSimpleName() + i, new int[] {i},
                    EdgeNet.size(shapes.get(i)) * bytesPerElement, inPlace));
        }
        return new MemoryPlanner(nodes);
    }

    /** @return the nodes */
    public List<Node> nodes() {
        return nodes;
    }

    private int[] lastUse(int[] order) {
        int[] last = new int[nodes.size()];
        Arrays.fill(last, -1);
        for (int step = 0; step < order.length; step++) {
            for (int j : nodes.get(order[step]).inputs()) {
                last[j] = step;
            }
        }
        last[nodes.size() - 1] = Integer.MAX_VALUE;
        return last;
    }

    /** @return the given (topological) order */
    public int[] defaultOrder() {
        int[] o = new int[nodes.size()];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        return o;
    }

    /**
     * The peak live memory of an execution order.
     *
     * @param order every node once, inputs before readers
     * @return the peak in bytes
     */
    public long peak(int[] order) {
        int[] last = lastUse(order);
        boolean[] live = new boolean[nodes.size()];
        long current = 0;
        long peak = 0;
        for (int step = 0; step < order.length; step++) {
            Node n = nodes.get(order[step]);
            long saving = 0;
            if (n.inPlace() && n.inputs().length > 0 && last[n.inputs()[0]] == step) {
                saving = Math.min(n.bytes(), nodes.get(n.inputs()[0]).bytes());
            }
            peak = Math.max(peak, current + n.bytes() - saving);
            live[order[step]] = true;
            current += n.bytes();
            for (int j : n.inputs()) {
                if (last[j] == step && live[j]) {
                    live[j] = false;
                    current -= nodes.get(j).bytes();
                }
            }
            if (last[order[step]] == -1 && order[step] != nodes.size() - 1) {
                live[order[step]] = false;
                current -= n.bytes();
            }
        }
        return peak;
    }

    /**
     * The execution order with the lowest peak, by dynamic programming over executed sets
     * (at most 24 nodes).
     *
     * @return the best order
     */
    public int[] bestOrder() {
        int n = nodes.size();
        if (n > 24) {
            throw new IllegalArgumentException("too many nodes for the exact search: " + n);
        }
        int[] inMask = new int[n];
        int[] readers = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j : nodes.get(i).inputs()) {
                inMask[i] |= 1 << j;
                readers[j] |= 1 << i;
            }
        }
        Map<Integer, Long> best = new HashMap<>();
        Map<Integer, Integer> from = new HashMap<>();
        best.put(0, 0L);
        List<Integer> frontier = List.of(0);
        int full = (1 << n) - 1;
        for (int size = 0; size < n; size++) {
            List<Integer> next = new ArrayList<>();
            for (int set : frontier) {
                long live = liveBytes(set, readers);
                long sofar = best.get(set);
                for (int v = 0; v < n; v++) {
                    if ((set >> v & 1) != 0 || (inMask[v] & ~set) != 0) {
                        continue;
                    }
                    Node node = nodes.get(v);
                    long saving = 0;
                    if (node.inPlace() && node.inputs().length > 0) {
                        int u = node.inputs()[0];
                        if ((readers[u] & ~(set | 1 << v)) == 0 && u != n - 1) {
                            saving = Math.min(node.bytes(), nodes.get(u).bytes());
                        }
                    }
                    long peak = Math.max(sofar, live + node.bytes() - saving);
                    int ns = set | 1 << v;
                    Long old = best.get(ns);
                    if (old == null) {
                        next.add(ns);
                    }
                    if (old == null || peak < old) {
                        best.put(ns, peak);
                        from.put(ns, v);
                    }
                }
            }
            frontier = next;
        }
        int[] order = new int[n];
        int set = full;
        for (int k = n - 1; k >= 0; k--) {
            int v = from.get(set);
            order[k] = v;
            set &= ~(1 << v);
        }
        return order;
    }

    private long liveBytes(int set, int[] readers) {
        long live = 0;
        int n = nodes.size();
        for (int i = 0; i < n; i++) {
            if ((set >> i & 1) != 0 && ((readers[i] & ~set) != 0 || i == n - 1)) {
                live += nodes.get(i).bytes();
            }
        }
        return live;
    }

    /**
     * Training memory of a network under a sparse update. The backward pass runs from the output
     * down to the earliest unit that updates anything. A unit whose weights are updated keeps its
     * input activation; every ReLU on the way keeps a one-bit mask; bias updates keep nothing.
     * Updated weights move to RAM with float32 gradient buffers.
     *
     * @param net the network
     * @param weightShare per unit, the share of output channels whose weights are updated
     * @param bias per unit, whether its bias is updated
     * @param extraStored further stored activations (side branches, for example), in bytes
     * @param extraParameters further updated parameters (adapters, for example)
     * @param activationBytes bytes per activation element (1 for int8)
     * @return the memory
     */
    public static TrainingMemory trainingMemory(EdgeNet net, double[] weightShare, boolean[] bias,
            long extraStored, long extraParameters, int activationBytes) {
        List<int[]> shapes = net.shapes();
        int n = net.units().size();
        int earliest = n;
        for (int i = 0; i < n; i++) {
            if (weightShare[i] > 0 || bias[i]) {
                earliest = i;
                break;
            }
        }
        if (extraParameters > 0 || extraStored > 0) {
            earliest = Math.min(earliest, 0);
        }
        long stored = extraStored;
        long grads = 4L * extraParameters;
        long weights = extraParameters;
        long trans = 0;
        long inference = 0;
        for (int i = 0; i < n; i++) {
            long in = EdgeNet.size(i == 0 ? net.inputShape() : shapes.get(i - 1));
            long out = EdgeNet.size(shapes.get(i));
            inference = Math.max(inference, (in + out) * activationBytes);
            if (i < earliest) {
                continue;
            }
            EdgeNet.Unit u = net.units().get(i);
            long wCount = switch (u) {
                case EdgeNet.Conv c -> c.weight().value().size();
                case EdgeNet.Linear l -> l.weight().value().size();
                case EdgeNet.Pool p -> 0;
            };
            int outC = switch (u) {
                case EdgeNet.Conv c -> c.outChannels();
                case EdgeNet.Linear l -> (int) l.bias().value().size();
                case EdgeNet.Pool p -> 0;
            };
            boolean relu = switch (u) {
                case EdgeNet.Conv c -> c.relu();
                case EdgeNet.Linear l -> l.relu();
                case EdgeNet.Pool p -> false;
            };
            if (weightShare[i] > 0) {
                stored += in * activationBytes;
                long updated = Math.round(Math.ceil(weightShare[i] * outC) * (wCount / Math.max(1, outC)));
                grads += 4 * updated;
                weights += updated;
            }
            if (bias[i]) {
                grads += 4L * outC;
                weights += 4L * outC;
            }
            if (relu && i > earliest) {
                stored += (out + 7) / 8;
            }
            if (i > earliest || weightShare[i] > 0) {
                trans = Math.max(trans, (in + out) * 4);
            }
        }
        trans = Math.max(trans, inference);
        return new TrainingMemory(stored, grads, weights, trans, stored + grads + weights + trans);
    }
}
