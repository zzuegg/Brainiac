package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs another optimizer on a host copy of the parameters (ZeRO-offload): each step moves
 * the gradients to the host, updates the host copies there, and moves the new values back.
 * The optimizer's state (Adam's two averages: twice the parameters) and a float32 master copy
 * of the weights then live in host memory, and the GPU keeps only the weights and gradients.
 */
final class Offloaded implements Optimizer {
    private final Optimizer inner;
    private final Device host;
    private final Map<Var, Var> copies = new IdentityHashMap<>();

    Offloaded(Optimizer inner, Device host) {
        this.inner = inner;
        this.host = host;
    }

    private List<Var> hosted(List<Var> parameters) {
        List<Var> out = new ArrayList<>(parameters.size());
        for (Var p : parameters) {
            out.add(copies.computeIfAbsent(p, k -> Var.param(k.value().to(host))));
        }
        return out;
    }

    @Override
    public void step(List<Var> parameters) {
        List<Var> onHost = hosted(parameters);
        List<Var> stepped = new ArrayList<>();
        List<Var> hostStepped = new ArrayList<>();
        for (int i = 0; i < parameters.size(); i++) {
            Var p = parameters.get(i);
            if (p.grad() == null) {
                continue;
            }
            Var h = onHost.get(i);
            Tensor g = p.grad();
            h.setGrad(g.to(host).to(h.value().dtype()));
            p.zeroGrad();
            g.release();
            stepped.add(p);
            hostStepped.add(h);
        }
        inner.step(hostStepped);
        for (int i = 0; i < stepped.size(); i++) {
            Var p = stepped.get(i);
            p.assign(hostStepped.get(i).value().to(p.value().device()).to(p.value().dtype()));
        }
    }

    @Override
    public void resetState(Var parameter, Tensor keep) {
        Var h = copies.get(parameter);
        if (h != null) {
            inner.resetState(h, keep.to(host));
        }
    }

    @Override
    public void forget(Var parameter) {
        Var h = copies.remove(parameter);
        if (h != null) {
            inner.forget(h);
        }
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        return inner.state(hosted(parameters));
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        List<Var> onHost = hosted(parameters);
        for (int i = 0; i < parameters.size(); i++) {
            onHost.get(i).assign(parameters.get(i).value().to(host));   // the weights a checkpoint restored
        }
        Map<String, Tensor> moved = new java.util.LinkedHashMap<>();
        state.forEach((k, v) -> moved.put(k, v.to(host)));
        inner.loadState(onHost, moved);
    }

    @Override
    public String toString() {
        return "Offloaded(" + inner + " on " + host + ")";
    }
}
