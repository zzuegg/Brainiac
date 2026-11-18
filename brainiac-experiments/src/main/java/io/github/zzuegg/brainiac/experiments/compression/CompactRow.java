package io.github.zzuegg.brainiac.experiments.compression;

/**
 * One compressed model in a cost-aware comparison.
 *
 * @param method the method and its setting
 * @param metric the quality: test accuracy, or a loss where stated
 * @param cost the cost: multiply-adds per example, or stored bits, as the experiment states
 * @param size the parameter count, or another size measure the experiment states
 * @param millis a measured time in milliseconds (inference latency), or -1
 */
public record CompactRow(String method, double metric, double cost, double size, double millis) {
}
