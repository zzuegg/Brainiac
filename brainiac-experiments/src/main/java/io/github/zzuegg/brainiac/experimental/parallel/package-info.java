/**
 * Parallel training building blocks simulated on one machine, with Java threads as devices and a
 * counting collective layer: pipeline schedules (GPipe, 1F1B, interleaved, zero-bubble, DualPipe)
 * with a real threaded runner and a calibrated cost model, split backward passes, Megatron tensor
 * and sequence parallelism, ZeRO sharding, checkpoint planning, expert parallelism and
 * asynchronous SGD. Core's {@code DataParallel}, {@code TcpCollective}, {@code Offloaded} and
 * {@code Recompute} stay where they are and are used as baselines.
 */
package io.github.zzuegg.brainiac.experimental.parallel;
