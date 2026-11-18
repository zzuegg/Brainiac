# Running Brainiac on rented Vast.ai GPUs

Brainiac trains on one GPU per run through its Vulkan backend. `brainiac-cloud` rents one
machine on [Vast.ai](https://vast.ai), copies this checkout to it, runs `setup.sh` there and
destroys the machine again.

## Once

1. Add your SSH public key at https://cloud.vast.ai/manage-keys/ (new instances get it).
2. Give the launcher your API key, either as the `VAST_API_KEY` environment variable or
   saved in the studio (Training → Vast.ai), which keeps it in
   `~/.brainiac/studio/keys.properties`. The launcher never prints it.

## Use

```sh
./gradlew :brainiac-cloud:run --args="offers"                 # cheapest single RTX 4090/5090, A100, H100
./gradlew :brainiac-cloud:run --args="run --offer 123456"     # asks, rents, smoke-tests, destroys
./gradlew :brainiac-cloud:run --args="run --offer 123456 --job tests"
./gradlew :brainiac-cloud:run --args="run --offer 123456 --job run -- ./gradlew :brainiac-examples:run --args=iris"
./gradlew :brainiac-cloud:run --args="worker --offer 123456"  # studio worker behind an SSH tunnel
./gradlew :brainiac-cloud:run --args="instances"              # anything still rented (and billing)?
./gradlew :brainiac-cloud:run --args="destroy --all"
```

Renting always asks first with the offer and its hourly price (`--yes` skips the question).
The instance is destroyed when the job ends or fails, after `--max-minutes` (default 120),
or on Ctrl+C. `--keep` keeps it; `--instance ID` reuses one. Logs go to `build/vast/`.
Only the `HEAD` commit is sent, so commit first. When that commit is pushed to the GitHub
remote `origin`, the instance fetches it from GitHub itself (seconds, instead of minutes of
upload from a home connection); otherwise, or if the fetch fails, it is uploaded over SSH.
`--upload` always uploads.

A chosen `--offer ID` is rented directly even when a fresh search does not list it (Vast
returns a different subset of offers on each search); only Vast's `no_such_ask` means it is gone.

### Secrets for the job

```sh
./gradlew :brainiac-cloud:run --args="run --offer 123456 --secret HF_TOKEN --job run -- ./gradlew ..."
```

`--secret NAME` (repeatable) sets the environment variable `NAME` for the job or worker.
The value is looked up on this machine: the environment variable `NAME`, else the studio's
key file under a provider (`--secret NAME=PROVIDER`; `HF_TOKEN` defaults to `huggingface`),
else, for `HF_TOKEN`, the Hugging Face token files (`~/.brainiac/hf_token`,
`~/.cache/huggingface/token`). It travels over the SSH connection's input and is read into an
exported variable by the job's shell, so it never lands on the instance's disk, on a command
line or in Vast's instance fields (which the Vast API shows). Its value is masked as
`***NAME***` in the console and in `build/vast/*.log`. A missing secret stops the launcher
before anything is rented.

Before copying anything, the launcher runs `setup.sh check` with just the script. Some hosts
give containers NVIDIA's CUDA driver without its Vulkan libraries, and `check` fails there with
exit code 2. The machine id then goes to `~/.brainiac/vast/bad-machines.txt`, and later searches
skip that machine. With `--cheapest`, the instance is destroyed and the next cheapest offer is
tried (up to 3 hosts, each within your limits). `offers` marks known bad machines.

A host that never starts, or whose SSH login does not work within `--ssh-minutes` (default 3),
is treated the same way: destroyed (even with `--keep`), remembered, and with `--cheapest` the
next offer is rented. Some hosts never install the account's SSH key, and Vast's SSH proxy can
stay dead for a long time while the API reports the instance as running.

`setup.sh` tests each JDK download (`gzip -t` and a `tar` listing) and downloads it again, up
to 3 times, when it arrived cut off or corrupt.

## setup.sh jobs

| Job | What it does |
| --- | --- |
| `check` | Installs the Vulkan loader and checks that an NVIDIA Vulkan device shows up |
| `smoke` | `check`, JDK 21 + 27, build, Vulkan backend tests, benchmarks, 1 epoch of MNIST training |
| `tests` | `check`, JDKs, `./gradlew build` |
| `bench` | `check`, JDKs, benchmarks only |
| `worker` | A studio worker (`--worker --listen 127.0.0.1 --port 7870`) for chat and training |
| `run CMD...` | `check`, JDKs, then any command |

## Vulkan on NVIDIA containers

NVIDIA's container runtime mounts the driver's Vulkan libraries only when the container asks
for the `graphics` capability, so instances are rented with `NVIDIA_DRIVER_CAPABILITIES=all`.
`setup.sh` writes the missing `nvidia_icd.json` when the libraries are there but the file is
not. Some hosts still expose CUDA only; then `check` fails with exit code 2 and the instance
is destroyed. Pick another offer.

## The worker

`worker` starts the studio in worker mode bound to the instance's `127.0.0.1`, reachable only
through the SSH tunnel the launcher holds open (`127.0.0.1:7870` on your PC). The launcher
generates the bearer token and sends it over the SSH connection, then prints the URL and token
to add under Training → Add a host. `VastLaunch.exposePorts(7870)` maps the port publicly
instead, but that sends the token and every prompt unencrypted.
