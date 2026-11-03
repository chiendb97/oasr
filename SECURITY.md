# Security Policy

## Supported versions

OASR has not cut a stable release yet. Security fixes land on `main`; please
reproduce against the latest `main` before reporting.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/chiendb97/oasr/security/advisories/new)
(the **Security** tab → **Report a vulnerability**). Include:

- what an attacker can do, and under which configuration;
- the request, file or command that triggers it;
- the commit you tested (`git rev-parse HEAD`).

You can expect an acknowledgement within a week. Once a fix is available, we
will publish an advisory and credit you unless you prefer otherwise.

## Deployment model

Some behaviour is by design and is not a vulnerability. Knowing it helps you
deploy OASR safely:

- **`oasr-server` performs no authentication or authorization, and does not
  terminate TLS.** It accepts an `Authorization` header so a reverse proxy can
  enforce one, but it does not check the header itself. Bind it to localhost or
  a private network, and put an authenticating, TLS-terminating proxy in front
  of anything reachable by untrusted clients. See
  [`docs/serving.md`](docs/serving.md).
- **Load model assets only from sources you trust.** Model weights are read
  with safetensors or `torch.load(weights_only=True)`, but the offline WFST
  graph export unpickles a k2 `HLG.pt` with `weights_only=False`, and the
  Silero speech detector loads a TorchScript archive.
- **Resource limits are configuration.** `--max-audio-seconds`,
  `--max-audio-mib` and `--max-concurrent-decodes` bound what one request can
  cost. Size them for your deployment before exposing the server.

In scope: a bypass of these limits; a crash, hang or memory-safety bug
reachable from a network request or an audio file; and code execution from
anything other than a model asset you chose to load.
