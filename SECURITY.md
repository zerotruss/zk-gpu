# Security policy

## Read this first

`zk-gpu` is **not audited** and is **not constant-time**. Do not use it with secret-dependent
scalars where timing could leak information, and do not rely on it for anything that needs a
reviewed cryptographic implementation. The tests compare it with arkworks and ffjavascript, but
tests are not a security review.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's private reporting:
open the **Security** tab of this repository and choose **Report a vulnerability**. Include what
you found, how to reproduce it, and which version you used.

You should get a first reply within about a week. This is a small volunteer project, so there is
no formal timeline for fixes; reports that give wrong answers on valid input are treated as bugs
of the highest priority.

## Supported versions

Only the latest release is supported.
