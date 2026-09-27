# Security Policy

## Supported Versions

This project has not yet reached a 1.0 release and does not maintain separate
release branches. Security fixes are made against the `master` branch only.

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, report it privately using one of the following channels:

- **GitHub Private Vulnerability Reporting**: open a report via the
  [Security tab](../../security/advisories/new) of this repository (preferred).
- **Email**: simon.bouchard31@gmail.com

Please include as much detail as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof of concept
- Any relevant logs, stack traces, or configuration

## Response

This is a solo-maintained project. I will acknowledge reports within a few
days and aim to provide an initial assessment within two weeks. There is no
bug bounty program.

## Scope

This repository includes several services (backend API, search, model
servers, chatbot/agent). Vulnerabilities in third-party dependencies should be
reported upstream, but feel free to flag them here too if they affect how
this project uses them.
