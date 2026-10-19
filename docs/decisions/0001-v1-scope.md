# ADR 0001: ATLAS V1 Scope

- **Status:** Accepted
- **Date:** 2026-10-18

## Context

ATLAS aims eventually to combine market data, machine learning, news, risk
management and automated decisions. Building all of that at once would make it
impossible to tell whether any result is real. Most failures in quantitative
research come from leakage, bad data and unrealistic costs, not from model
choice.

## Decision

V1 answers one question: *can information available at time `t` predict SPY
5-minute-ahead direction, and does that support a long/flat strategy with
positive out-of-sample returns after realistic costs?*

**In scope:**
- SPY historical 1-minute OHLCV and validated datasets
- Causal feature engineering
- Logistic regression and LightGBM baselines
- Chronological walk-forward validation
- Long/flat simulation with realistic fees and slippage
- Reproducible experiments and test-driven development

**Deferred:**
- Live or paper brokerage integration (Alpaca or others) and order placement
- News and alternative data, and LLM agents
- Deep models (CNNs, Transformers) and reinforcement learning
- Multi-asset universes, shorting and leverage
- Distributed execution across CSIL workstations, GPU pipelines
- Databases, web frontends, CI/CD, orchestration and experiment-tracking servers

## Rationale

- Simple baselines give a reference point. Without one, gains from complex
  models cannot be judged.
- Every deferred component is another source of leakage, operational risk or
  irreproducibility.
- The data contract and causality discipline must be proven on the simple case
  first. Everything later inherits them.

## Consequences

Each deferred item needs its own decision record, justified by a demonstrated
need such as a V1 result or a measured bottleneck, before any work starts.
