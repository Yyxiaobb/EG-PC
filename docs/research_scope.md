# Research Scope

## Central question

For a long prompt, when should the system compress, how much should it compress, and does the resulting request actually finish faster after including compressor overhead?

## Primary claim to test

An adaptive budget policy can achieve a lower end-to-end latency than no compression and fixed-budget compression while keeping quality degradation within a predefined tolerance.

## Non-goals for the first milestone

- Training a new large language model.
- Rebuilding the old evidence-retention project.
- Claiming general superiority over every prompt compression method.
- Treating token reduction alone as inference acceleration.

## Required measurement breakdown

Every request should record:

1. original prompt tokens;
2. compressed prompt tokens;
3. compressor latency;
4. prefill latency or time-to-first-token;
5. decode latency;
6. end-to-end latency;
7. output quality and quality drop.
