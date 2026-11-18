# Notebook 09, Synthesis

Documents `notebooks/09_synthesis.ipynb`.

## Purpose

State what the study found, what it can defend, and what it cannot. This stage runs no code
and writes no file. It is the written record that ties the eight preceding stages to the
thesis.

## Inputs

Every table and figure produced by stages 01 to 08. It reads them as a reader would, rather
than loading them programmatically.

## Main processing stages

Section 9.1 cross checks the research blueprint against the thesis text against what the
repository actually implements. Section 9.2 is the strengths, weaknesses, opportunities and
threats assessment. Section 9.3 states the solution in specific, measurable, achievable,
relevant and time bound terms. Section 9.4 covers ethics, reliability and defensibility.
Section 9.5 gives the conclusion and the next steps. Section 9.6 states, for completeness,
that this stage produces no output.

## Important functions called

None. This stage contains no code cells.

## Important outputs

None. Nothing is written, and nothing downstream depends on this stage.

## Relationship with other notebooks

It is the end of the pipeline. It consumes everything and produces nothing.

## Methodological decisions made here

1. **Claims are tied to intervals, not to point estimates.** Every statement in this stage
   either cites an interval that excludes the null or is labelled as suggestive.
2. **Negative results are stated as findings.** That return magnitude is not predictable and
   that exact recovery duration is not predictable are reported as results of the study, not
   as shortcomings of it.
3. **The known limitations are listed rather than softened.** Seventy four events, forty
   pooled test points, no final lockbox holdout, and an adaptive selection risk that the
   bootstrap and the family wise correction reduce but do not eliminate.

## Justification

A synthesis stage that recomputed numbers would be a second chance to arrive at a different
answer. Keeping it text only means the conclusions can only restate what stages 06 and 08
already produced. The fuller versions of what this stage summarises live in
`docs/results.md` for the numbers and `docs/interpretation.md` for the claims those numbers
license.

## Execution requirements

None technically, but the statements here describe results that must exist in the current
cache. If the earlier stages have not run, this stage is describing numbers that are not
there.
