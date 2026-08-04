# Houdayer coordinate-cluster feasibility

This frozen local probe tests whether a two-replica isoenergetic cluster move
has a meaningful sparse coordinate structure before any MCMC trajectory is
launched.  It uses a code-only basis of `ker(H_Z)`: independent H_X rows plus
an HGP tensor logical complement, rather than a dense Gaussian nullspace.

For two coordinate states `x,x'`, let `D={i:x_i != x'_i}`.  A physical-bit
factor connects all members of `D` in its coordinate support.  Swapping one
complete connected component between replicas exchanges the two factor inputs
at every touched factor, preserving the product posterior exactly.  This is
the generalized Houdayer/isoenergetic-cluster identity; it is not a heuristic
acceptance rule.

The probe measures component sizes and logical-label ranks for four frozen,
code-defined coordinate differences, including a deterministic exact-uniform
coordinate pair.  It also reports only diagnostic physical weights after the
pair definitions are frozen.  A useful result must show more than a single
whole-population swap: component decomposition and logical delta rank are the
relevant structural signals.

No MCMC, posterior estimate, q_top, remote work, held-out work, or production
authority is created here.  A structurally promising result would still need
the small-HGP joint detailed-balance oracle and a fresh adversarial-start
local viability contract.
