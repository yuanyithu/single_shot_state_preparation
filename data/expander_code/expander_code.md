Implement a Python module for constructing and analyzing a family of quantum expander codes obtained from random biregular bipartite expander graphs via the hypergraph product construction.

Use exact GF(2) linear algebra throughout. Do not use floating point arithmetic for rank, kernel, quotient spaces, logical operators, or distance. Expansion verification and distance computation must be mathematically exact; exponential complexity is acceptable.

Mathematical convention
============================================================

Use the following convention.

Let G = (A ∪ B, E) be a simple bipartite graph with

    |A| = n_A,
    |B| = n_B,

left degree d_A on A and right degree d_B on B.

The classical parity-check matrix is

    H ∈ F_2^{B × A},

with

    H[b, a] = 1  iff  (a,b) ∈ E.

Thus H has n_B rows and n_A columns.

The quantum code has physical qubits indexed by

    V = (A × A) ⊔ (B × B).

Use the column order

    first all qubits in A × A,
    then all qubits in B × B.

Construct the CSS parity-check matrices

    H_X = [ I_{n_A} ⊗ H      |      H^T ⊗ I_{n_B} ],
    
    H_Z = [ H ⊗ I_{n_A}      |      I_{n_B} ⊗ H^T ].

Both matrices have n_A n_B rows and n_A^2 + n_B^2 columns.

Use the standard CSS convention:

    rows of H_X define X-type stabilizer generators,
    rows of H_Z define Z-type stabilizer generators,

and verify

    H_X H_Z^T = 0 over F_2.

If this convention conflicts with any naming in comments or tests, keep the above formulas as authoritative.

Required functionality
============================================================

Implement the following objects and functions.

------------------------------------------------------------
1. Bipartite graph data structure
------------------------------------------------------------

Create a class, for example

    BiregularBipartiteGraph

containing at least:

    n_A: int
    n_B: int
    d_A: int
    d_B: int
    A_to_B: list[set[int]]
    B_to_A: list[set[int]]
    seed: int
    rng_description: str
    construction_attempts: int

The graph must be simple: no multiple edges.

Add validation methods:

    check_degrees()
    check_simple()
    check_consistency()

where consistency means that A_to_B and B_to_A agree.

------------------------------------------------------------
2. Random construction of the classical bipartite graph
------------------------------------------------------------

Implement

    random_biregular_graph_from_m(
        m: int,
        d_A: int,
        d_B: int,
        seed: int,
        max_attempts: int = 10000,
    ) -> BiregularBipartiteGraph

The role of m is as follows.

Let

    g = gcd(d_A, d_B),
    a = d_A / g,
    b = d_B / g.

Set

    n_A = b m,
    n_B = a m.

Then automatically

    n_A d_A = n_B d_B.

Construct a random simple (d_A,d_B)-biregular bipartite graph using the configuration model:

    left_stubs  = each vertex in A repeated d_A times,
    right_stubs = each vertex in B repeated d_B times.

Shuffle the right stubs using a deterministic RNG initialized from seed, pair them with left stubs, and reject if there are parallel edges. Repeat until a simple graph is found or max_attempts is exceeded.

The returned graph object must store:

    seed,
    rng_description,
    construction_attempts.

This is important because I need to reuse exactly the same random code family later.

Use Python's standard-library random.Random(seed), unless there is a strong reason not to. Record the RNG choice in rng_description.

------------------------------------------------------------
3. Classical parity-check matrix
------------------------------------------------------------

Implement

    classical_parity_check_matrix(
        graph: BiregularBipartiteGraph,
    ) -> GF2Matrix

or return a NumPy uint8 array, but all operations must be interpreted over F_2.

The output H must have shape

    (n_B, n_A)

with

    H[b,a] = 1 iff b is adjacent to a.

------------------------------------------------------------
4. Quantum parity-check matrices
------------------------------------------------------------

Implement

    quantum_expander_parity_checks_from_graph(
        graph: BiregularBipartiteGraph,
    ) -> tuple[GF2Matrix, GF2Matrix]

returning

    H_Z, H_X

in that order.

However, internally and in the documentation, make very clear that the formulas are

    H_X = [ I_{n_A} ⊗ H      |      H^T ⊗ I_{n_B} ],
    H_Z = [ H ⊗ I_{n_A}      |      I_{n_B} ⊗ H^T ].

Also implement

    verify_css_commutation(H_X, H_Z) -> bool

checking exactly over F_2 that

    H_X H_Z^T = 0.

------------------------------------------------------------
5. Exact vertex-expansion verification
------------------------------------------------------------

Implement

    verify_vertex_expansion(
        graph: BiregularBipartiteGraph,
        gamma,
        delta,
        sides: str = "both",
        return_witness: bool = True,
    ) -> ExpansionVerificationResult

The function must exactly verify the vertex-expansion conditions:

Left expansion:

    for every S ⊆ A with |S| <= gamma * |A|,
    |Γ(S)| >= (1 - delta) d_A |S|.

Right expansion:

    for every T ⊆ B with |T| <= gamma * |B|,
    |Γ(T)| >= (1 - delta) d_B |T|.

Use exact rational arithmetic for gamma and delta. Accept either Fraction objects or strings such as "1/10", "1/16". Avoid floating-point comparison.

The function may be exponential and should enumerate all subsets up to the required size.

The return object should contain:

    passed: bool
    checked_left: bool
    checked_right: bool
    worst_left_ratio: optional exact Fraction
    worst_right_ratio: optional exact Fraction
    failing_side: optional str
    failing_subset: optional list[int]
    failing_neighborhood_size: optional int
    required_neighborhood_size: optional Fraction

For a subset S, the useful diagnostic ratio is

    |Γ(S)| / (d_A |S|)

on the left side, and

    |Γ(T)| / (d_B |T|)

on the right side.

Do not replace this with spectral expansion. The requirement is exact vertex expansion.

------------------------------------------------------------
6. Exact GF(2) linear algebra
------------------------------------------------------------

Implement a small GF(2) linear algebra toolkit.

Required functions:

    gf2_row_echelon(M)
    gf2_rank(M)
    gf2_nullspace(M)
    gf2_rowspace_basis(M)
    gf2_in_rowspace(v, rowspace_basis)
    gf2_extend_basis(existing_basis, candidate_vectors)
    gf2_quotient_basis(kernel_basis, subspace_basis)

All operations must be exact over F_2. Use XOR for addition.

It is acceptable to represent vectors as Python integers for efficiency, or as NumPy uint8 arrays, but the API should be clear and tested.

The quotient_basis function should return representatives of

    span(kernel_basis) / span(subspace_basis),

assuming span(subspace_basis) is contained in span(kernel_basis).

------------------------------------------------------------
7. Logical Pauli operators
------------------------------------------------------------

Implement

    logical_pauli_operators(
        H_X,
        H_Z,
    ) -> LogicalPauliResult

Using the standard CSS convention:

Logical Z operators are representatives of

    ker(H_X) / row(H_Z).

Logical X operators are representatives of

    ker(H_Z) / row(H_X).

The function must:

    1. Compute a quotient basis z_1,...,z_k for ker(H_X)/row(H_Z).
    2. Compute a quotient basis x_1,...,x_k for ker(H_Z)/row(H_X).
    3. Compute the pairing matrix
    
           M[i,j] = x_i · z_j mod 2.
    
    4. Verify M is invertible over F_2.
    5. Replace the X representatives by a linear combination so that the final representatives satisfy
    
           x_i · z_j = δ_ij mod 2.

Return an object containing:

    logical_X: list[vectors]
    logical_Z: list[vectors]
    pairing_matrix: GF2Matrix
    k: int

Also provide checks:

    H_Z x_i = 0 for all logical X representatives,
    H_X z_i = 0 for all logical Z representatives,
    x_i not in row(H_X),
    z_i not in row(H_Z),
    x_i · z_j = δ_ij,
    all X logicals mutually commute,
    all Z logicals mutually commute.

Remember that for CSS codes, two X-type vectors commute with each other automatically, and two Z-type vectors commute with each other automatically, but still provide consistency checks where appropriate.

------------------------------------------------------------
8. Exact code parameters [[n,k,d]]
------------------------------------------------------------

Implement

    code_parameters(
        H_X,
        H_Z,
        compute_distance: bool = True,
    ) -> CodeParameters

The function must compute

    n = number of columns,
    
    k = n - rank(H_X) - rank(H_Z).

If compute_distance is True, compute the exact CSS distance:

    d_X = min {|x| : x ∈ ker(H_Z) \ row(H_X)},
    
    d_Z = min {|z| : z ∈ ker(H_X) \ row(H_Z)},
    
    d = min(d_X, d_Z).

This can be exponential.

For exact distance, implement a brute-force search over vectors in the corresponding kernels, or a smarter exact search if desired, but it must be exact. No heuristic approximation is allowed.

The function should return:

    n: int
    k: int
    rank_H_X: int
    rank_H_Z: int
    d_X: optional int
    d_Z: optional int
    d: optional int
    min_logical_X: optional vector
    min_logical_Z: optional vector

Important:

    d_X searches vectors in ker(H_Z) that are not in row(H_X).
    d_Z searches vectors in ker(H_X) that are not in row(H_Z).

The naming follows the standard CSS convention:

    X logicals commute with Z checks, hence x ∈ ker(H_Z).
    Z logicals commute with X checks, hence z ∈ ker(H_X).

------------------------------------------------------------
9. Family-level wrapper
------------------------------------------------------------

Implement a convenience function

    build_quantum_expander_code_instance(
        m: int,
        d_A: int,
        d_B: int,
        seed: int,
        gamma = None,
        delta = None,
        verify_expansion: bool = False,
        compute_logicals: bool = True,
        compute_distance: bool = False,
    ) -> QuantumExpanderCodeInstance

It should:

    1. Build the random biregular graph from m,d_A,d_B,seed.
    2. Build H from the graph.
    3. Build H_Z,H_X.
    4. Verify CSS commutation.
    5. Optionally verify vertex expansion exactly if gamma and delta are provided.
    6. Optionally compute logical Pauli operators.
    7. Optionally compute exact [[n,k,d]] parameters.

The returned object should include the graph, H, H_Z, H_X, seed metadata, expansion result, logical operators, and code parameters.

------------------------------------------------------------
10. Tests and validation
------------------------------------------------------------

Write unit tests for the following.

A. Graph construction:

    - n_A = b m, n_B = a m.
    - all A degrees equal d_A.
    - all B degrees equal d_B.
    - no multiple edges.
    - A_to_B and B_to_A are consistent.
    - same seed gives exactly same graph.
    - different seeds usually give different graphs.

B. Parity-check construction:

    - H has shape (n_B,n_A).
    - row weights are d_B.
    - column weights are d_A.

C. Quantum parity-check construction:

    - H_X and H_Z have shape (n_A n_B, n_A^2 + n_B^2).
    - CSS commutation H_X H_Z^T = 0 over F_2.
    - row weights are d_A + d_B.
    - column weights are bounded by 2 max(d_A,d_B).

D. GF(2) linear algebra:

    - rank-nullity holds.
    - nullspace vectors really satisfy M v = 0.
    - rowspace membership works.
    - quotient basis dimension is correct.

E. Logical operators:

    - number of logical pairs equals k = n - rank(H_X) - rank(H_Z).
    - logical X representatives are in ker(H_Z).
    - logical Z representatives are in ker(H_X).
    - logical X representatives are not stabilizers.
    - logical Z representatives are not stabilizers.
    - final pairing matrix is identity.

F. Code parameters:

    - k agrees with the number of logical Pauli pairs.
    - exact d_X,d_Z,d are correct on very small manually checkable examples.
    - distance search never returns a stabilizer as a logical operator.

G. Expansion verifier:

    - test on small graphs where expansion can be manually checked.
    - if a graph fails expansion, return a concrete failing subset witness.

------------------------------------------------------------
11. Documentation
------------------------------------------------------------

Add clear docstrings explaining the mathematical convention.

Especially document:

    - H is B × A.
    - qubit order is (A × A) followed by (B × B).
    - H_X formula.
    - H_Z formula.
    - standard CSS convention for logical X and logical Z.
    - expansion verifier is exact and exponential.
    - distance computation is exact and exponential.
    - random seed is stored for reproducibility.

Avoid using approximate or heuristic methods for expansion and distance unless they are explicitly separate optional functions with names containing "heuristic". The main requested functions must be exact.

Expected final deliverable
============================================================

Produce a clean Python module, preferably with dataclasses for return types, unit tests, and a short example script showing:

    instance = build_quantum_expander_code_instance(
        m=2,
        d_A=3,
        d_B=4,
        seed=12345,
        gamma="1/10",
        delta="1/16",
        verify_expansion=True,
        compute_logicals=True,
        compute_distance=True,
    )

Then print:

    seed,
    n_A,n_B,
    n,
    k,
    d_X,d_Z,d,
    whether CSS commutation holds,
    whether expansion verification passed.