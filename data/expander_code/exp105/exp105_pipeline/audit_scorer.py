import numpy as np


def parity_product(matrix, vector):
    matrix = np.asarray(matrix, dtype=np.uint8)
    vector = np.asarray(vector, dtype=np.uint8)
    return np.asarray(matrix @ vector, dtype=np.uint8) & np.uint8(1)


def pairing_score(H_Z, logical_Z, error, correction):
    residual = np.bitwise_xor(error, correction)
    syndrome_match = not parity_product(H_Z, residual).any()
    labels = parity_product(logical_Z, residual)
    return (not syndrome_match) or bool(labels.any()), syndrome_match, labels


def row_echelon_basis(matrix):
    rows = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    pivot_row = 0
    pivots = []
    for column in range(rows.shape[1]):
        candidates = np.flatnonzero(rows[pivot_row:, column])
        if not candidates.size:
            continue
        source = pivot_row + int(candidates[0])
        rows[[pivot_row, source]] = rows[[source, pivot_row]]
        for row in np.flatnonzero(rows[:, column]):
            if row != pivot_row:
                rows[row] ^= rows[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows.shape[0]:
            break
    return rows[:pivot_row], tuple(pivots)


def in_rowspace(vector, echelon_rows, pivots):
    residual = (np.asarray(vector, dtype=np.uint8) & 1).copy()
    for row, pivot in zip(echelon_rows, pivots):
        if residual[pivot]:
            residual ^= row
    return not residual.any()


def rowspace_score(H_Z, H_X_echelon, H_X_pivots, error, correction):
    residual = np.bitwise_xor(error, correction)
    syndrome_match = not parity_product(H_Z, residual).any()
    success = syndrome_match and in_rowspace(residual, H_X_echelon, H_X_pivots)
    return not success, syndrome_match


def trivial_class_generators(H_X, H_Z):
    """Generators of ker(phi_r), built without touching the exp101 frame.

    exp101 builds the label map as W = Z (I xor r_sec H) and asserts that W
    annihilates the stabilizers and im(r_sec) and pairs with the logical moves.
    Those three laws force

        phi_r(v) = 0  <=>  v in rowspace(H_X) + im(r_sec),

    by a dimension count: W has rank k, so its kernel has dimension n - k =
    rank(H_X) + rank(H_Z), which is exactly the dimension of that sum. And
    r_sec places values only on the RREF pivot columns of H_Z and has rank
    rank(H_Z), so im(r_sec) is the span of the unit vectors on those columns.
    The pivot column *set* is a property of the matrix, not of the elimination
    order, so this reconstruction needs nothing from the exp101 section object.

    The replay path scores through these generators so that a bug shared with
    the worker's label basis cannot hide in both.
    """
    H_X = np.asarray(H_X, dtype=np.uint8) & 1
    H_Z = np.asarray(H_Z, dtype=np.uint8) & 1
    _, pivots = row_echelon_basis(H_Z)
    unit_rows = np.zeros((len(pivots), H_Z.shape[1]), dtype=np.uint8)
    for row, column in enumerate(pivots):
        unit_rows[row, column] = 1
    generators = np.vstack((H_X, unit_rows))
    echelon, echelon_pivots = row_echelon_basis(generators)
    return echelon, echelon_pivots


def logical_class_score(generators, error, correction, n):
    """Score one q > 0 trial: does the residual carry a nontrivial class?"""
    echelon, pivots = generators
    residual = np.bitwise_xor(
        np.asarray(error, dtype=np.uint8),
        np.asarray(correction, dtype=np.uint8)[:n],
    )
    return not in_rowspace(residual, echelon, pivots)


def independent_label_map(H_Z, logical_Z):
    """Rebuild phi_r as a k x n matrix without using the exp101 frame.

    The section is defined by its pivot rule: choose the RREF pivot columns of
    H_Z, solve for a chain supported on those columns, and leave the rest zero.
    Reduced row echelon form is unique, so an independent implementation of that
    rule lands on the same matrix rather than on a different valid section --
    which matters, because sections that differ by a logical component do not
    give the same labels (exp101 PHYSICS_CONTRACT section 7.1).
    """
    H_Z = np.asarray(H_Z, dtype=np.uint8) & 1
    logical_Z = np.asarray(logical_Z, dtype=np.uint8) & 1
    num_checks, num_qubits = H_Z.shape
    _, pivots = row_echelon_basis(H_Z)
    rank = len(pivots)
    augmented = np.hstack((
        H_Z[:, list(pivots)], np.eye(num_checks, dtype=np.uint8),
    ))
    reduced, reduced_pivots = row_echelon_basis(augmented)
    if tuple(reduced_pivots[:rank]) != tuple(range(rank)):
        raise ValueError("pivot block is not full column rank")
    solve = reduced[:rank, rank:]
    # r(H_Z v) as a matrix: pivot rows carry solve @ H_Z, everything else zero.
    section_after_H = np.zeros((num_qubits, num_qubits), dtype=np.uint8)
    section_after_H[list(pivots), :] = (
        solve.astype(np.int64) @ H_Z.astype(np.int64) % 2
    ).astype(np.uint8)
    product = (
        logical_Z.astype(np.int64) @ section_after_H.astype(np.int64) % 2
    ).astype(np.uint8)
    return np.bitwise_xor(logical_Z, product)


def apply_label_map(label_map, vector):
    label_map = np.asarray(label_map, dtype=np.uint8)
    vector = np.asarray(vector, dtype=np.uint8)
    return np.asarray(label_map @ vector, dtype=np.uint8) & np.uint8(1)
