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
