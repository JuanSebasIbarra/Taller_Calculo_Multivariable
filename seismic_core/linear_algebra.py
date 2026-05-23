from __future__ import annotations


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def normal_equations(
    jacobian: list[list[float]], residual: list[float], damping: float
) -> tuple[list[list[float]], list[float]]:
    jt = transpose(jacobian)
    lhs: list[list[float]] = []
    for r, row in enumerate(jt):
        lhs_row = []
        for c, col in enumerate(jt):
            value = dot(row, col)
            if r == c:
                value += damping
            lhs_row.append(value)
        lhs.append(lhs_row)
    rhs = [dot(row, residual) for row in jt]
    return lhs, rhs


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("Sistema singular o mal condicionado")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]
