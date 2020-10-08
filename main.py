dataset = "US_Census"

if dataset == "US_Census" or dataset == "height_shoesize" or dataset == "Books_attend_grade" or dataset == "alpswater":
    x_input = []
    y_input = []

    f = open("data/" + dataset + ".txt")
    lines = f.readlines()

    for line in lines:
        line = line.strip("\n")
        x, y = line.split(";")
        x_input.append(float(x))
        y_input.append(float(y))
        # x0, x1, y = line.split(";")
        # x_input.append([float(x0), float(x1)])
        # y_input.append(float(y))
else:
    x_input = [69, 67, 71, 65, 72, 68, 74, 65, 66, 72]
    y_input = [9.5, 8.5, 11.5, 10.5, 11, 7.5, 12, 7, 7.5, 13]


def print_matrix(matrix, decimals=4):
    """
    Imprime matriz linha a linha
        :param matrix: matriz que deverá ser impressa
        :param decimals: qtde casas decimais
    """
    for row in matrix:
        print([round(element, decimals) + 0 for element in row])


def add_bias(v):
    """
    Inclui bias no vetor de entrada
        :param v: vetor de entrada

        :return: matriz do vetor de entrada com bias
    """
    v_bias = []
    # v_bias = [[1] + x for x in v]
    for i in v:
        v_bias.append([1, i])

    return v_bias


def add_bias_quadratic(v):
    """
    Inclui bias no vetor de entrada
        :param v: vetor de entrada

        :return: matriz do vetor de entrada com bias e x^2
    """
    v_bias = []
    # v_bias = [[1] + x for x in v]
    for i in v:
        v_bias.append([1, i, i ** 2])

    return v_bias


def matrix_transpose(m):
    """
    Retorna a matriz transposta
        :param m: matriz a ser transposta

        :return: resultado da matriz de entrada transposta
    """
    if not isinstance(m[0], list):
        m = [m]

    rows = len(m)
    cols = len(m[0])

    mt = matrix_empty(cols, rows)

    for i in range(rows):
        for j in range(cols):
            mt[j][i] = m[i][j]

    return mt


def matrix_2x2_i(input_matrix):
    """This is a quick summary line used as a description of the object."""
    formula_2x2_i = 1 / ((input_matrix[0][0] * input_matrix[1][1]) - (input_matrix[0][1]) * input_matrix[1][0])
    Xt_X_i_00 = formula_2x2_i * input_matrix[1][1]
    Xt_X_i_01 = formula_2x2_i * -input_matrix[0][1]
    Xt_X_i_10 = formula_2x2_i * -input_matrix[1][0]
    Xt_X_i_11 = formula_2x2_i * input_matrix[0][0]
    Xt_X_inverse = [[Xt_X_i_00, Xt_X_i_01], [Xt_X_i_10, Xt_X_i_11]]
    return Xt_X_inverse


def matrix_empty(rows, cols):
    """
    Cria uma matriz vazia
        :param rows: número de linhas da matriz
        :param cols: número de colunas da matriz

        :return: matriz preenchida com 0.0
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def zeros_vector(rows):
    """
    Cria um vetor vazio
        :param rows: número de linhas do vetor

        :return: vetor preenchido com 0.0
    """
    M = []
    while len(M) < rows:
        M.append(0.0)

    return M


def matrix_multiply(a, b):
    """
    Retorna o produto da multiplicação da matriz a com b
        :param a: primeira matriz
        :param b: segunda matriz

        :return: matriz resultante
    """
    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    if cols_a != rows_b:
        raise ArithmeticError('O número de colunas da matriz a deve ser igual ao número de linhas da matriz b.')

    C = matrix_empty(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for ii in range(cols_a):
                total += a[i][ii] * b[ii][j]
            C[i][j] = total

    return C


def matrix_vector_multiply(matrix, vector):
    """
        Retorna o produto da multiplicação da matriz com o vetor
            :param matrix: matriz
            :param vector: vetor

            :return: vetor resultante
        """
    rows_matrix = len(matrix)
    cols_matrix = len(matrix[0])

    if cols_matrix != len(vector):
        raise ArithmeticError('O número de colunas da matriz deve ser igual ao número de linhas do vetor.')

    output = zeros_vector(rows_matrix)

    for i in range(rows_matrix):
        for j in range(cols_matrix):
            output[i] += matrix[i][j] * vector[j]

    return output


def weigthing(weight_list, x_list):
    """
        Retorna o produto da multiplicação da matriz com o vetor
            :param weight_list:
            :param x_list: x

            :return: vetor resultante
        """
    rows_x = len(x_list)
    for i in range(rows_x):
        x_list[i] = x_list[i] * weight_list[i]

    return x_list


def calculate_beta(x_bias):
    x_bias_t = matrix_transpose(x_bias)
    Xt_X = matrix_multiply(x_bias_t, x_bias)
    Xt_Y = matrix_vector_multiply(x_bias_t, y_input)
    Xt_X_i = getMatrixInverse(Xt_X)
    beta = matrix_vector_multiply(Xt_X_i, Xt_Y)
    return beta


def regression(input_regression):
    """
        Retorna a estimativa da regressão
            :param input_regression: entrada que precisa ser estimada

            :return: estimativa resultante
        """
    x_bias = add_bias(x_input)
    beta = calculate_beta(x_bias)
    regression_result = matrix_vector_multiply(input_regression, beta)

    return regression_result


def regression_quadratic(input_regression):
    """
        Retorna a estimativa da regressão quadratica
            :param input_regression: entrada que precisa ser estimada

            :return: estimativa resultante
        """
    x_bias = add_bias_quadratic(x_input)
    beta = calculate_beta(x_bias)
    regression_result = matrix_vector_multiply(input_regression, beta)

    return regression_result


def regression_weighted(input_regression):
    """
        Retorna a estimativa da regressão quadratica
            :param input_regression: entrada que precisa ser estimada

            :return: estimativa resultante
        """
    x_bias = add_bias(x_input)
    beta = calculate_beta(x_bias)

    x_bias_t = matrix_transpose(x_bias)
    weight_vector = weight(x_input, y_input, beta)
    x_bias_weighted = weigthing(weight_vector, x_input)
    x_bias_weighted = add_bias(x_bias_weighted)
    Xt_X = matrix_multiply(x_bias_t, x_bias_weighted)
    y_weighted = weigthing(weight_vector, y_input)
    Xt_Y = matrix_vector_multiply(x_bias_t, y_weighted)
    Xt_X_i = getMatrixInverse(Xt_X)
    beta = matrix_vector_multiply(Xt_X_i, Xt_Y)
    regression_result = matrix_vector_multiply(input_regression, beta)

    return regression_result


def weight(x, y, beta):
    size = len(x)
    weight = []
    for i in range(size):
        weight.append(1 / (y[i] - beta[0] + x[i] * beta[1]))

    return weight


# ************* test *************
def getMatrixMinor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def getMatrixDeternminant(m):
    # base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = matrix_transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


# ************* test *************


estimation = regression([[1, 2010]])
estimation_quadratic = regression_quadratic([[1, 2010, 2010 ** 2]])
estimation_robust = regression_weighted([[1, 2010]])

print("estimation:")
print(estimation)
print("estimation_quadratic:")
print(estimation_quadratic)
print("estimation_robust:")
print(estimation_robust)