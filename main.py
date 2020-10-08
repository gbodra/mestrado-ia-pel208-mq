def add_bias(v):
    """
    Inclui bias no vetor de entrada
        :param v: vetor de entrada

        :return: matriz do vetor de entrada com bias
    """
    v_bias = []
    if dataset == "Books_attend_grade":
        v_bias = [[1] + x for x in v]
    else:
        for i in v:
            v_bias.append([1, i])

    return v_bias


def add_bias_quadratic(v):
    """
    Inclui bias + quadratico no vetor de entrada
        :param v: vetor de entrada

        :return: matriz do vetor de entrada com bias e x^2
    """
    v_bias = []
    for item in v:
        if not isinstance(item, list):
            v_bias.append([1, item, item ** 2])
        else:
            quadratic = item.copy()
            for col in range(len(item)):
                quadratic.append(item[col] ** 2)
            v_bias.append([1] + quadratic)

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


def vector_empty(rows):
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

    result_matrix = matrix_empty(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            total = 0
            for ii in range(cols_a):
                total += a[i][ii] * b[ii][j]
            result_matrix[i][j] = total

    return result_matrix


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

    output = vector_empty(rows_matrix)

    for i in range(rows_matrix):
        for j in range(cols_matrix):
            output[i] += matrix[i][j] * vector[j]

    return output


def matrix_minor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def matrix_determinant(m):
    # caso especial para matriz 2x2
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * matrix_determinant(matrix_minor(m, 0, c))
    return determinant


def matrix_inverse(m):
    determinant = matrix_determinant(m)

    # caso especial para matriz 2x2
    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]

    # calcular matriz de cofatores
    cofactors = []

    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = matrix_minor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * matrix_determinant(minor))
        cofactors.append(cofactorRow)

    cofactors = matrix_transpose(cofactors)

    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


def weigthing(weight_list, x_list):
    """
    Retorna vetor com x * pesos
        :param weight_list: pesos
        :param x_list: x

        :return: vetor resultante
    """
    rows_x = len(x_list)

    for i in range(rows_x):
        if isinstance(x_list[0], list):
            for j in range(len(x_list[i])):
                x_list[i][j] = x_list[i][j] * weight_list[i]
        else:
            x_list[i] = x_list[i] * weight_list[i]

    return x_list


def calculate_beta(x_bias):
    """
    Retorna o vetor com beta calculado
        :param x_bias: x já com bias

        :return: vetor resultante
    """
    x_bias_t = matrix_transpose(x_bias)
    Xt_X = matrix_multiply(x_bias_t, x_bias)
    Xt_Y = matrix_vector_multiply(x_bias_t, y_input)
    Xt_X_i = matrix_inverse(Xt_X)
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

    weight_vector = weight(x_bias, y_input, beta)
    x_bias_weighted = weigthing(weight_vector, x_input)
    x_bias_weighted = add_bias(x_bias_weighted)
    x_bias_t = matrix_transpose(x_bias_weighted)
    Xt_X = matrix_multiply(x_bias_t, x_bias_weighted)
    y_weighted = weigthing(weight_vector, y_input)
    Xt_Y = matrix_vector_multiply(x_bias_t, y_weighted)
    Xt_X_i = matrix_inverse(Xt_X)
    beta = matrix_vector_multiply(Xt_X_i, Xt_Y)
    regression_result = matrix_vector_multiply(input_regression, beta)

    return regression_result


def weight(x, y, beta):
    """
    Retorna o cálculo dos pesos
        :param x: entradas
        :param y: saídas
        :param beta: beta calculado

        :return: vetor com pesos calculados
    """
    weight = []
    x_beta = matrix_vector_multiply(x, beta)
    size = len(x_beta)
    for i in range(size):
        w = 1 / (y[i] - x_beta[i])
        weight.append(abs(w))

    return weight


def load_file(filename):
    f = open("data/" + filename + ".txt")
    lines = f.readlines()

    for line in lines:
        if filename == "Books_attend_grade":
            x0, x1, y = line.split(";")
            x_input.append([float(x0), float(x1)])
            y_input.append(float(y))
        else:
            line = line.strip("\n")
            x, y = line.split(";")
            x_input.append(float(x))
            y_input.append(float(y))


# teste com height_shoesize
x_input = []
y_input = []
dataset = "height_shoesize"
load_file(dataset)
height_shoesize_estimation = regression([[1, 70]])
height_shoesize_estimation_quadratic = regression_quadratic([[1, 70, 70 ** 2]])
height_shoesize_estimation_robust = regression_weighted([[1, 70]])

print("height_shoesize_estimation: ", height_shoesize_estimation)
print("height_shoesize_estimation_quadratic: ", height_shoesize_estimation_quadratic)
print("height_shoesize_estimation_robust: ", height_shoesize_estimation_robust)
print("******************************")

# teste com US_Census
x_input = []
y_input = []
dataset = "US_Census"
load_file(dataset)
us_census_estimation = regression([[1, 2010]])
us_census_estimation_quadratic = regression_quadratic([[1, 2010, 2010 ** 2]])
us_census_estimation_robust = regression_weighted([[1, 2010]])

print("us_census_estimation: ", us_census_estimation)
print("us_census_estimation_quadratic: ", us_census_estimation_quadratic)
print("us_census_estimation_robust: ", us_census_estimation_robust)
print("******************************")

# teste com alpswater
x_input = []
y_input = []
dataset = "alpswater"
load_file(dataset)
alpswater_estimation = regression([[1, 31.06]])
alpswater_estimation_quadratic = regression_quadratic([[1, 31.06, 31.06 ** 2]])
alpswater_estimation_robust = regression_weighted([[1, 31.06]])

print("alpswater_estimation: ", alpswater_estimation)
print("alpswater_estimation_quadratic: ", alpswater_estimation_quadratic)
print("alpswater_estimation_robust: ", alpswater_estimation_robust)
print("******************************")

# teste com Books_attend_grade
x_input = []
y_input = []
dataset = "Books_attend_grade"
load_file(dataset)
books_attend_grade_estimation = regression([[1, 2, 15]])
books_attend_grade_estimation_quadratic = regression_quadratic([[1, 2, 15, 2 ** 2, 15 ** 2]])
books_attend_grade_estimation_robust = regression_weighted([[1, 2, 15]])

print("books_attend_grade_estimation: ", books_attend_grade_estimation)
print("books_attend_grade_estimation_quadratic: ", books_attend_grade_estimation_quadratic)
print("books_attend_grade_estimation_robust: ", books_attend_grade_estimation_robust)
print("******************************")