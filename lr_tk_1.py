import numpy as np
import itertools

def REF(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy()  # Создаем копию матрицы, чтобы не изменять оригинал
    m, n = matrix.shape
    current_row = 0

    for col in range(n):
        # Находим ненулевую строку для текущего столбца
        row_with_leading_one = np.argmax(matrix[current_row:m, col]) + current_row

        # Проверяем, есть ли ненулевой элемент
        if matrix[row_with_leading_one, col] == 0:
            continue  # Пропускаем столбцы без ненулевых элементов

        # Меняем строки местами, если это необходимо
        if row_with_leading_one != current_row:
            matrix[[row_with_leading_one, current_row]] = matrix[[current_row, row_with_leading_one]]

        # Обнуляем все элементы ниже ведущего
        for row_below in range(current_row + 1, m):
            if matrix[row_below, col] == 1:
                matrix[row_below] ^= matrix[current_row]

        current_row += 1
        if current_row == m:
            break

    # Удаляем строки, состоящие только из нулей
    return matrix[np.any(matrix, axis=1)]


def rref(matrix):
    # Приведение к ступенчатой форме
    matrix = REF(matrix)
    rows, cols = matrix.shape
    # Обработка строк в обратном порядке
    for current_row in range(rows - 1, -1, -1):
        # Находим позицию ведущего элемента
        leading_entry_index = next((index for index, value in enumerate(matrix[current_row]) if value != 0), -1)

        if leading_entry_index != -1:
            # Обнуляем элементы выше ведущего
            for above_row in range(current_row - 1, -1, -1):
                if matrix[above_row, leading_entry_index] != 0:
                    matrix[above_row] = (matrix[above_row] + matrix[current_row]) % 2
    # Удаление нулевых строк
    while not np.any(matrix[-1]):
        matrix = matrix[:-1, :]
        rows -= 1

    return matrix


def get_lead_columns(matrix):
    leading_indices = []
    for row in matrix:
        leading_index = next((index for index, value in enumerate(row) if value == 1), None)
        if leading_index is not None:
            leading_indices.append(leading_index)

    return leading_indices


def delete_leading_columns(matrix, lead_indices):
    # Преобразуем список в массив NumPy для более удобной обработки
    array_matrix = np.array(matrix)

    # Удаляем указанные ведущие столбцы из массива
    del_matrix = np.delete(array_matrix, lead_indices, axis=1)

    return del_matrix


def build_H_matrix(X, leading_column_indices, total_columns):
    # Определение количества строк в матрице X
    num_rows = np.shape(X)[1]

    # Создание нулевой матрицы размерности (кол-во столбцов) на (кол-во строк)
    H_matrix = np.zeros((total_columns, num_rows), dtype=int)

    # Генерируем единичную матрицу заданного размера
    identity_matrix = np.eye(6, dtype=int)

    # Заполнение ведущих столбцов значениями из матрицы X
    H_matrix[leading_column_indices, :] = X

    # Находим индексы неведущих столбцов
    non_leading_indices = [i for i in range(total_columns) if i not in leading_column_indices]

    # Заполнение нулевых столбцов единичной матрицей
    H_matrix[non_leading_indices, :] = identity_matrix

    return H_matrix


def generate_code_words_summing(matrix):
    zeros = np.zeros(matrix.shape[1], dtype=int)
    final_word = set()
    final_word.add(tuple(zeros.tolist()))

    def add_code_words(current_sum, index):
        if index >= matrix.shape[0]:
            return

        # Генерируем кодовые слова с использованием текущего индекса
        new_sum = (current_sum + matrix[index, :]) % 2
        final_word.add(tuple(new_sum.tolist()))

        # Рекурсивно вызываем для следующего индекса
        add_code_words(new_sum, index + 1)
        # Также продолжаем без добавления текущего индекса
        add_code_words(current_sum, index + 1)

    add_code_words(zeros, 0)

    return np.array(list(final_word))


def generate_binary_words(k):
    return np.array(list(itertools.product(range(2), repeat=k)))


def encode_words(binary_words, G):
    return np.dot(binary_words, G) % 2


def check_codewords(codewords, H):
    results = []
    for codeword in codewords:
        # Убедитесь, что кодовое слово имеет правильный размер, иначе обрезайте
        if codeword.shape[0] != H.shape[0]:
            codeword = codeword[:H.shape[0]]  # Если нужно, используйте только первые элементы

        result = np.dot(codeword, H) % 2
        results.append(result)
    return results

def compute_code_distance(codewords):
    return np.min([np.sum(np.abs(codewords[i] - codewords[j])) for i in range(len(codewords)) for j in range(i + 1, len(codewords))])

def check_for_errors(v, H):
    return (v @ H) % 2

def introduce_error(vector, index):
    erroneous_vector = vector.copy()
    erroneous_vector[index] = 1 - erroneous_vector[index]  # Переключение бита
    return erroneous_vector

#Вывод результатов

matrix = np.array([
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ])
print("input:")
print(matrix)
print("G*:")
ref_matrix = rref(matrix)
print(ref_matrix)
lead_columns = get_lead_columns(ref_matrix)
print(f"lead = {lead_columns}")
print("Сокращенная матрица X:")
X = delete_leading_columns(ref_matrix, lead_columns)
print(X)
print("Матрица H:")
n_cols = np.shape(matrix)[1]
H = build_H_matrix(X, lead_columns, n_cols)
print(H)
print("Кодовые слова:")
print("Способ 1:")
way_1=generate_code_words_summing(ref_matrix)
print(way_1)
print("Способ 2:")

k = matrix.shape[0] # Длина кодового слова
binary_words = generate_binary_words(k)
codewords = encode_words(binary_words, matrix)

# Проверяем кодовые слова
check_results = check_codewords(codewords, H)
way_2=generate_binary_words(k)
# Выводим результат
for i in range(len(binary_words)):
    # print(f"u = {binary_words[i]} -> v = {codewords[i]} -> v@H = {check_results[i]}")
    print(codewords[i])



d = compute_code_distance(way_1)
error_correction_capability = max(1, (d - 1) // 2)

print(f"Кодовое расстояние d = {d}")
print(f"Кратность обнаруживаемой ошибки t = {error_correction_capability}")

# Проверка ошибки кратности t
e1 = np.zeros(n_cols, dtype=int)
e1[2] = 1  # Внесение ошибки в один бит

v = codewords[4]
print(f"Ошибка e1 = {e1}")
print(f"Кодовое слово v = {v}")

v_with_e1 = (v + e1) % 2
print(f"v + e1 = {v_with_e1}")
print(f"(v + e1)@H = {check_for_errors(v_with_e1, H)} - ошибка")
# Проверка ошибки кратности t+1
e2 = np.zeros(n_cols, dtype=int)
e2[6] = 1
e2[9] = 1  # Внесение ошибок в два бита

print(f"Ошибка e2 = {e2}")
v_with_e2 = (v + e2) % 2
print(f"v + e2 = {v_with_e2}")
print(f"(v + e2)@H ={check_for_errors(v_with_e2, H)} - нет ошибки")








