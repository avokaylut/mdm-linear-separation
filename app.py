import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(45)


def mdm_soft_separation(P1, P2, C=1.0, tol=1e-6, max_iter=20000):
    """
    Реализация MDM-алгоритма для мягкого линейного разделения двух множеств

    Параметры:
    P1, P2 - массивы точек
    C - параметр штрафа за ошибки
    tol - критерий остановки

    Возвращает:
    w - вектор весов
    beta - смещение
    eta - вектор ошибок
    iterations - число выполненных итераций
    """
    P1, P2 = np.asarray(P1, float), np.asarray(P2, float)
    if P1.ndim != 2 or P2.ndim != 2 or P1.shape[1] != P2.shape[1] or not len(P1) or not len(P2):
        raise ValueError("Нужны два непустых массива одинаковой размерности")
    if not np.isfinite(P1).all() or not np.isfinite(P2).all() or not np.isfinite(C) or C <= 0 or tol <= 0:
        raise ValueError("Точки должны быть конечными, C и tol — положительными")
    m1, n = P1.shape
    m2 = len(P2)
    m = m1 + m2
    A = np.column_stack([P1.T, -P2.T])
    mass = min(m1, m2) * C / 2
    u = np.r_[np.full(m1, mass / m1), np.full(m2, mass / m2)]

    v = A @ u  # Вектор v0 = A*u0
    iterations = 0

    while True:
        if iterations >= max_iter:
            raise RuntimeError("Достигнут предел итераций; измените данные или C")
        iterations += 1
        J1 = [i for i in range(m1) if u[i] == 0]  # Точки P1 с u[i]=0
        J2 = [i for i in range(m1) if 0 < u[i] < C]  # Точки P1 с 0<u[i]<C
        J3 = [i for i in range(m1) if u[i] == C]  # Точки P1 с u[i]=C
        J4 = [i for i in range(m1, m) if u[i] == 0]  # Точки P2 с u[i]=0
        J5 = [i for i in range(m1, m) if 0 < u[i] < C]  # Точки P2 с 0<u[i]<C
        J6 = [i for i in range(m1, m) if u[i] == C]  # Точки P2 с u[i]=C

        Delta1 = 0
        if (J2 or J3) and (J1 or J2):
            max_J2J3 = max(np.dot(v, A[:, i]) for i in J2 + J3)
            min_J1J2 = min(np.dot(v, A[:, i]) for i in J1 + J2)
            Delta1 = max_J2J3 - min_J1J2

        Delta2 = 0
        if (J5 or J6) and (J4 or J5):
            max_J5J6 = max(np.dot(v, A[:, i]) for i in J5 + J6)
            min_J4J5 = min(np.dot(v, A[:, i]) for i in J4 + J5)
            Delta2 = max_J5J6 - min_J4J5

        Delta = max(Delta1, Delta2)
        if Delta <= tol:
            break

        if Delta == Delta1: # Для случая Δ(u) = Δ1(u):
            i_prime = (J1 + J2)[np.argmin([np.dot(v, A[:, i]) for i in J1 + J2])] # i' = argmin <v,p_i> // индекс столбца матрицы, который имеет минимальное скалярное произведение с вектором v
            i_dprime = (J2 + J3)[np.argmax([np.dot(v, A[:, i]) for i in J2 + J3])]  # i'' = argmax <v,p_i>
            direction = A[:, i_prime] - A[:, i_dprime] # p_i' - p_i''
        else: # Для случая Δ(u) = Δ2(u):
            i_prime = (J4 + J5)[np.argmin([np.dot(v, A[:, i]) for i in J4 + J5])] # i' = argmin <-v,p_i>
            i_dprime = (J5 + J6)[np.argmax([np.dot(v, A[:, i]) for i in J5 + J6])] # i'' = argmax <-v,p_i>
            direction = A[:, i_prime] - A[:, i_dprime] # p_i'' - p_i' направление

        norm_sq = np.dot(direction, direction)
        if norm_sq <= np.finfo(float).eps:
            break
        lambda_ = min(Delta / norm_sq, u[i_dprime], C - u[i_prime])  # Ограничение λ (усечение)

        u[i_prime] = np.clip(u[i_prime] + lambda_, 0, C)  # точка с "наихудшим" выполнением условия
        u[i_dprime] = np.clip(u[i_dprime] - lambda_, 0, C) # точка с "наилучшим" выполнением условия
        v = A @ u

    w = v  # w_* = v_*
    beta = compute_beta(u, A, m1, w,C) # β_* вычисляется по опорным векторам
    eta = np.maximum(0, 1 - np.array([np.dot(w, p) + beta for p in P1] +[-np.dot(w, p) - beta for p in P2]))

    return w, beta, eta, iterations


def compute_beta(u, A, m1, w, C=1.0):
    """Вычисление смещения beta по опорным векторам"""
    SV = [i for i in range(len(u)) if 0 < u[i] < C]
    if not SV:
        return -0.5 * (np.median(w @ A[:, :m1]) - np.median(w @ A[:, m1:]))

    beta_values = []
    for i in SV:
        if i < m1:  # Точки из P1
            beta_values.append(1 - np.dot(w, A[:, i]))
        else:  # Точки из P2
            beta_values.append(-1 + np.dot(w, A[:, i]))

    return np.median(beta_values)



def generate_data(n_points, dim):
    if not random_mode.get():
        np.random.seed(45)
    P1 = np.random.randn(n_points, dim) * 0.5 + np.ones(dim)
    P2 = np.random.randn(n_points, dim) * 0.5
    return P1, P2


def visualize(P1, P2, w, beta):

    if np.linalg.norm(w) < 1e-10:
        messagebox.showinfo("Результат", "Нормаль близка к нулю: разделяющая граница не определена")
        return
    dim = P1.shape[1]
    if dim == 2:
        fig = plt.figure(figsize=(8, 6), facecolor="#E6E6FA")
        ax = fig.add_subplot(111)
        ax.scatter(P1[:, 0], P1[:, 1], c='red', label='P1')
        ax.scatter(P2[:, 0], P2[:, 1], c='blue', label='P2')

        x_min = min(P1[:, 0].min(), P2[:, 0].min()) - 1
        x_max = max(P1[:, 0].max(), P2[:, 0].max()) + 1
        x = np.linspace(x_min, x_max, 100)
        if np.abs(w[1]) > 1e-6:
            y = (-w[0] * x - beta) / w[1]
            y1 = (-w[0] * x - beta + 1) / w[1]
            y2 = (-w[0] * x - beta - 1) / w[1]
            plt.plot(x, y, 'k--', label='Разделяющая гиперплоскость')
            plt.plot(x, y1, color='g', alpha=0.5)
            plt.plot(x, y2, color='g', alpha=0.5)
            plt.fill_between(x, y1, y2, color='g', alpha=0.25)
        elif abs(w[0]) > 1e-10:
            x_val = -beta / w[0]
            plt.axvline(x_val, color='k', label='Разделяющая гиперплоскость')
            plt.axvline(x_val - 1 / w[0], color='g', alpha=0.5)
            plt.axvline(x_val + 1 / w[0], color='g', alpha=0.5)
            plt.axvspan(x_val - 1 / w[0], x_val + 1 / w[0], color='g', alpha=0.25)

        plt.title("Результат MDM-разделения (2D)")
        plt.grid()
        plt.legend()
        plt.show()

    elif dim == 3:
        fig = plt.figure(figsize=(10, 7), facecolor ="#E6E6FA" )
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#E6E6FA")
        ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], c='red', label='P1')
        ax.scatter(P2[:, 0], P2[:, 1], P2[:, 2], c='blue', label='P2')

        xx, yy = np.meshgrid(
            np.linspace(-3, 3, 10), np.linspace(-3, 3, 10)
        )
        if abs(w[2]) > 1e-6:
            zz = (-w[0] * xx - w[1] * yy - beta) / w[2]
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='g')
        ax.set_title("Результат MDM-разделения (3D)")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()


def run_mdm():
    if (not entry_points.get().strip() or not entry_c.get().strip() or not combo_dim.get().strip()):
        messagebox.showwarning("Внимание", "Заполните все поля")
        return

    try:
        n = int(entry_points.get())
        dim = int(combo_dim.get())
        c = float(entry_c.get())
        if n <= 0 or n > 1000 or not np.isfinite(c) or c <= 0:
            raise ValueError()
    except ValueError:
        messagebox.showwarning("Параметры", "Укажите от 1 до 1000 точек и положительное C")
        return


    P1, P2 = generate_data(n, dim)
    try:
        w, beta, eta, iterations = mdm_soft_separation(P1, P2, C=c)
    except (ValueError, RuntimeError) as error:
        messagebox.showwarning("Расчёт", str(error))
        return
    margin = 2 / np.linalg.norm(w) if np.linalg.norm(w) > 1e-10 else float("inf")
    X = np.vstack([P1, P2])
    y_true = np.hstack([np.ones(len(P1)), -np.ones(len(P2))])
    y_pred = np.sign(np.dot(X, w) + beta)
    accuracy = np.mean(y_pred == y_true)
    result_text = (
        f"Вектор весов w: {np.array2string(w, precision=3, separator=', ')}\n"
        f"Смещение beta: {beta:.4f}\n"
        f"Число итераций: {iterations}\n"
        f"Сумма ошибок η: {np.sum(eta):.4f}\n"
        f"Ширина полосы: {margin:.4f}\n"
        f"Доля правильно классифицированных: {accuracy:.2%}"
    )
    result_label.config(text=result_text)

    visualize(P1, P2, w, beta)


def validate_positive_int(text):
    return text.isdigit() and int(text) > 0 or text == ""

def validate_positive_float(text):
    if text == "" :
        return True
    if text.count('.') > 1 or text[0] == '.' or text == "0":
        return False
    for c in text:
        if not (c.isdigit() or c == '.'):
            return False
    return True

def toggle_random():
    random_mode.set(not random_mode.get())
    if random_mode.get():
        random_btn.config(relief="sunken", bg="#C6F7C6", text="Случайные данные при каждом запуске: ВКЛ")
    else:
        random_btn.config(relief="raised", bg="#E6E6FA", text="Случайные данные при каждом запуске: ВЫКЛ")

if __name__ == "__main__":
    root = tk.Tk()
    root.title('MDM Soft Separation')
    root.configure(bg="#E6E6FA")  # Лавандовый цвет
    root.geometry("505x510")
    root.resizable(False, False)

    random_mode = tk.BooleanVar(value=False)

    font_big = ("Arial", 18)
    font_medium = ("Arial", 16)
    font_small = ("Consolas", 13)

    vcmd_int = (root.register(validate_positive_int), '%P')
    vcmd_float = (root.register(validate_positive_float), '%P')

    tk.Label(root, text="Число точек на класс:", bg="#E6E6FA", font=font_big).grid(row=0, column=0, sticky='e', pady=10, padx=10)
    entry_points = tk.Entry(root, validate='key', validatecommand=vcmd_int, font=font_big, justify='center', width=8)
    entry_points.insert(0, "100")
    entry_points.grid(row=0, column=1, pady=10, padx=10)

    tk.Label(root, text="Размерность:", bg="#E6E6FA", font=font_big).grid(row=1, column=0, sticky='e', pady=10, padx=10)
    combo_dim = ttk.Combobox(root, values=["2", "3"], font=font_big, state="readonly", justify='center', width=8)
    combo_dim.current(0)
    combo_dim.grid(row=1, column=1, pady=10, padx=10)

    tk.Label(root, text="Штраф C:", bg="#E6E6FA", font=font_big).grid(row=3, column=0, sticky='e', pady=10, padx=10)
    entry_c = tk.Entry(root, validate='key', validatecommand=vcmd_float, font=font_big, justify='center', width=8)
    entry_c.insert(0, "1.0")
    entry_c.grid(row=3, column=1, pady=10, padx=10)


    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    random_btn = tk.Button(root,text="Случайные данные при каждом запуске: ВЫКЛ",font=font_small,bg="#E6E6FA",command=toggle_random)
    random_btn.grid(row=4, column=0, columnspan=2, pady=5)

    btn = tk.Button(root, text="Запустить MDM", command=run_mdm, font=font_medium, bg="#D6C6F7", activebackground="#C6B6E7")
    btn.grid(row=5, column=0, columnspan=2, pady=20, sticky='ew')  # Кнопка по центру и растягивается по ширине

    result_label = tk.Label(root, text="", bg="#E6E6FA", justify='left', anchor='center', font=font_small)
    result_label.grid(row=6, column=0, columnspan=2, sticky='n', padx=8, pady=(0, 0))


    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws // 2) - (w // 2)
    y = (hs // 2) - (h // 2)
    root.geometry(f'+{x}+{y}')

    root.mainloop()
