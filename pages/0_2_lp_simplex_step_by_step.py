import streamlit as st
import json
from pathlib import Path
import sympy as sp
import numpy as np
import pandas as pd

# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "saved_lp_problems"

# ======================================================
# Simplex Utilities
# ======================================================

def build_tableau(A, b, c):
    """
    Build initial simplex tableau for:
      max c^T x
      s.t. Ax <= b, x >= 0
    """
    m, n = A.shape
    I = np.eye(m)

    tableau = np.block([
        [A, I, b.reshape(-1, 1)],
        [-c, np.zeros((1, m + 1))]
    ])

    columns = (
        [f"x{i+1}" for i in range(n)] +
        [f"s{i+1}" for i in range(m)] +
        ["RHS"]
    )

    rows = [f"s{i+1}" for i in range(m)] + ["z"]

    return pd.DataFrame(tableau, columns=columns, index=rows)


def choose_entering(T):
    z = T.loc["z"].drop("RHS")
    return z.idxmin() if z.min() < 0 else None


def choose_leaving(T, entering):
    ratios = {}
    for r in T.index[:-1]:
        a = T.loc[r, entering]
        if a > 0:
            ratios[r] = T.loc[r, "RHS"] / a
    return min(ratios, key=ratios.get) if ratios else None


def pivot_with_steps(T, enter, leave):
    T = T.copy()
    steps = []

    pivot = T.loc[leave, enter]
    steps.append(f"R_{leave} ← R_{leave} / {pivot:.2f}")
    T.loc[leave] /= pivot

    for r in T.index:
        if r != leave:
            coef = T.loc[r, enter]
            if coef != 0:
                steps.append(f"R_{r} ← R_{r} − ({coef:.2f})·R_{leave}")
                T.loc[r] -= coef * T.loc[leave]

    T.rename(index={leave: enter}, inplace=True)
    return T, steps


# ======================================================
# LP Parsing
# ======================================================

def parse_expr(expr, sym):
    return sp.sympify(expr, locals=sym)


def parse_constraints_to_Ab(constraints, sym):
    """
    Convert constraints in <= 0 form into Ax <= b.
    Automatically skips non-negativity constraints:
      -x1, -x2
    """
    A = []
    b = []

    x1, x2 = sym["x1"], sym["x2"]

    for raw in constraints.split(","):
        c = raw.strip()
        expr = parse_expr(c, sym)

        # 🚫 Skip implicit non-negativity constraints
        if expr == -x1 or expr == -x2:
            continue

        a1 = float(expr.coeff(x1))
        a2 = float(expr.coeff(x2))
        const = float(-expr.subs({x1: 0, x2: 0}))

        A.append([a1, a2])
        b.append(const)

    return np.array(A), np.array(b)


# ======================================================
# Streamlit App
# ======================================================

def main():
    st.set_page_config(page_title="Simplex from Saved LP", layout="centered")
    st.title("📘 Simplex Method — Load Saved LP (Clean Form)")

    # --------------------------------------------------
    # Load saved problems
    # --------------------------------------------------
    if not SAVE_DIR.exists():
        st.error("Folder `saved_lp_problems` not found.")
        return

    files = sorted(p.stem for p in SAVE_DIR.glob("*.json"))
    if not files:
        st.error("No saved LP problems found.")
        return

    selected = st.selectbox("📂 Choose a saved LP problem", files)

    with open(SAVE_DIR / f"{selected}.json") as f:
        data = json.load(f)

    objective = data["objective"]
    constraints = data["constraints"]

    # --------------------------------------------------
    # Display problem
    # --------------------------------------------------
    st.subheader("Loaded Problem")

    st.latex(r"\min\; " + objective)
    st.markdown("**Constraints (user-defined):**")
    for c in constraints.split(","):
        st.latex(c.strip())

    st.info("ℹ️ Non-negativity constraints (x ≥ 0) are handled implicitly by Simplex.")

    # --------------------------------------------------
    # Build Simplex Model
    # --------------------------------------------------
    x1, x2 = sp.symbols("x1 x2")
    sym = {"x1": x1, "x2": x2}

    f = parse_expr(objective, sym)

    # Convert min → max
    c = -np.array([
        float(f.coeff(x1)),
        float(f.coeff(x2))
    ])

    A, b = parse_constraints_to_Ab(constraints, sym)

    # --------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------
    if "iter" not in st.session_state:
        st.session_state.iter = 0

    if "history" not in st.session_state:
        st.session_state.history = []

    if "tableau" not in st.session_state:
        st.session_state.tableau = build_tableau(A, b, c)

    # --------------------------------------------------
    # Display Tableau
    # --------------------------------------------------
    st.header(f"Iteration {st.session_state.iter}")
    st.dataframe(st.session_state.tableau.round(3))

    # --------------------------------------------------
    # Simplex Step
    # --------------------------------------------------
    entering = choose_entering(st.session_state.tableau)

    if entering is None:
        st.success("Optimal solution reached 🎉")
    else:
        leaving = choose_leaving(st.session_state.tableau, entering)

        st.markdown(
            f"""
            **Entering variable:** `{entering}`  
            **Leaving variable:** `{leaving}`  
            **Pivot element:** {st.session_state.tableau.loc[leaving, entering]:.2f}
            """
        )

        if st.button("➡️ Next Simplex Step"):
            T_new, ops = pivot_with_steps(
                st.session_state.tableau,
                entering,
                leaving
            )

            st.session_state.tableau = T_new
            st.session_state.history.append(ops)
            st.session_state.iter += 1
            st.experimental_rerun()

    # --------------------------------------------------
    # Row Operations
    # --------------------------------------------------
    if st.session_state.history:
        st.header("Row Operations (Last Step)")
        for op in st.session_state.history[-1]:
            st.code(op)

    # --------------------------------------------------
    # Current Basic Solution
    # --------------------------------------------------
    st.header("Current Basic Solution")

    sol = {}
    T = st.session_state.tableau

    for col in T.columns[:-1]:
        sol[col] = T.loc[col, "RHS"] if col in T.index else 0.0

    for k, v in sol.items():
        st.latex(f"{k} = {v:.2f}")

    st.latex(rf"z = {T.loc['z','RHS']:.2f}")

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    if st.button("🔄 Reset Simplex"):
        st.session_state.clear()
        st.experimental_rerun()


# ======================================================
if __name__ == "__main__":
    main()


# import streamlit as st
# import json
# from pathlib import Path
# import sympy as sp
# import numpy as np
# import pandas as pd

# # ======================================================
# # Paths
# # ======================================================

# BASE_DIR = Path(__file__).parent
# SAVE_DIR = BASE_DIR / "saved_lp_problems"

# # ======================================================
# # Simplex Utilities
# # ======================================================

# def build_tableau(A, b, c):
#     m, n = A.shape
#     I = np.eye(m)

#     tableau = np.block([
#         [A, I, b.reshape(-1, 1)],
#         [-c, np.zeros((1, m + 1))]
#     ])

#     columns = (
#         [f"x{i+1}" for i in range(n)] +
#         [f"s{i+1}" for i in range(m)] +
#         ["RHS"]
#     )

#     rows = [f"s{i+1}" for i in range(m)] + ["z"]

#     return pd.DataFrame(tableau, columns=columns, index=rows)


# def choose_entering(T):
#     z = T.loc["z"].drop("RHS")
#     return z.idxmin() if z.min() < 0 else None


# def choose_leaving(T, entering):
#     ratios = {}
#     for r in T.index[:-1]:
#         a = T.loc[r, entering]
#         if a > 0:
#             ratios[r] = T.loc[r, "RHS"] / a
#     return min(ratios, key=ratios.get) if ratios else None


# def pivot_with_steps(T, enter, leave):
#     T = T.copy()
#     steps = []

#     p = T.loc[leave, enter]
#     steps.append(f"R_{leave} ← R_{leave} / {p:.2f}")
#     T.loc[leave] /= p

#     for r in T.index:
#         if r != leave:
#             coef = T.loc[r, enter]
#             if coef != 0:
#                 steps.append(f"R_{r} ← R_{r} − ({coef:.2f})·R_{leave}")
#                 T.loc[r] -= coef * T.loc[leave]

#     T.rename(index={leave: enter}, inplace=True)
#     return T, steps


# # ======================================================
# # LP Parsing
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def parse_constraints_to_Ab(constraints, sym):
#     """
#     Convert constraints in <= 0 form into Ax <= b
#     """
#     A = []
#     b = []

#     x1, x2 = sym["x1"], sym["x2"]

#     for c in constraints.split(","):
#         expr = parse_expr(c.strip(), sym)

#         a1 = float(expr.coeff(x1))
#         a2 = float(expr.coeff(x2))
#         const = float(-expr.subs({x1: 0, x2: 0}))

#         A.append([a1, a2])
#         b.append(const)

#     return np.array(A), np.array(b)


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="Simplex from Saved LP", layout="centered")
#     st.title("📘 Simplex Method — Load Saved LP")

#     # --------------------------------------------------
#     # Load Saved Problems
#     # --------------------------------------------------
#     if not SAVE_DIR.exists():
#         st.error("No saved LP problems found.")
#         return

#     files = sorted(p.stem for p in SAVE_DIR.glob("*.json"))
#     if not files:
#         st.error("No saved LP problems found.")
#         return

#     selected = st.selectbox("📂 Choose a saved LP problem", files)

#     with open(SAVE_DIR / f"{selected}.json") as f:
#         data = json.load(f)

#     objective = data["objective"]
#     constraints = data["constraints"]

#     st.subheader("Loaded Problem")
#     st.latex(r"\min\; " + objective)
#     st.markdown("**Constraints (≤ 0 form):**")
#     for c in constraints.split(","):
#         st.latex(c.strip())

#     # --------------------------------------------------
#     # Build Simplex
#     # --------------------------------------------------
#     x1, x2 = sp.symbols("x1 x2")
#     sym = {"x1": x1, "x2": x2}

#     f = parse_expr(objective, sym)

#     # Convert min → max
#     c = -np.array([
#         float(f.coeff(x1)),
#         float(f.coeff(x2))
#     ])

#     A, b = parse_constraints_to_Ab(constraints, sym)

#     # --------------------------------------------------
#     # Session State Init
#     # --------------------------------------------------
#     if "iter" not in st.session_state:
#         st.session_state.iter = 0

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     if "tableau" not in st.session_state:
#         st.session_state.tableau = build_tableau(A, b, c)

#     # --------------------------------------------------
#     # Display Tableau
#     # --------------------------------------------------
#     st.header(f"Iteration {st.session_state.iter}")
#     st.dataframe(st.session_state.tableau.round(3))

#     # --------------------------------------------------
#     # Simplex Step
#     # --------------------------------------------------
#     entering = choose_entering(st.session_state.tableau)

#     if entering is None:
#         st.success("Optimal solution reached 🎉")
#     else:
#         leaving = choose_leaving(st.session_state.tableau, entering)

#         st.markdown(
#             f"""
#             **Entering variable:** `{entering}`  
#             **Leaving variable:** `{leaving}`  
#             **Pivot:** {st.session_state.tableau.loc[leaving, entering]:.2f}
#             """
#         )

#         if st.button("➡️ Next Simplex Step"):
#             T_new, ops = pivot_with_steps(
#                 st.session_state.tableau,
#                 entering,
#                 leaving
#             )

#             st.session_state.tableau = T_new
#             st.session_state.history.append(ops)
#             st.session_state.iter += 1
#             st.experimental_rerun()

#     # --------------------------------------------------
#     # Row Operations
#     # --------------------------------------------------
#     if st.session_state.history:
#         st.header("Row Operations (Last Step)")
#         for op in st.session_state.history[-1]:
#             st.code(op)

#     # --------------------------------------------------
#     # Reset
#     # --------------------------------------------------
#     if st.button("🔄 Reset Simplex"):
#         st.session_state.clear()
#         st.experimental_rerun()


# # ======================================================
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import numpy as np
# import pandas as pd

# # ======================================================
# # Simplex Utilities
# # ======================================================

# def build_tableau(A, b, c):
#     """
#     max c^T x
#     s.t. Ax <= b, x >= 0
#     """
#     m, n = A.shape
#     I = np.eye(m)

#     tableau = np.block([
#         [A, I, b.reshape(-1, 1)],
#         [-c, np.zeros((1, m + 1))]
#     ])

#     columns = (
#         [f"x{i+1}" for i in range(n)] +
#         [f"s{i+1}" for i in range(m)] +
#         ["RHS"]
#     )

#     rows = [f"s{i+1}" for i in range(m)] + ["z"]

#     return pd.DataFrame(tableau, columns=columns, index=rows)


# def choose_entering(T):
#     z = T.loc["z"].drop("RHS")
#     return z.idxmin() if z.min() < 0 else None


# def choose_leaving(T, entering):
#     ratios = {}
#     for r in T.index[:-1]:
#         a = T.loc[r, entering]
#         if a > 0:
#             ratios[r] = T.loc[r, "RHS"] / a
#     return min(ratios, key=ratios.get) if ratios else None


# def pivot_with_steps(T, enter, leave):
#     T = T.copy()
#     steps = []

#     pivot = T.loc[leave, enter]
#     steps.append(f"R_{leave} ← R_{leave} / {pivot:.2f}")
#     T.loc[leave] /= pivot

#     for r in T.index:
#         if r != leave:
#             coef = T.loc[r, enter]
#             if coef != 0:
#                 steps.append(f"R_{r} ← R_{r} − ({coef:.2f})·R_{leave}")
#                 T.loc[r] -= coef * T.loc[leave]

#     T.rename(index={leave: enter}, inplace=True)
#     return T, steps


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="Simplex Method", layout="centered")
#     st.title("📘 Simplex Method — Step by Step (with Row Operations)")

#     # --------------------------------------------------
#     # Example LP (from your problem)
#     # --------------------------------------------------
#     A = np.array([
#         [1, -3],
#         [-2, -1],
#         [1, 1]
#     ], dtype=float)

#     b = np.array([3, -2, 5], dtype=float)

#     # min (3x1 − 4x2) → max (−3x1 + 4x2)
#     c = np.array([-3, 4], dtype=float)

#     # --------------------------------------------------
#     # Session State Initialization (CRITICAL FIX)
#     # --------------------------------------------------
#     if "iter" not in st.session_state:
#         st.session_state.iter = 0

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     if "tableau" not in st.session_state:
#         st.session_state.tableau = build_tableau(A, b, c)

#     # --------------------------------------------------
#     # Display Tableau
#     # --------------------------------------------------
#     st.header(f"Iteration {st.session_state.iter}")
#     st.dataframe(st.session_state.tableau.round(3))

#     # --------------------------------------------------
#     # Simplex Step
#     # --------------------------------------------------
#     entering = choose_entering(st.session_state.tableau)

#     if entering is None:
#         st.success("Optimal solution reached 🎉")
#     else:
#         leaving = choose_leaving(st.session_state.tableau, entering)

#         st.subheader("Pivot Selection")
#         st.markdown(
#             f"""
#             **Entering variable:** `{entering}`  
#             **Leaving variable:** `{leaving}`  
#             **Pivot element:** {st.session_state.tableau.loc[leaving, entering]:.2f}
#             """
#         )

#         if st.button("➡️ Perform one Simplex step"):
#             T_new, ops = pivot_with_steps(
#                 st.session_state.tableau,
#                 entering,
#                 leaving
#             )

#             st.session_state.tableau = T_new
#             st.session_state.history.append({
#                 "enter": entering,
#                 "leave": leaving,
#                 "ops": ops
#             })
#             st.session_state.iter += 1
#             st.experimental_rerun()

#     # --------------------------------------------------
#     # Row Operations
#     # --------------------------------------------------
#     if st.session_state.history:
#         st.header("Row Operations (Last Step)")
#         for op in st.session_state.history[-1]["ops"]:
#             st.code(op)

#     # --------------------------------------------------
#     # Current Basic Solution
#     # --------------------------------------------------
#     st.header("Current Basic Solution")

#     sol = {}
#     T = st.session_state.tableau

#     for col in T.columns[:-1]:
#         sol[col] = T.loc[col, "RHS"] if col in T.index else 0.0

#     for k, v in sol.items():
#         st.latex(f"{k} = {v:.2f}")

#     st.latex(rf"z = {T.loc['z','RHS']:.2f}")

#     # --------------------------------------------------
#     # Reset Button
#     # --------------------------------------------------
#     if st.button("🔄 Reset Simplex"):
#         st.session_state.clear()
#         st.experimental_rerun()


# # ======================================================
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import sympy as sp
# import numpy as np
# import pandas as pd

# # ======================================================
# # Simplex Utilities
# # ======================================================

# def build_simplex_tableau(A, b, c):
#     """
#     Build initial simplex tableau for:
#       max c^T x
#       Ax <= b, x >= 0
#     """
#     m, n = A.shape

#     # Slack variables
#     I = np.eye(m)
#     tableau = np.block([
#         [A, I, b.reshape(-1, 1)],
#         [-c, np.zeros((1, m + 1))]
#     ])

#     columns = (
#         [f"x{i+1}" for i in range(n)] +
#         [f"s{i+1}" for i in range(m)] +
#         ["RHS"]
#     )

#     index = (
#         [f"s{i+1}" for i in range(m)] +
#         ["z"]
#     )

#     return pd.DataFrame(tableau, columns=columns, index=index)


# def choose_entering_variable(tableau):
#     row = tableau.loc["z"]
#     return row.drop("RHS").idxmin()


# def choose_leaving_variable(tableau, entering):
#     ratios = {}

#     for idx in tableau.index[:-1]:
#         val = tableau.loc[idx, entering]
#         if val > 0:
#             ratios[idx] = tableau.loc[idx, "RHS"] / val

#     return min(ratios, key=ratios.get) if ratios else None


# def pivot(tableau, entering, leaving):
#     pivot_val = tableau.loc[leaving, entering]
#     tableau.loc[leaving] /= pivot_val

#     for row in tableau.index:
#         if row != leaving:
#             tableau.loc[row] -= tableau.loc[row, entering] * tableau.loc[leaving]

#     tableau.rename(index={leaving: entering}, inplace=True)
#     return tableau


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="Simplex Method", layout="centered")
#     st.title("📘 Simplex Method — Step by Step")

#     # --------------------------------------------------
#     # Problem Input
#     # --------------------------------------------------
#     st.subheader("LP in standard form")

#     st.markdown(
#         """
#         We solve LPs of the form:

#         max cᵀx  
#         s.t. Ax ≤ b, x ≥ 0
#         """
#     )

#     A = np.array([
#         [1, -3],
#         [-2, -1],
#         [1, 1]
#     ], dtype=float)

#     b = np.array([3, -2, 5], dtype=float)

#     c = np.array([-3, 4], dtype=float)  # max (-f)

#     # --------------------------------------------------
#     # Initial Tableau
#     # --------------------------------------------------
#     tableau = build_simplex_tableau(A, b, c)

#     if "tableau" not in st.session_state:
#         st.session_state.tableau = tableau.copy()
#         st.session_state.step = 0

#     st.header(f"Step {st.session_state.step}: Simplex Tableau")

#     st.dataframe(st.session_state.tableau.round(3))

#     # --------------------------------------------------
#     # One Simplex Step
#     # --------------------------------------------------
#     if st.button("➡️ Next Simplex Step"):
#         T = st.session_state.tableau

#         entering = choose_entering_variable(T)
#         leaving = choose_leaving_variable(T, entering)

#         if entering is None or leaving is None:
#             st.success("Optimal solution reached 🎉")
#             return

#         st.markdown(
#             f"""
#             **Entering variable:** `{entering}`  
#             **Leaving variable:** `{leaving}`
#             """
#         )

#         T = pivot(T, entering, leaving)

#         st.session_state.tableau = T
#         st.session_state.step += 1

#         st.experimental_rerun()

#     # --------------------------------------------------
#     # Final Solution
#     # --------------------------------------------------
#     if st.session_state.step > 0:
#         st.subheader("Current Basic Solution")

#         solution = {}
#         for col in st.session_state.tableau.columns[:-1]:
#             if col in st.session_state.tableau.index:
#                 solution[col] = st.session_state.tableau.loc[col, "RHS"]
#             else:
#                 solution[col] = 0.0

#         for k, v in solution.items():
#             st.latex(f"{k} = {v:.2f}")

#         z = st.session_state.tableau.loc["z", "RHS"]
#         st.latex(rf"z = {z:.2f}")


# # ======================================================
# if __name__ == "__main__":
#     main()
