import streamlit as st
import sympy as sp
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ======================================================
# Persistence
# ======================================================

SAVE_DIR = Path(__file__).parent / "saved_lp_problems"
SAVE_DIR.mkdir(exist_ok=True)

# ======================================================
# Helpers
# ======================================================

def parse_expr(expr, sym):
    return sp.sympify(expr, locals=sym)


def parse_lp_constraints(text, sym):
    """
    Accept:
      x1 - 3*x2 - 3
      -x1
      x1 + x2 <= 5
    All interpreted as <= 0
    """
    constraints = []
    originals = []

    for raw in text.split(","):
        c = raw.strip()
        originals.append(c)

        if "<=" in c:
            lhs, rhs = c.split("<=")
            expr = parse_expr(lhs, sym) - parse_expr(rhs, sym)
        elif ">=" in c:
            lhs, rhs = c.split(">=")
            expr = parse_expr(rhs, sym) - parse_expr(lhs, sym)
        elif "=" in c:
            lhs, rhs = c.split("=")
            expr = parse_expr(lhs, sym) - parse_expr(rhs, sym)
        else:
            expr = parse_expr(c, sym)

        constraints.append(expr)

    return constraints, originals


def intersection(e1, e2, x1, x2):
    sol = sp.solve([e1, e2], (x1, x2), dict=True)
    if sol:
        p = sol[0]
        return float(p[x1]), float(p[x2])
    return None


def is_feasible(p, constraints, tol=1e-6):
    subs = {"x1": p[0], "x2": p[1]}
    return all(float(c.subs(subs)) <= tol for c in constraints)


# ======================================================
# Streamlit App
# ======================================================

def main():
    st.set_page_config(page_title="LP Solver", layout="centered")
    st.title("📘 Linear Programming — Graphical Method (Correct)")

    # --------------------------------------------------
    # Session State
    # --------------------------------------------------
    if "objective" not in st.session_state:
        st.session_state.objective = "3*x1 - 4*x2"

    if "constraints" not in st.session_state:
        st.session_state.constraints = (
            "x1 - 3*x2 - 3, "
            "-2*x1 - x2 + 2, "
            "x1 + x2 - 5, "
            "-x1, -x2"
        )

    # --------------------------------------------------
    # Symbols
    # --------------------------------------------------
    x1, x2 = sp.symbols("x1 x2")
    sym = {"x1": x1, "x2": x2}

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    f_str = st.text_input("Objective (minimize)", st.session_state.objective)
    cons_str = st.text_area("Constraints (≤ 0 form)", st.session_state.constraints)

    st.session_state.objective = f_str
    st.session_state.constraints = cons_str

    if not st.button("🧮 Solve graphically"):
        return

    st.divider()

    f = parse_expr(f_str, sym)
    constraints, originals = parse_lp_constraints(cons_str, sym)

    # ==================================================
    # (a) GRAPHICAL SOLUTION
    # ==================================================
    st.header("a) Draw the feasible set & solve graphically")

    xs = np.linspace(0, 6, 400)
    fig, ax = plt.subplots()

    # Plot constraint boundaries
    for c in constraints:
        try:
            y = sp.solve(c, x2)
            if y:
                y = np.array([float(v.subs(x1, xi)) for xi, v in zip(xs, [y[0]]*len(xs))])
                ax.plot(xs, y)
        except Exception:
            pass

    # ==================================================
    # Compute candidate vertices
    # ==================================================
    points = []
    for c1, c2 in combinations(constraints, 2):
        p = intersection(c1, c2, x1, x2)
        if p and is_feasible(p, constraints):
            points.append(p)

    points = list(set(points))  # unique

    # Plot feasible points
    for p in points:
        ax.plot(p[0], p[1], "ro")

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid()

    st.pyplot(fig)

    # ==================================================
    # (b) CANDIDATE SOLUTIONS
    # ==================================================
    st.header("Candidate vertices")

    if not points:
        st.error("No feasible region.")
        return

    results = []
    for p in points:
        val = f.subs({x1: p[0], x2: p[1]})
        results.append((p, float(val)))

        st.latex(
            rf"x = ({p[0]:.2f}, {p[1]:.2f}),\quad f(x) = {sp.latex(val)}"
        )

    # ==================================================
    # Optimal solution
    # ==================================================
    best = min(results, key=lambda t: t[1])

    st.success(
        f"Optimal solution: x = {best[0]},  f(x) = {best[1]:.2f}"
    )


# ======================================================
if __name__ == "__main__":
    main()


# import streamlit as st
# import sympy as sp
# import json
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================================================
# # Persistence
# # ======================================================

# SAVE_DIR = Path(__file__).parent / "saved_lp_problems"
# SAVE_DIR.mkdir(exist_ok=True)

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def parse_lp_constraints(text, sym):
#     """
#     Parse LP constraints.

#     Accepted formats:
#     1) a*x1 + b*x2 <= c
#     2) a*x1 + b*x2 >= c
#     3) a*x1 + b*x2 - c        (assumed <= 0)
#     """
#     constraints = []
#     originals = []

#     for raw in text.split(","):
#         c = raw.strip()
#         originals.append(c)

#         if "<=" in c:
#             lhs, rhs = c.split("<=")
#             expr = parse_expr(lhs, sym) - parse_expr(rhs, sym)

#         elif ">=" in c:
#             lhs, rhs = c.split(">=")
#             expr = parse_expr(rhs, sym) - parse_expr(lhs, sym)

#         elif "=" in c:
#             lhs, rhs = c.split("=")
#             expr = parse_expr(lhs, sym) - parse_expr(rhs, sym)

#         else:
#             # 👈 NEW: already-normalized constraint
#             expr = parse_expr(c, sym)

#         constraints.append(expr)

#     return constraints, originals


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="LP Solver", layout="centered")
#     st.title("📘 Linear Programming — Step by Step (Exam Style)")

#     # --------------------------------------------------
#     # Session State
#     # --------------------------------------------------
#     if "objective" not in st.session_state:
#         st.session_state.objective = "3*x1 - 4*x2"

#     if "constraints" not in st.session_state:
#         st.session_state.constraints = (
#             "x1 - 3*x2 - 3, "
#             "-2*x1 - x2 + 2, "
#             "x1 + x2 - 5, "
#             "-x1, -x2"
#         )

#     # --------------------------------------------------
#     # Symbols
#     # --------------------------------------------------
#     x1, x2 = sp.symbols("x1 x2")
#     sym = {"x1": x1, "x2": x2}

#     # --------------------------------------------------
#     # Save / Load
#     # --------------------------------------------------
#     st.subheader("📌 Save / Load Problem")

#     save_name = st.text_input("Problem name", "lp_example")
#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("💾 Save"):
#             data = {
#                 "objective": st.session_state.objective,
#                 "constraints": st.session_state.constraints
#             }
#             with open(SAVE_DIR / f"{save_name}.json", "w") as f:
#                 json.dump(data, f, indent=2)
#             st.success("Problem saved")

#     with col2:
#         files = [p.stem for p in SAVE_DIR.glob("*.json")]
#         selected = st.selectbox("Load problem", ["—"] + files)

#         if selected != "—" and st.button("📂 Load"):
#             with open(SAVE_DIR / f"{selected}.json") as f:
#                 data = json.load(f)
#             st.session_state.objective = data["objective"]
#             st.session_state.constraints = data["constraints"]
#             st.experimental_rerun()

#     st.divider()

#     # --------------------------------------------------
#     # Inputs
#     # --------------------------------------------------
#     f_str = st.text_input(
#         "Objective (minimize)",
#         value=st.session_state.objective
#     )

#     cons_str = st.text_area(
#         "Constraints (≤ 0 form recommended)",
#         value=st.session_state.constraints
#     )

#     st.session_state.objective = f_str
#     st.session_state.constraints = cons_str

#     # --------------------------------------------------
#     # Solve
#     # --------------------------------------------------
#     if st.button("🧮 Solve Step-by-Step"):
#         st.divider()

#         f = parse_expr(f_str, sym)
#         constraints, originals = parse_lp_constraints(cons_str, sym)

#         # ==================================================
#         # (a) Graphical Solution
#         # ==================================================
#         st.header("a) Draw the feasible set & solve graphically")

#         x = np.linspace(0, 10, 500)
#         fig, ax = plt.subplots()

#         y_feasible = np.full_like(x, np.inf)

#         for expr in constraints:
#             try:
#                 y = sp.solve(expr.subs(x1, x), x2)
#                 if y:
#                     y = np.array([float(v) for v in y])
#                     ax.plot(x, y[0])
#                     y_feasible = np.minimum(y_feasible, y[0])
#             except Exception:
#                 pass

#         y_feasible = np.maximum(y_feasible, 0)
#         ax.fill_between(x, 0, y_feasible, where=(y_feasible >= 0), alpha=0.3)

#         ax.set_xlim(0, 6)
#         ax.set_ylim(0, 6)
#         ax.set_xlabel("x1")
#         ax.set_ylabel("x2")
#         ax.grid()

#         st.pyplot(fig)

#         # ==================================================
#         # (b) Standard Form
#         # ==================================================
#         st.header("b) Transform the problem into standard form")

#         for i, c in enumerate(originals, 1):
#             st.latex(c + rf" + s_{i} = 0")

#         st.latex(r"x_1, x_2, s_i \ge 0")

#         # ==================================================
#         # (c) Phase I
#         # ==================================================
#         st.header("c) Phase I — find a feasible point")

#         test_point = {x1: 1, x2: 1}
#         st.latex(r"x = (1,1)")

#         feasible = True
#         for c in constraints:
#             val = c.subs(test_point)
#             st.latex(sp.latex(c) + " = " + sp.latex(val))
#             if val > 0:
#                 feasible = False

#         if feasible:
#             st.success("This point is feasible → Phase I complete.")
#         else:
#             st.error("Chosen point is not feasible.")

#         # ==================================================
#         # (d) One Simplex Step
#         # ==================================================
#         st.header("d) Take one simplex step")

#         candidates = [(1, 1), (2, 1), (1, 2)]

#         for v in candidates:
#             val = f.subs({x1: v[0], x2: v[1]})
#             st.latex(rf"x={v},\quad f(x)={sp.latex(val)}")

#         st.success("Move to the vertex with the lowest objective value.")


# # ======================================================
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import sympy as sp
# import json
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================================================
# # Persistence
# # ======================================================

# SAVE_DIR = Path(__file__).parent / "saved_lp_problems"
# SAVE_DIR.mkdir(exist_ok=True)

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def parse_lp_constraints(text, sym):
#     """
#     Parse LP constraints of the form:
#       a*x1 + b*x2 <= c
#       a*x1 + b*x2 >= c
#     """
#     constraints = []
#     originals = []

#     for c in text.split(","):
#         c = c.strip()
#         originals.append(c)

#         if "<=" in c:
#             lhs, rhs = c.split("<=")
#             expr = parse_expr(lhs, sym) - parse_expr(rhs, sym)
#             constraints.append(expr)

#         elif ">=" in c:
#             lhs, rhs = c.split(">=")
#             expr = parse_expr(rhs, sym) - parse_expr(lhs, sym)
#             constraints.append(expr)

#         else:
#             raise ValueError(f"Invalid constraint: {c}")

#     return constraints, originals


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="LP Solver", layout="centered")
#     st.title("📘 Linear Programming — Step by Step (Exam Style)")

#     # --------------------------------------------------
#     # Session State
#     # --------------------------------------------------
#     if "objective" not in st.session_state:
#         st.session_state.objective = "3*x1 - 4*x2"

#     if "constraints" not in st.session_state:
#         st.session_state.constraints = (
#             "x1 - 3*x2 <= 3, "
#             "-2*x1 - x2 <= -2, "
#             "x1 + x2 <= 5, "
#             "x1 >= 0, x2 >= 0"
#         )

#     # --------------------------------------------------
#     # Symbols
#     # --------------------------------------------------
#     x1, x2 = sp.symbols("x1 x2")
#     sym = {"x1": x1, "x2": x2}

#     # --------------------------------------------------
#     # Save / Load
#     # --------------------------------------------------
#     st.subheader("📌 Save / Load Problem")

#     save_name = st.text_input("Problem name", "lp_example")
#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("💾 Save"):
#             data = {
#                 "objective": st.session_state.objective,
#                 "constraints": st.session_state.constraints
#             }
#             with open(SAVE_DIR / f"{save_name}.json", "w") as f:
#                 json.dump(data, f, indent=2)
#             st.success("Problem saved")

#     with col2:
#         files = [p.stem for p in SAVE_DIR.glob("*.json")]
#         selected = st.selectbox("Load problem", ["—"] + files)

#         if selected != "—" and st.button("📂 Load"):
#             with open(SAVE_DIR / f"{selected}.json") as f:
#                 data = json.load(f)
#             st.session_state.objective = data["objective"]
#             st.session_state.constraints = data["constraints"]
#             st.experimental_rerun()

#     st.divider()

#     # --------------------------------------------------
#     # Inputs
#     # --------------------------------------------------
#     f_str = st.text_input(
#         "Objective (minimize)",
#         value=st.session_state.objective
#     )

#     cons_str = st.text_area(
#         "Constraints",
#         value=st.session_state.constraints
#     )

#     st.session_state.objective = f_str
#     st.session_state.constraints = cons_str

#     # --------------------------------------------------
#     # Solve
#     # --------------------------------------------------
#     if st.button("🧮 Solve Step-by-Step"):
#         st.divider()

#         f = parse_expr(f_str, sym)
#         constraints, originals = parse_lp_constraints(cons_str, sym)

#         # ==================================================
#         # (a) Graphical Solution
#         # ==================================================
#         st.header("a) Draw the feasible set & solve graphically")

#         x = np.linspace(0, 10, 400)
#         fig, ax = plt.subplots()

#         y_feasible = np.full_like(x, np.inf)

#         for expr in constraints:
#             y = sp.solve(expr.subs(x1, x), x2)
#             if y:
#                 y = np.array([float(v) for v in y])
#                 ax.plot(x, y[0])
#                 y_feasible = np.minimum(y_feasible, y[0])

#         y_feasible = np.maximum(y_feasible, 0)
#         ax.fill_between(x, 0, y_feasible, where=(y_feasible >= 0), alpha=0.3)

#         ax.set_xlim(0, 6)
#         ax.set_ylim(0, 6)
#         ax.set_xlabel("x1")
#         ax.set_ylabel("x2")
#         ax.grid()

#         st.pyplot(fig)

#         st.markdown(
#             """
#             **Key idea:**  
#             For linear programs, the optimum lies at a **vertex** of the feasible region.
#             """
#         )

#         # ==================================================
#         # (b) Standard Form
#         # ==================================================
#         st.header("b) Transform the problem into standard form")

#         st.markdown("Add slack variables to convert inequalities into equalities:")

#         for i, c in enumerate(originals, 1):
#             st.latex(c.replace("<=", "+ s_%d =" % i))

#         st.latex(r"x_1, x_2, s_i \ge 0")

#         # ==================================================
#         # (c) Phase I
#         # ==================================================
#         st.header("c) Phase I — find a feasible point")

#         st.markdown(
#             """
#             A feasible point can often be found by inspection.
#             """
#         )

#         test_point = {x1: 1, x2: 1}
#         st.latex(r"x = (1,1)")

#         feasible = True
#         for c in constraints:
#             val = c.subs(test_point)
#             st.latex(sp.latex(c) + " = " + sp.latex(val))
#             if val > 0:
#                 feasible = False

#         if feasible:
#             st.success("This point is feasible → Phase I complete.")
#         else:
#             st.error("Chosen point is not feasible.")

#         # ==================================================
#         # (d) One Simplex Step
#         # ==================================================
#         st.header("d) Take one simplex step")

#         st.markdown(
#             """
#             Evaluate the objective function at nearby vertices and move
#             in the direction of improvement.
#             """
#         )

#         candidates = [(1, 1), (2, 1), (1, 2)]

#         for v in candidates:
#             val = f.subs({x1: v[0], x2: v[1]})
#             st.latex(
#                 rf"x = {v},\quad f(x) = {sp.latex(val)}"
#             )

#         st.success(
#             "Choose the vertex with the lowest objective value → one simplex step."
#         )


# # ======================================================
# if __name__ == "__main__":
#     main()
