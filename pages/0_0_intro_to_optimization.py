import streamlit as st
import sympy as sp
import json
from pathlib import Path

# ======================================================
# Persistence
# ======================================================

SAVE_DIR = Path(__file__).parent / "saved_problems"
SAVE_DIR.mkdir(exist_ok=True)

# ======================================================
# Helpers
# ======================================================

def parse_expr(expr, sym):
    return sp.sympify(expr, locals=sym)


def parse_constraints(text, sym):
    """
    Parse constraints of the form:
      g(x) >= 0
      h(x) = 0

    We KEEP the constraint exactly as written.
    """
    constraints = []
    names = []
    types = []
    originals = []

    for i, c in enumerate(text.split(",")):
        c = c.strip()

        if ">=" in c:
            lhs, rhs = c.split(">=")
            g = parse_expr(lhs, sym) - parse_expr(rhs, sym)
            constraints.append(g)
            names.append(f"g{i+1}(x)")
            types.append("ineq")
            originals.append(c)

        elif "=" in c:
            lhs, rhs = c.split("=")
            h = parse_expr(lhs, sym) - parse_expr(rhs, sym)
            constraints.append(h)
            names.append(f"h{i+1}(x)")
            types.append("eq")
            originals.append(c)

        else:
            raise ValueError(f"Invalid constraint format: {c}")

    return constraints, names, types, originals


# ======================================================
# Streamlit App
# ======================================================

def main():
    st.set_page_config(page_title="KKT Solver", layout="centered")
    st.title("📘 Step-by-Step KKT Solver (Exact Lecture Form)")

    # --------------------------------------------------
    # Session State
    # --------------------------------------------------
    if "objective" not in st.session_state:
        st.session_state.objective = (
            "2*x1**2 + 2*x1*x2 + x2**2 - 10*x1 - 10*x2"
        )

    if "constraints" not in st.session_state:
        st.session_state.constraints = (
            "5 - x1**2 - x2**2 >= 0, 6 - 3*x1 - x2 >= 0"
        )

    # --------------------------------------------------
    # Symbols
    # --------------------------------------------------
    x1, x2 = sp.symbols("x1 x2")
    sym = {"x1": x1, "x2": x2}

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    st.subheader("📌 Save / Load Problem")

    save_name = st.text_input("Problem name", "example_problem")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save"):
            data = {
                "objective": st.session_state.objective,
                "constraints": st.session_state.constraints
            }
            with open(SAVE_DIR / f"{save_name}.json", "w") as f:
                json.dump(data, f, indent=2)
            st.success("Problem saved")

    with col2:
        files = [p.stem for p in SAVE_DIR.glob("*.json")]
        selected = st.selectbox("Load problem", ["—"] + files)

        if selected != "—" and st.button("📂 Load"):
            with open(SAVE_DIR / f"{selected}.json") as f:
                data = json.load(f)
            st.session_state.objective = data["objective"]
            st.session_state.constraints = data["constraints"]
            st.experimental_rerun()

    st.divider()

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    f_str = st.text_input(
        "Objective (minimize)",
        value=st.session_state.objective
    )

    cons_str = st.text_area(
        "Constraints (>= 0 form)",
        value=st.session_state.constraints
    )

    st.session_state.objective = f_str
    st.session_state.constraints = cons_str

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    if st.button("🧮 Solve Step-by-Step"):
        st.divider()

        f = parse_expr(f_str, sym)
        g, g_names, g_types, originals = parse_constraints(cons_str, sym)

        lambdas = sp.symbols(f"\\lambda_1:{len(g)+1}")

        # ==================================================
        # a) KKT CONDITIONS
        # ==================================================
        st.header("a) Write down the KKT conditions")

        # --------------------------------------------------
        # Step 1: Original Problem
        # --------------------------------------------------
        st.subheader("Step 1: Original Problem")

        st.latex(r"\min\; " + sp.latex(f))
        st.markdown("**Subject to:**")
        for orig in originals:
            st.latex(orig)

        # --------------------------------------------------
        # Step 2: Lagrangian (YOUR EXACT FORM)
        # --------------------------------------------------
        st.subheader("Step 2: Form the Lagrangian")

        L = f
        for lam, gi in zip(lambdas, g):
            L -= lam * gi   # <-- EXACTLY AS YOU REQUESTED

        st.latex(r"\mathcal{L}(x,\lambda) = " + sp.latex(L))

        # --------------------------------------------------
        # Step 3: Gradients
        # --------------------------------------------------
        st.subheader("Step 3: Gradients")

        grad_L = [sp.diff(L, x1), sp.diff(L, x2)]

        st.latex(r"\frac{\partial \mathcal{L}}{\partial x_1} = " + sp.latex(grad_L[0]))
        st.latex(r"\frac{\partial \mathcal{L}}{\partial x_2} = " + sp.latex(grad_L[1]))

        # --------------------------------------------------
        # Step 4: KKT Conditions
        # --------------------------------------------------
        st.subheader("Step 4: The KKT Conditions")

        st.markdown("**1. Stationarity**")
        for eq in grad_L:
            st.latex(sp.latex(eq) + " = 0")

        st.markdown("**2. Primal Feasibility**")
        for name, gi in zip(g_names, g):
            st.latex(name + "(x) = " + sp.latex(gi) + r"\ge 0")

        st.markdown("**3. Dual Feasibility**")
        for lam in lambdas:
            st.latex(sp.latex(lam) + r"\ge 0")

        st.markdown("**4. Complementary Slackness**")
        for lam, gi in zip(lambdas, g):
            st.latex(sp.latex(lam * gi) + " = 0")

        # --------------------------------------------------
        # Inactive Constraint Case
        # --------------------------------------------------
        st.divider()
        st.header("b) Case: All Constraints Inactive")

        inactive = {lam: 0 for lam in lambdas}
        st.latex(r"\lambda_1 = \cdots = \lambda_m = 0")

        eqs = [eq.subs(inactive) for eq in grad_L]
        sol = sp.solve(eqs, (x1, x2), dict=True)

        if not sol:
            st.error("No stationary point found.")
            return

        sol = sol[0]

        st.markdown("**Candidate point:**")
        st.latex(
            r"x^* = \left(" +
            sp.latex(sol[x1]) + "," +
            sp.latex(sol[x2]) + r"\right)"
        )

        # --------------------------------------------------
        # Feasibility Check
        # --------------------------------------------------
        st.subheader("Feasibility Check")

        violated = False
        for name, gi in zip(g_names, g):
            val = gi.subs(sol)
            st.latex(name + r"(x^*) = " + sp.latex(val))
            if val < 0:
                violated = True
                st.markdown("❌ Constraint violated")

        if violated:
            st.error(
                "Conclusion: The unconstrained minimum is **not feasible**.\n\n"
                "The solution must lie on the **boundary of the feasible region**."
            )
        else:
            st.success("Unconstrained minimum is feasible.")

# ======================================================
if __name__ == "__main__":
    main()


# import streamlit as st
# import sympy as sp
# import json
# from pathlib import Path

# # ======================================================
# # Persistence
# # ======================================================

# SAVE_DIR = Path(__file__).parent / "saved_problems"
# SAVE_DIR.mkdir(exist_ok=True)

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def normalize_constraint(text, sym):
#     """
#     Normalize constraints to standard KKT form: c(x) <= 0
#     Returns:
#       - c(x)
#       - type: 'ineq' or 'eq'
#       - original string
#     """
#     text = text.strip()

#     if "<=" in text:
#         lhs, rhs = text.split("<=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "ineq", text

#     if ">=" in text:
#         lhs, rhs = text.split(">=")
#         return parse_expr(rhs, sym) - parse_expr(lhs, sym), "ineq", text

#     if "=" in text:
#         lhs, rhs = text.split("=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "eq", text

#     raise ValueError(f"Invalid constraint: {text}")


# def parse_constraints(text, sym):
#     constraints, names, types, originals = [], [], [], []

#     for i, c in enumerate(text.split(",")):
#         ci, t, orig = normalize_constraint(c, sym)
#         constraints.append(ci)
#         names.append(f"c{i+1}(x)")
#         types.append(t)
#         originals.append(orig)

#     return constraints, names, types, originals


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="KKT Solver", layout="centered")
#     st.title("📘 Step-by-Step KKT Solver (Textbook Correct)")

#     # --------------------------------------------------
#     # Session State
#     # --------------------------------------------------
#     if "objective" not in st.session_state:
#         st.session_state.objective = (
#             "2*x1**2 + 2*x1*x2 + x2**2 - 10*x1 - 10*x2"
#         )

#     if "constraints" not in st.session_state:
#         st.session_state.constraints = (
#             "5 - x1**2 - x2**2 >= 0, 6 - 3*x1 - x2 >= 0"
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

#     save_name = st.text_input("Problem name", "example_problem")
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
#         c, c_names, c_types, originals = parse_constraints(cons_str, sym)
#         lambdas = sp.symbols(f"\\lambda_1:{len(c)+1}")

#         # ==================================================
#         # a) KKT CONDITIONS
#         # ==================================================
#         st.header("a) Write down the KKT conditions")

#         # --------------------------------------------------
#         # Constraint normalization
#         # --------------------------------------------------
#         st.subheader("Step 1: Normalize Constraints")

#         st.markdown(
#             "We first rewrite all constraints in the standard KKT form "
#             r"$c_i(x) \le 0$."
#         )

#         for name, orig, ci in zip(c_names, originals, c):
#             st.latex(
#                 r"\text{Original: } " + orig +
#                 r"\quad\Rightarrow\quad " +
#                 name + "(x)=" + sp.latex(ci) + r"\le 0"
#             )

#         # --------------------------------------------------
#         # Lagrangian
#         # --------------------------------------------------
#         st.subheader("Step 2: Form the Lagrangian")

#         L = f
#         for lam, ci in zip(lambdas, c):
#             L -= lam * ci

#         st.latex(r"\mathcal{L}(x,\lambda)=" + sp.latex(L))

#         # --------------------------------------------------
#         # Gradients
#         # --------------------------------------------------
#         st.subheader("Step 3: Gradients")

#         grad_L = [sp.diff(L, x1), sp.diff(L, x2)]
#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_1}=" + sp.latex(grad_L[0]))
#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_2}=" + sp.latex(grad_L[1]))

#         # --------------------------------------------------
#         # KKT Conditions
#         # --------------------------------------------------
#         st.subheader("Step 4: The Four KKT Conditions")

#         st.markdown("**1. Stationarity**")
#         for eq in grad_L:
#             st.latex(sp.latex(eq) + "=0")

#         st.markdown("**2. Primal Feasibility**")
#         for name, ci in zip(c_names, c):
#             st.latex(name + "(x)=" + sp.latex(ci) + r"\le 0")

#         st.markdown("**3. Dual Feasibility**")
#         for lam in lambdas:
#             st.latex(sp.latex(lam) + r"\ge 0")

#         st.markdown("**4. Complementary Slackness**")
#         for lam, ci in zip(lambdas, c):
#             st.latex(sp.latex(lam * ci) + "=0")

#         # --------------------------------------------------
#         # Inactive constraint case
#         # --------------------------------------------------
#         st.divider()
#         st.header("b) Case: All Constraints Inactive")

#         inactive = {lam: 0 for lam in lambdas}
#         st.latex(r"\lambda_1=\cdots=\lambda_m=0")

#         eqs = [eq.subs(inactive) for eq in grad_L]
#         sol = sp.solve(eqs, (x1, x2), dict=True)

#         if not sol:
#             st.error("No stationary point found.")
#             return

#         sol = sol[0]

#         st.markdown("**Candidate point:**")
#         st.latex(
#             r"x^*=\left(" +
#             sp.latex(sol[x1]) + "," +
#             sp.latex(sol[x2]) + r"\right)"
#         )

#         # --------------------------------------------------
#         # Feasibility check
#         # --------------------------------------------------
#         st.subheader("Feasibility Check")

#         violated = False
#         for name, ci in zip(c_names, c):
#             val = ci.subs(sol)
#             st.latex(name + r"(x^*)=" + sp.latex(val))
#             if val > 0:
#                 violated = True
#                 st.markdown("❌ Constraint violated")

#         if violated:
#             st.error(
#                 "Conclusion: The unconstrained minimum is **not feasible**.\n\n"
#                 "The solution must lie on the **boundary of the feasible region**."
#             )
#         else:
#             st.success("Unconstrained minimum is feasible.")

# # ======================================================
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import sympy as sp
# import json
# from pathlib import Path

# # ======================================================
# # Persistence
# # ======================================================

# SAVE_DIR = Path(__file__).parent / "saved_problems"
# SAVE_DIR.mkdir(exist_ok=True)

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def normalize_constraint(text, sym):
#     """
#     Normalize constraints to c(x) <= 0
#     Supports <=, >=, =
#     """
#     if "<=" in text:
#         lhs, rhs = text.split("<=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "ineq"

#     if ">=" in text:
#         lhs, rhs = text.split(">=")
#         return parse_expr(rhs, sym) - parse_expr(lhs, sym), "ineq"

#     if "=" in text:
#         lhs, rhs = text.split("=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "eq"

#     raise ValueError(f"Invalid constraint: {text}")


# def parse_constraints(text, sym):
#     constraints, names, types = [], [], []

#     for i, c in enumerate(text.split(",")):
#         g, t = normalize_constraint(c.strip(), sym)
#         constraints.append(g)
#         names.append(f"c{i+1}(x)")
#         types.append(t)

#     return constraints, names, types


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="KKT Solver", layout="centered")
#     st.title("📘 Step-by-Step KKT Solver (with Save / Load)")

#     # --------------------------------------------------
#     # Session State Init
#     # --------------------------------------------------
#     if "objective" not in st.session_state:
#         st.session_state.objective = (
#             "2*x1**2 + 2*x1*x2 + x2**2 - 10*x1 - 10*x2"
#         )

#     if "constraints" not in st.session_state:
#         st.session_state.constraints = (
#             "5 - x1**2 + x2**2 >= 0, 6 - 3*x1 - x2 >= 0"
#         )

#     # --------------------------------------------------
#     # Symbols
#     # --------------------------------------------------
#     x1, x2 = sp.symbols("x1 x2")
#     sym = {"x1": x1, "x2": x2}

#     # --------------------------------------------------
#     # Save / Load UI
#     # --------------------------------------------------
#     st.subheader("📌 Save / Load Problem")

#     save_name = st.text_input("Problem name", "example_problem")

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("💾 Save"):
#             data = {
#                 "objective": st.session_state.objective,
#                 "constraints": st.session_state.constraints
#             }
#             path = SAVE_DIR / f"{save_name}.json"
#             with open(path, "w") as f:
#                 json.dump(data, f, indent=2)
#             st.success(f"Saved as {path.name}")

#     with col2:
#         files = [p.stem for p in SAVE_DIR.glob("*.json")]
#         selected = st.selectbox("Load saved problem", ["—"] + files)

#         if selected != "—" and st.button("📂 Load"):
#             path = SAVE_DIR / f"{selected}.json"
#             with open(path) as f:
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

#     # Sync state
#     st.session_state.objective = f_str
#     st.session_state.constraints = cons_str

#     # --------------------------------------------------
#     # Solve
#     # --------------------------------------------------
#     if st.button("🧮 Solve Step-by-Step"):
#         st.divider()

#         f = parse_expr(f_str, sym)
#         c, c_names, c_types = parse_constraints(cons_str, sym)

#         lambdas = sp.symbols(f"λ1:{len(c)+1}")

#         # ==================================================
#         # a) KKT CONDITIONS
#         # ==================================================
#         st.header("a) Write down the KKT conditions")

#         st.subheader("Step 1: Lagrangian")

#         L = f
#         for lam, ci in zip(lambdas, c):
#             L -= lam * ci   # your convention

#         st.latex(r"\mathcal{L}(x,\lambda) = " + sp.latex(L))

#         st.subheader("Step 2: Gradients")

#         grad_L = [sp.diff(L, x1), sp.diff(L, x2)]
#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_1} = " + sp.latex(grad_L[0]))
#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_2} = " + sp.latex(grad_L[1]))

#         st.subheader("Step 3: The Four KKT Conditions")

#         st.markdown("**1. Stationarity**")
#         for eq in grad_L:
#             st.latex(sp.latex(eq) + "=0")

#         st.markdown("**2. Primal Feasibility**")
#         for name, ci in zip(c_names, c):
#             st.latex(name + "(x)=" + sp.latex(ci) + r"\le 0")

#         st.markdown("**3. Dual Feasibility**")
#         for lam in lambdas:
#             st.latex(sp.latex(lam) + r"\ge 0")

#         st.markdown("**4. Complementary Slackness**")
#         for lam, ci in zip(lambdas, c):
#             st.latex(sp.latex(lam * ci) + "=0")

#         # ==================================================
#         # b) INACTIVE CONSTRAINT CASE
#         # ==================================================
#         st.divider()
#         st.header("b) If all constraints are inactive")

#         inactive = {lam: 0 for lam in lambdas}
#         st.latex(r"\lambda_1=\cdots=\lambda_m=0")

#         eqs = [eq.subs(inactive) for eq in grad_L]
#         sol = sp.solve(eqs, (x1, x2), dict=True)

#         if not sol:
#             st.error("No stationary point found.")
#             return

#         sol = sol[0]

#         st.subheader("Solve Stationarity Equations")
#         for i, eq in enumerate(eqs, 1):
#             st.latex(f"(Eq {i})\\quad " + sp.latex(eq) + "=0")

#         st.markdown("**Candidate point:**")
#         st.latex(
#             r"x^*=\left(" +
#             sp.latex(sol[x1]) + "," +
#             sp.latex(sol[x2]) + r"\right)"
#         )

#         # ==================================================
#         # Feasibility Check
#         # ==================================================
#         st.subheader("Check Feasibility")

#         violated = False
#         for name, ci in zip(c_names, c):
#             val = ci.subs(sol)
#             st.latex(name + r"(x^*)=" + sp.latex(val))
#             if val > 0:
#                 violated = True
#                 st.markdown("❌ Constraint violated")

#         if violated:
#             st.error(
#                 "Conclusion: The unconstrained minimum is **not feasible**.\n\n"
#                 "Therefore, the solution must lie on the **boundary**."
#             )
#         else:
#             st.success("Unconstrained minimum is feasible.")

# # ======================================================
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import sympy as sp

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expr(expr, sym):
#     return sp.sympify(expr, locals=sym)


# def normalize_constraint(text, sym):
#     """
#     Converts constraints to c(x) <= 0 form
#     Returns (c(x), type)
#     """
#     if "<=" in text:
#         lhs, rhs = text.split("<=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "ineq"

#     if ">=" in text:
#         lhs, rhs = text.split(">=")
#         return parse_expr(rhs, sym) - parse_expr(lhs, sym), "ineq"

#     if "=" in text:
#         lhs, rhs = text.split("=")
#         return parse_expr(lhs, sym) - parse_expr(rhs, sym), "eq"

#     raise ValueError(f"Invalid constraint: {text}")


# def parse_constraints(text, sym):
#     constraints = []
#     names = []
#     types = []

#     for i, c in enumerate(text.split(",")):
#         c = c.strip()
#         g, t = normalize_constraint(c, sym)
#         constraints.append(g)
#         names.append(f"c{i+1}(x)")
#         types.append(t)

#     return constraints, names, types


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(layout="centered")
#     st.title("📘 Step-by-Step KKT Solver (Correct & Robust)")

#     # Symbols
#     x1, x2 = sp.symbols("x1 x2")
#     sym = {"x1": x1, "x2": x2}

#     # Inputs
#     f_str = st.text_input(
#         "Objective (minimize)",
#         "2*x1**2 + 2*x1*x2 + x2**2 - 10*x1 - 10*x2"
#     )

#     cons_str = st.text_area(
#         "Constraints",
#         "5 - x1**2 + x2**2 >= 0, 6 - 3*x1 - x2 >= 0"
#     )

#     if st.button("Solve Step-by-Step"):
#         st.divider()

#         # Parse
#         f = parse_expr(f_str, sym)
#         c, c_names, c_types = parse_constraints(cons_str, sym)

#         lambdas = sp.symbols(f"λ1:{len(c)+1}")

#         # ==================================================
#         # a) KKT CONDITIONS
#         # ==================================================
#         st.header("a) Write down the KKT conditions")

#         st.subheader("Step 1: Lagrangian")

#         L = f
#         for lam, ci in zip(lambdas, c):
#             L -= lam * ci   # <-- YOUR convention

#         st.latex(r"\mathcal{L}(x,\lambda) = " + sp.latex(L))

#         st.subheader("Step 2: Gradients")

#         grad_L = [sp.diff(L, x1), sp.diff(L, x2)]

#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_1} = " + sp.latex(grad_L[0]))
#         st.latex(r"\frac{\partial \mathcal{L}}{\partial x_2} = " + sp.latex(grad_L[1]))

#         st.subheader("Step 3: The Four KKT Conditions")

#         st.markdown("**1. Stationarity**")
#         for eq in grad_L:
#             st.latex(sp.latex(eq) + "=0")

#         st.markdown("**2. Primal Feasibility**")
#         for name, ci in zip(c_names, c):
#             st.latex(name + "(x)=" + sp.latex(ci) + r"\le 0")

#         st.markdown("**3. Dual Feasibility**")
#         for lam in lambdas:
#             st.latex(sp.latex(lam) + r"\ge 0")

#         st.markdown("**4. Complementary Slackness**")
#         for lam, ci in zip(lambdas, c):
#             st.latex(sp.latex(lam * ci) + "=0")

#         # ==================================================
#         # b) INACTIVE CONSTRAINT CASE
#         # ==================================================
#         st.divider()
#         st.header("b) If both constraints are inactive")

#         st.markdown(
#             "If both constraints are inactive, then all Lagrange multipliers are zero:"
#         )

#         inactive = {lam: 0 for lam in lambdas}
#         st.latex(r"\lambda_1=\lambda_2=0")

#         eqs = [eq.subs(inactive) for eq in grad_L]

#         sol = sp.solve(eqs, (x1, x2), dict=True)

#         if not sol:
#             st.error("No stationary point found.")
#             return

#         sol = sol[0]

#         st.subheader("Solve Stationarity Equations")
#         for i, eq in enumerate(eqs, 1):
#             st.latex(f"(Eq {i})\\quad " + sp.latex(eq) + "=0")

#         st.markdown("**Candidate point:**")
#         st.latex(
#             r"x^*=\left(" +
#             sp.latex(sol[x1]) + "," +
#             sp.latex(sol[x2]) + r"\right)"
#         )

#         # ==================================================
#         # FEASIBILITY CHECK
#         # ==================================================
#         st.subheader("Check Feasibility")

#         violated = False
#         for name, ci in zip(c_names, c):
#             val = ci.subs(sol)
#             st.latex(name + r"(x^*)=" + sp.latex(val))
#             if val > 0:
#                 violated = True
#                 st.markdown("❌ Constraint violated")

#         if violated:
#             st.error(
#                 "Conclusion: The unconstrained minimum is **not feasible**.\n\n"
#                 "Therefore, the solution must lie on the **boundary**."
#             )
#         else:
#             st.success("Unconstrained minimum is feasible.")

# # ======================================================
# if __name__ == "__main__":
#     main()



# import streamlit as st
# from sympy import symbols, Eq, diff, solve

# def main():
#     st.title("Optimization KKT Conditions Solver")

#     # Define variables
#     x = symbols('x')
#     y = symbols('y')

#     # Objective function (example - modify as needed)
#     objective = x**2 + y  # Minimize this

#     # Constraints (modify these to your problem)
#     constraint1 = Eq(x + y, 5)  # Changed from inequality to equality for example
#     constraint2 = Eq(-x, 0)

#     st.write("### Problem Definition")
#     st.write(f"Minimize: {objective}")
#     st.write(f"Constraints:")
#     st.write(f"- Constraint 1: {constraint1}")
#     st.write(f"- Constraint 2: {constraint2}")

#     # Calculate gradients
#     def get_gradients():
#         grad_x = diff(objective, x)
#         grad_y = diff(objective, y)

#         # For constraints (simplified - in real KKT you'd need proper constraint gradients)
#         cons_grad_x1 = diff(constraint1, x)  # Should be 1 for this example
#         cons_grad_y1 = diff(constraint1, y)  # Should be 1 for this example

#         return grad_x, grad_y, cons_grad_x1, cons_grad_y1

#     grad_x, grad_y, cons_grad_x1, cons_grad_y1 = get_gradients()

#     st.write("### Stationarity Conditions")
#     st.write(f"∂f/∂x = {grad_x}")
#     st.write(f"∂f/∂y = {grad_y}")

#     # Solve unconstrained minimum
#     def solve_unconstrained():
#         return solve((grad_x, grad_y), (x, y))

#     x_star, y_star = solve_unconstrained()
#     candidate_point = (x_star.evalf(), y_star.evalf())
#     st.write(f"Unconstrained minimum: ({candidate_point[0]:.2f}, {candidate_point[1]:.2f})")

#     # Check feasibility
#     def check_feasibility(x_val, y_val):
#         feasible = True

#         if not constraint1.subs({x: x_val, y: y_val}).evalf():
#             st.error("Constraint 1 violated!")
#             feasible = False

#         if not constraint2.subs({x: x_val, y: y_val}).evalf():
#             st.error("Constraint 2 violated!")
#             feasible = False

#         return feasible

#     is_feasible = check_feasibility(x_star.evalf(), y_star.evalf())
#     st.write(f"Is solution feasible? {is_feasible}")

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import sympy as sp

# # ======================================================
# # Helpers
# # ======================================================

# def parse_expression(expr_str, symbols_dict):
#     """Safely parse string into a SymPy expression"""
#     return sp.sympify(expr_str, locals=symbols_dict)


# def parse_constraints(constraints_str, symbols_dict):
#     """
#     Parse constraints of form:
#     x1 + x2 - 5 <= 0
#     x1 - 1 = 0
#     """
#     constraints = []
#     types = []  # 'ineq' or 'eq'

#     for line in constraints_str.split(","):
#         line = line.strip()

#         if "<=" in line:
#             lhs, rhs = line.split("<=")
#             g = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
#             constraints.append(g)
#             types.append("ineq")

#         elif "=" in line:
#             lhs, rhs = line.split("=")
#             h = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
#             constraints.append(h)
#             types.append("eq")

#         else:
#             raise ValueError(f"Invalid constraint format: {line}")

#     return constraints, types


# def is_feasible(sol, constraints, types, tol=1e-6):
#     """Check primal feasibility"""
#     for g, t in zip(constraints, types):
#         val = g.subs(sol)
#         val = float(val)

#         if t == "ineq" and val > tol:
#             return False
#         if t == "eq" and abs(val) > tol:
#             return False

#     return True


# # ======================================================
# # Streamlit App
# # ======================================================

# def main():
#     st.set_page_config(page_title="KKT Solver", layout="centered")
#     st.title("KKT Optimization Solver (Educational)")

#     st.markdown(
#         """
#         This tool **symbolically constructs and solves KKT conditions**  
#         for small constrained optimization problems.

#         - Supports ≤ and = constraints
#         - Solves symbolically using SymPy
#         - Intended for **learning and inspection**, not large-scale optimization
#         """
#     )

#     # --------------------------------------------------
#     # Inputs
#     # --------------------------------------------------
#     problem_type = st.radio("Problem Type", ["Minimize", "Maximize"])

#     n_vars = st.slider("Number of variables", 1, 3, 2)
#     var_names = [f"x{i+1}" for i in range(n_vars)]
#     x = sp.symbols(var_names)
#     symbols_dict = dict(zip(var_names, x))

#     obj_str = st.text_input(
#         "Objective function",
#         value="x1**2 + x2**2"
#     )

#     constraints_str = st.text_area(
#         "Constraints (comma-separated)",
#         value="x1 + x2 - 5 <= 0, -x1 <= 0"
#     )

#     # --------------------------------------------------
#     # Solve Button
#     # --------------------------------------------------
#     if st.button("Solve KKT"):
#         try:
#             # Parse objective
#             f = parse_expression(obj_str, symbols_dict)
#             if problem_type == "Maximize":
#                 f = -f

#             # Parse constraints
#             constraints, types = parse_constraints(constraints_str, symbols_dict)

#             # Lagrange multipliers
#             lambdas = sp.symbols(f"lambda0:{len(constraints)}")

#             # Build Lagrangian
#             L = f
#             for lam, g in zip(lambdas, constraints):
#                 L += lam * g

#             # --------------------------------------------------
#             # Display Lagrangian
#             # --------------------------------------------------
#             st.subheader("Lagrangian")
#             st.latex(sp.latex(L))

#             # --------------------------------------------------
#             # KKT Conditions
#             # --------------------------------------------------
#             stationarity = [sp.diff(L, xi) for xi in x]
#             complementary = [
#                 lam * g if t == "ineq" else g
#                 for lam, g, t in zip(lambdas, constraints, types)
#             ]

#             equations = stationarity + complementary

#             st.subheader("KKT Conditions")

#             st.markdown("**Stationarity**")
#             for eq in stationarity:
#                 st.latex(sp.latex(eq))

#             st.markdown("**Complementary Slackness / Equality Constraints**")
#             for eq in complementary:
#                 st.latex(sp.latex(eq))

#             # --------------------------------------------------
#             # Solve System
#             # --------------------------------------------------
#             sol = sp.solve(
#                 equations,
#                 list(x) + list(lambdas),
#                 dict=True
#             )

#             if not sol:
#                 st.warning("No symbolic KKT solutions found.")
#                 return

#             # --------------------------------------------------
#             # Display Feasible Solutions
#             # --------------------------------------------------
#             st.subheader("Feasible KKT Solutions")

#             found = False
#             for i, s in enumerate(sol, 1):
#                 if not is_feasible(s, constraints, types):
#                     continue

#                 found = True

#                 primal = {
#                     str(xi): float(s[xi])
#                     for xi in x
#                     if xi in s
#                 }

#                 multipliers = {
#                     str(lam): float(s[lam])
#                     for lam in lambdas
#                     if lam in s
#                 }

#                 st.markdown(f"### Solution {i}")
#                 st.markdown("**Primal Variables**")
#                 st.json(primal)

#                 st.markdown("**Lagrange Multipliers**")
#                 st.json(multipliers)

#             if not found:
#                 st.warning("KKT candidates found, but none are feasible.")

#         except Exception as e:
#             st.error(f"Error: {e}")


# if __name__ == "__main__":
#     main()


# import streamlit as st
# import sympy as sp

# # ----------------------------
# # Helpers
# # ----------------------------
# def parse_expression(expr_str, symbols_dict):
#     """Safely parse string into SymPy expression"""
#     return sp.sympify(expr_str, locals=symbols_dict)


# def parse_constraints(constraints_str, symbols_dict):
#     """
#     Parse constraints of form:
#     x1 + x2 - 5 <= 0
#     x1 - 1 <= 0
#     """
#     constraints = []
#     for line in constraints_str.split(","):
#         line = line.strip()
#         if "<=" in line:
#             lhs, rhs = line.split("<=")
#             expr = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
#             constraints.append(expr)  # g(x) <= 0
#         elif "=" in line:
#             lhs, rhs = line.split("=")
#             expr = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
#             constraints.append(expr)  # h(x) = 0
#         else:
#             raise ValueError(f"Invalid constraint: {line}")
#     return constraints


# # ----------------------------
# # Streamlit App
# # ----------------------------
# def main():
#     st.title("KKT Optimization Solver (Educational)")

#     problem_type = st.radio("Problem Type", ["Minimize", "Maximize"])

#     n_vars = st.slider("Number of variables", 1, 3, 2)
#     var_names = [f"x{i+1}" for i in range(n_vars)]
#     x = sp.symbols(var_names)
#     symbols_dict = dict(zip(var_names, x))

#     obj_str = st.text_input("Objective function", "x1**2 + x2**2")

#     constraints_str = st.text_area(
#         "Constraints (comma separated, use <= or =)",
#         "x1 + x2 - 5 <= 0, -x1 <= 0"
#     )

#     if st.button("Solve KKT"):
#         try:
#             # Parse objective
#             f = parse_expression(obj_str, symbols_dict)
#             if problem_type == "Maximize":
#                 f = -f

#             # Parse constraints
#             constraints = parse_constraints(constraints_str, symbols_dict)

#             # Lagrange multipliers
#             lambdas = sp.symbols(f"lambda0:{len(constraints)}")

#             # Lagrangian
#             L = f
#             for lam, g in zip(lambdas, constraints):
#                 L += lam * g

#             st.subheader("Lagrangian")
#             st.latex(sp.latex(L))

#             # Stationarity conditions
#             stationarity = [sp.diff(L, xi) for xi in x]

#             # Complementary slackness
#             comp_slackness = [lam * g for lam, g in zip(lambdas, constraints)]

#             # Build system
#             equations = stationarity + comp_slackness

#             st.subheader("KKT Conditions")
#             for eq in equations:
#                 st.latex(sp.latex(eq))

#             # Solve
#             sol = sp.solve(equations, list(x) + list(lambdas), dict=True)

#             if not sol:
#                 st.warning("No symbolic KKT solution found.")
#                 return

#             st.subheader("Candidate Solutions")
#             for s in sol:
#                 st.write(s)

#         except Exception as e:
#             st.error(f"Error: {e}")


# if __name__ == "__main__":
#     main()


# import streamlit as st
# from sympy import symbols, diff, Eq, solve

# def parse_constraints(constraints_str):
#     constraints = []
#     for constraint in constraints_str.split(","):
#         constraint = constraint.strip()
#         if "=" in constraint:
#             # Equality constraint (e.g., x + y = 5)
#             lhs, rhs = constraint.split("=")
#             lhs = lhs.strip()
#             rhs = rhs.strip().replace("=", "-")
#             constraints.append((lhs, rhs))
#         else:
#             # Inequality constraint (e.g., x + y <= 5) -> convert to Eq form
#             parts = constraint.split("<=")
#             if len(parts) != 2:
#                 raise ValueError(f"Invalid inequality: {constraint}")
#             lhs, rhs = parts
#             constraints.append((lhs.strip(), f"-{rhs.strip()}"))
#     return constraints

# def main():
#     st.title("Optimization KKT Solver")

#     # Inputs
#     problem_type = st.radio("Problem Type:", ["Minimize", "Maximize"])
#     obj_func_str = st.text_input("Objective Function (e.g., x^2 + y):")
#     constraints_str = st.text_area(
#         "Constraints (e.g., 'x + y <= 5, -x >= 0'):",
#         value="x + y <= 5, -x <= 0"
#     )
#     n_vars = st.slider("Number of Variables:", min_value=1, max_value=3, value=2)
#     var_names = [f"x{i+1}" for i in range(n_vars)]

#     # Parse inputs
#     x = symbols(var_names)

#     if problem_type == "Maximize":
#         obj_func_str = "-" + obj_func_str  # Convert to minimize form

#     constraints = parse_constraints(constraints_str)
#     inequality_constraints = []
#     equality_constraints = []

#     for lhs, rhs in constraints:
#         if "=" in lhs:  # Equality constraint
#             equality_constraints.append(Eq(eval(lhs), eval(rhs)))
#         else:  # Inequality constraint (e.g., x + y <= 5 -> -x - y >= 0)
#             inequality_constraints.append(Eq(0, eval(lhs) - eval(rhs)))

#     # Compute gradients
#     def compute_gradients():
#         return [diff(obj_func_str, var) for var in x]

#     grad_obj = compute_gradients()
#     st.write("Objective Gradient:", grad_obj)

#     # KKT Conditions: Stationarity (∇f(x*) + Σ λ_i ∇g_i(x*) = 0)
#     def kkt_conditions(grad_obj):
#         stationarity_eqs = []
#         for i, constraint in enumerate(constraints):
#             if "=" in constraint.lhs:  # Skip equality constraints
#                 continue
#             lhs, rhs = constraint.split("<=")[:2]
#             grad_g = diff(eval(lhs), x)
#             stationarity_eqs.append(Eq(grad_obj[i], -eval(rhs)))
#         return stationarity_eqs

#     stationarity_eqs = kkt_conditions(grad_obj)

#     # Solve KKT system (simplified example)
#     try:
#         # Example: Assume a simple problem
#         x_star = solve(stationarity_eqs, dict=True)[0]  # Placeholder for actual solver

#         # Check feasibility
#         feasible = True
#         for i, constraint in enumerate(constraints):
#             if "=" in constraint.lhs:
#                 continue  # Skip equality constraints
#             lhs, rhs = constraint.split("<=")[:2]
#             val = eval(lhs).subs(dict(zip(var_names, x_star)))
#             if not (val <= eval(rhs)):
#                 st.error(f"Violates constraint: {lhs} ≤ {rhs}")
#                 feasible = False

#         if feasible:
#             st.success("Feasible KKT solution found!")
#             return x_star
#         else:
#             st.warning("Unconstrained min violates constraints.")
#     except Exception as e:
#         st.error(f"Error solving KKT: {e}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from sympy import symbols, diff, Eq, Inequality, solve, lambdify
# import numpy as np

# # --- Streamlit UI ---
# st.title("General KKT Conditions Solver")
# st.markdown("""
# This app solves optimization problems using Karush-Kuhn-Tucker (KKT) conditions.
# Enter your objective function and constraints below.
# """)

# # --- User Inputs ---
# problem_type = st.radio(
#     "Optimization Type:",
#     ("Minimize", "Maximize")
# )

# # Objective function (e.g., x^2 + y)
# obj_func_str = st.text_input("Objective Function (e.g., x**2 + y):", value="x**2 + y")

# # Constraints (e.g., x + y <= 5, x >= 0)
# constraints_str = st.text_area(
#     "Constraints (Inequality: g_i(x) ≤ 0; Equality: h_j(x) = 0)",
#     value="x + y <= 5, -x <= 0, -y <= 0"
# )

# # Number of variables
# n_vars = st.slider("Number of Variables:", min_value=1, max_value=3, value=2)
# var_names = [f"x{i+1}" for i in range(n_vars)]
# st.write(f"Variables: {', '.join(var_names)}")

# # --- Parse Inputs ---
# def parse_constraints(constraints_str):
#     constraints = []
#     for constraint in constraints_str.split(","):
#         constraint = constraint.strip()
#         if "=" in constraint:
#             # Equality constraint (e.g., x + y = 5)
#             lhs, rhs = constraint.split("=")
#             lhs = lhs.strip()
#             rhs = rhs.strip().replace("=", "-")
#             constraints.append((lhs, rhs))
#         else:
#             # Inequality constraint (e.g., x + y <= 5)
#             parts = constraint.split("<=")
#             if len(parts) != 2:
#                 raise ValueError(f"Invalid inequality: {constraint}")
#             lhs, rhs = parts
#             constraints.append((lhs.strip(), f"-{rhs.strip()}"))
#     return constraints

# # --- Symbolic Math Setup ---
# x = symbols(var_names)

# # Parse objective function
# obj_func = obj_func_str.replace(" ", "")
# if problem_type == "Maximize":
#     obj_func = "-" + obj_func  # Convert to minimize form

# # Parse constraints
# constraints = parse_constraints(constraints_str)
# inequality_constraints = []
# equality_constraints = []

# for lhs, rhs in constraints:
#     if "=" in lhs:  # Equality constraint
#         equality_constraints.append(Eq(eval(lhs), eval(rhs)))
#     else:  # Inequality constraint
#         inequality_constraints.append(Inequality(eval(lhs), eval(rhs)))

# # --- Compute Gradients ---
# def compute_gradients():
#     grad_obj = [diff(obj_func, var) for var in x]
#     return grad_obj

# grad_obj = compute_gradients()
# st.write("Objective Gradient:", grad_obj)

# # --- KKT Conditions ---
# def kkt_conditions(grad_obj):
#     # Stationarity: ∇f(x*) + Σ λ_i ∇g_i(x*) = 0
#     stationarity_eqs = []
#     for i, (lhs, rhs) in enumerate(constraints):
#         if "=" in lhs:
#             continue  # Skip equality constraints for now
#         grad_g = diff(eval(lhs), x)
#         stationarity_eqs.append(Eq(grad_obj[i], -eval(rhs)))

#     return stationarity_eqs

# stationarity_eqs = kkt_conditions(grad_obj)
# st.write("Stationarity Conditions:", stationarity_eqs)

# # --- Solve KKT System ---
# def solve_kkt():
#     # For demo, assume we can solve the system
#     try:
#         # Example: Assume a simple problem (replace with actual solver)
#         x_star = np.array([0.0, 5.0])  # Placeholder from your earlier example

#         # Check feasibility
#         feasible = True
#         for i, (lhs, rhs) in enumerate(constraints):
#             if "=" in lhs:
#                 continue  # Skip equality constraints
#             val = eval(lhs).subs(dict(zip(var_names, x_star)))
#             if not (val <= eval(rhs)):
#                 st.error(f"Violates constraint: {lhs} ≤ {rhs}")
#                 feasible = False

#         if feasible:
#             st.success("Feasible KKT solution found!")
#             return x_star
#         else:
#             st.warning("Unconstrained min violates constraints.")
#     except Exception as e:
#         st.error(f"Error solving KKT: {e}")

# x_star = solve_kkt()

# # --- Visualization (Optional) ---
# if n_vars == 2 and len(x) == 2:
#     import matplotlib.pyplot as plt

#     def plot_constraints():
#         fig, ax = plt.subplots()
#         for lhs, rhs in constraints:
#             if "=" not in lhs:  # Skip equality constraints
#                 expr = eval(lhs)
#                 ax.plot(*expr.doit().plot((-10, 10), (0, 10))[0], label=lhs)

#         ax.scatter(x_star[0], x_star[1], color="red", label="KKT Point")
#         ax.legend()
#         st.pyplot(fig)

#     plot_constraints()

# # --- Run ---
# if __name__ == "__main__":
#     solve_kkt()
