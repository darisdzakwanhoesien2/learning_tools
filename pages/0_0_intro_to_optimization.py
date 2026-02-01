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

import streamlit as st
import sympy as sp

# ======================================================
# Helpers
# ======================================================

def parse_expression(expr_str, symbols_dict):
    """Safely parse string into a SymPy expression"""
    return sp.sympify(expr_str, locals=symbols_dict)


def parse_constraints(constraints_str, symbols_dict):
    """
    Parse constraints of form:
    x1 + x2 - 5 <= 0
    x1 - 1 = 0
    """
    constraints = []
    types = []  # 'ineq' or 'eq'

    for line in constraints_str.split(","):
        line = line.strip()

        if "<=" in line:
            lhs, rhs = line.split("<=")
            g = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
            constraints.append(g)
            types.append("ineq")

        elif "=" in line:
            lhs, rhs = line.split("=")
            h = parse_expression(lhs, symbols_dict) - parse_expression(rhs, symbols_dict)
            constraints.append(h)
            types.append("eq")

        else:
            raise ValueError(f"Invalid constraint format: {line}")

    return constraints, types


def is_feasible(sol, constraints, types, tol=1e-6):
    """Check primal feasibility"""
    for g, t in zip(constraints, types):
        val = g.subs(sol)
        val = float(val)

        if t == "ineq" and val > tol:
            return False
        if t == "eq" and abs(val) > tol:
            return False

    return True


# ======================================================
# Streamlit App
# ======================================================

def main():
    st.set_page_config(page_title="KKT Solver", layout="centered")
    st.title("KKT Optimization Solver (Educational)")

    st.markdown(
        """
        This tool **symbolically constructs and solves KKT conditions**  
        for small constrained optimization problems.

        - Supports ≤ and = constraints
        - Solves symbolically using SymPy
        - Intended for **learning and inspection**, not large-scale optimization
        """
    )

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------
    problem_type = st.radio("Problem Type", ["Minimize", "Maximize"])

    n_vars = st.slider("Number of variables", 1, 3, 2)
    var_names = [f"x{i+1}" for i in range(n_vars)]
    x = sp.symbols(var_names)
    symbols_dict = dict(zip(var_names, x))

    obj_str = st.text_input(
        "Objective function",
        value="x1**2 + x2**2"
    )

    constraints_str = st.text_area(
        "Constraints (comma-separated)",
        value="x1 + x2 - 5 <= 0, -x1 <= 0"
    )

    # --------------------------------------------------
    # Solve Button
    # --------------------------------------------------
    if st.button("Solve KKT"):
        try:
            # Parse objective
            f = parse_expression(obj_str, symbols_dict)
            if problem_type == "Maximize":
                f = -f

            # Parse constraints
            constraints, types = parse_constraints(constraints_str, symbols_dict)

            # Lagrange multipliers
            lambdas = sp.symbols(f"lambda0:{len(constraints)}")

            # Build Lagrangian
            L = f
            for lam, g in zip(lambdas, constraints):
                L += lam * g

            # --------------------------------------------------
            # Display Lagrangian
            # --------------------------------------------------
            st.subheader("Lagrangian")
            st.latex(sp.latex(L))

            # --------------------------------------------------
            # KKT Conditions
            # --------------------------------------------------
            stationarity = [sp.diff(L, xi) for xi in x]
            complementary = [
                lam * g if t == "ineq" else g
                for lam, g, t in zip(lambdas, constraints, types)
            ]

            equations = stationarity + complementary

            st.subheader("KKT Conditions")

            st.markdown("**Stationarity**")
            for eq in stationarity:
                st.latex(sp.latex(eq))

            st.markdown("**Complementary Slackness / Equality Constraints**")
            for eq in complementary:
                st.latex(sp.latex(eq))

            # --------------------------------------------------
            # Solve System
            # --------------------------------------------------
            sol = sp.solve(
                equations,
                list(x) + list(lambdas),
                dict=True
            )

            if not sol:
                st.warning("No symbolic KKT solutions found.")
                return

            # --------------------------------------------------
            # Display Feasible Solutions
            # --------------------------------------------------
            st.subheader("Feasible KKT Solutions")

            found = False
            for i, s in enumerate(sol, 1):
                if not is_feasible(s, constraints, types):
                    continue

                found = True

                primal = {
                    str(xi): float(s[xi])
                    for xi in x
                    if xi in s
                }

                multipliers = {
                    str(lam): float(s[lam])
                    for lam in lambdas
                    if lam in s
                }

                st.markdown(f"### Solution {i}")
                st.markdown("**Primal Variables**")
                st.json(primal)

                st.markdown("**Lagrange Multipliers**")
                st.json(multipliers)

            if not found:
                st.warning("KKT candidates found, but none are feasible.")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()


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
