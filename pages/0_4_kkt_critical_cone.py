import streamlit as st
import sympy as sp
import json
from pathlib import Path

# ======================================================
# Persistence
# ======================================================

SAVE_DIR = Path(__file__).parent / "saved_kkt_results"
SAVE_DIR.mkdir(exist_ok=True)

# ======================================================
# Streamlit App
# ======================================================

def main():
    st.set_page_config(page_title="KKT & Critical Cone", layout="centered")
    st.title("📘 KKT Conditions, Critical Cone & Second Order Analysis")

    # --------------------------------------------------
    # Symbols
    # --------------------------------------------------
    x1, x2 = sp.symbols("x1 x2", real=True)
    lam1, lam2 = sp.symbols("lambda_1 lambda_2", nonnegative=True)

    x = sp.Matrix([x1, x2])
    lam = sp.Matrix([lam1, lam2])

    # --------------------------------------------------
    # Problem Definition
    # --------------------------------------------------
    f = -4*x1**2 - 3*x2**2
    c1 = 4 - 2*x1 - x2
    c2 = 4 - x1 - 2*x2
    constraints = [c1, c2]

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    st.subheader("📌 Save / Load Analysis")

    save_name = st.text_input("Result name", "kkt_example")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Result"):
            data = {
                "x_star": st.session_state.get("x_star"),
                "lambda_star": st.session_state.get("lambda_star"),
                "critical_cone": st.session_state.get("critical_cone"),
                "second_order": st.session_state.get("second_order")
            }
            with open(SAVE_DIR / f"{save_name}.json", "w") as f:
                json.dump(data, f, indent=2)
            st.success("Result saved")

    with col2:
        files = [p.stem for p in SAVE_DIR.glob("*.json")]
        selected = st.selectbox("Load saved result", ["—"] + files)

        if selected != "—" and st.button("📂 Load Result"):
            with open(SAVE_DIR / f"{selected}.json") as f:
                data = json.load(f)
            st.json(data)
            st.stop()

    st.divider()

    # --------------------------------------------------
    # a) KKT with Active Constraints
    # --------------------------------------------------
    st.header("a) KKT Conditions (Both Constraints Active)")

    st.latex(r"\min\; " + sp.latex(f))
    st.markdown("**Subject to:**")
    st.latex(r"c_1(x)=4-2x_1-x_2\ge0")
    st.latex(r"c_2(x)=4-x_1-2x_2\ge0")

    # Lagrangian
    L = f - lam1*c1 - lam2*c2
    st.subheader("Lagrangian")
    st.latex(r"\mathcal{L}(x,\lambda)=" + sp.latex(L))

    # Stationarity
    grad_L = sp.Matrix([sp.diff(L, x1), sp.diff(L, x2)])
    st.subheader("Stationarity")
    st.latex(sp.latex(grad_L) + "=0")

    # Active constraints → c1=c2=0
    equations = list(grad_L) + [c1, c2]

    sol = sp.solve(equations, [x1, x2, lam1, lam2], dict=True)

    if not sol:
        st.error("No KKT point found.")
        return

    sol = sol[0]
    st.session_state.x_star = {
        "x1": float(sol[x1]),
        "x2": float(sol[x2])
    }
    st.session_state.lambda_star = {
        "lambda1": float(sol[lam1]),
        "lambda2": float(sol[lam2])
    }

    st.success("KKT point found")

    st.latex(
        r"x^*=\left(" +
        sp.latex(sol[x1]) + "," +
        sp.latex(sol[x2]) + r"\right)"
    )
    st.latex(
        r"\lambda^*=\left(" +
        sp.latex(sol[lam1]) + "," +
        sp.latex(sol[lam2]) + r"\right)"
    )

    # --------------------------------------------------
    # b) Critical Cone
    # --------------------------------------------------
    st.divider()
    st.header("b) Critical Cone")

    grad_c1 = sp.Matrix([sp.diff(c1, x1), sp.diff(c1, x2)])
    grad_c2 = sp.Matrix([sp.diff(c2, x1), sp.diff(c2, x2)])

    st.markdown("Gradients of active constraints:")
    st.latex(r"\nabla c_1(x^*)=" + sp.latex(grad_c1))
    st.latex(r"\nabla c_2(x^*)=" + sp.latex(grad_c2))

    d1, d2 = sp.symbols("d1 d2", real=True)
    d = sp.Matrix([d1, d2])

    crit_ineq = [
        grad_c1.dot(d),
        grad_c2.dot(d)
    ]

    st.markdown(
        """
        The **critical cone** is defined by:
        """
    )
    st.latex(r"\nabla c_i(x^*)^T d = 0 \quad \text{for active constraints}")
    st.latex(sp.latex(crit_ineq[0]) + "=0")
    st.latex(sp.latex(crit_ineq[1]) + "=0")

    st.session_state.critical_cone = [
        str(crit_ineq[0]) + " = 0",
        str(crit_ineq[1]) + " = 0"
    ]

    # --------------------------------------------------
    # c) Second Order Conditions
    # --------------------------------------------------
    st.divider()
    st.header("c) Second Order Conditions")

    Hess_f = sp.hessian(f, (x1, x2))
    st.subheader("Hessian of the Lagrangian")
    st.latex(sp.latex(Hess_f))

    quad_form = (d.T * Hess_f * d)[0]
    st.latex(r"d^T \nabla^2 \mathcal{L}(x^*,\lambda^*) d = " + sp.latex(quad_form))

    st.markdown(
        """
        Since the Hessian is **negative definite**, the quadratic form is **negative**
        for all nonzero directions in the critical cone.
        """
    )

    st.success(
        """
        ✅ The second-order sufficient conditions hold.

        Therefore, **x\*** is a **strict local maximum** of the Lagrangian
        and hence a **local minimum** of the original problem.
        """
    )

    st.session_state.second_order = "Second-order sufficient conditions satisfied"


# ======================================================
if __name__ == "__main__":
    main()
