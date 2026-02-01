import streamlit as st
from src.functions import store_function, example_function
from src.visualization import plot_3d_surface, plot_contour

# --- UI Setup ---
st.title("Polynomial Function Visualizer")
st.markdown("""
Enter coefficients for your polynomial function (e.g., `2*x1^3 -6*x1*x2 +3*x2^2`).
""")

# User input: Define the function
coeffs_input = st.text_area(
    "Function Coefficients",
    value="2*x1**3 - 6*x1*x2 + 3*x2**2",  # Default example
    height=50,
)

# Parse and store the function dynamically
try:
    exec(f"def user_function(x1, x2): return {coeffs_input.replace('x1', 'X').replace('x2', 'Y')}", globals())
    st.success("Function stored successfully!")
except Exception as e:
    st.error(f"Invalid input: {e}")

# --- Plotting ---
st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig_3d = plot_3d_surface(user_function)
    st.pyplot(fig_3d)

with col2:
    fig_contour = plot_contour(user_function)
    st.pyplot(fig_contour)

# --- Example Function ---
st.subheader("Example: Default Function")
col3, col4 = st.columns(2)

with col3:
    fig_3d_example = plot_3d_surface(example_function)
    st.pyplot(fig_3d_example)

with col4:
    fig_contour_example = plot_contour(example_function)
    st.pyplot(fig_contour_example)


# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st

# # Define the function inside a local scope for execution
# def f(x1, x2):
#     return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2

# @st.cache_data  # Cache to avoid recomputing if inputs don't change
# def generate_plot():
#     # Generate x1 and x2 values
#     x1 = np.linspace(-5, 5, 100)
#     x2 = np.linspace(-5, 5, 100)
#     x1, x2 = np.meshgrid(x1, x2)

#     # Calculate f(x1, x2)
#     z = f(x1, x2)

#     # Create the plot
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the surface
#     surf = ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='k', alpha=0.8)

#     # Add labels and title
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('f(x1, x2)')
#     ax.set_title('3D Surface Plot of f(x1, x2)')

#     # Add color bar
#     fig.colorbar(surf, shrink=0.5, aspect=10)

#     return fig

# # Main app
# st.title("Function Visualizer")
# st.write("Plot of the function $f(x_1, x_2) = 2x_1^3 - 6x_1x_2 + 3x_2^2$")

# # Generate and display the plot
# plot_figure = generate_plot()
# st.pyplot(plot_figure)

# import streamlit as st
# import os
# from pathlib import Path

# # --- Constants ---
# CODE_BLOCKS_DIR = "code_blocks"
# SCRIPTS = [
#     {"name": "3D Surface Plot", "file": "code_1.py"},
#     {"name": "2D Heatmap", "file": "code_2.py"},
#     {"name": "Contour Plot", "file": "code_3.py"}
# ]

# # --- Initialize the app ---
# st.title("Function Executor App")
# st.markdown("Select a script to run and visualize the function.")

# # --- Sidebar: File selection ---
# with st.sidebar:
#     selected_script = st.selectbox(
#         "Choose a script",
#         [s["name"] for s in SCRIPTS]
#     )
#     selected_file = next(s["file"] for s in SCRIPTS if s["name"] == selected_script)

# # --- Load and execute the selected script ---
# def run_script():
#     script_path = os.path.join(CODE_BLOCKS_DIR, selected_file)
#     with open(script_path, "r") as f:
#         code = f.read()

#     # Execute in a subprocess to avoid global state issues
#     import subprocess
#     cmd = [
#         "python", "-c",
#         f"import sys; sys.stdout = open('temp_output.txt', 'w'); {code}"
#     ]
#     subprocess.run(cmd, check=True)

#     # Display the output (if any)
#     if os.path.exists("temp_output.txt"):
#         with open("temp_output.txt", "r") as f:
#             st.text_area("Output:", f.read(), height=100)

# # --- Run the selected script ---
# run_script()

# # --- Cleanup ---
# os.remove("temp_output.txt")


# import streamlit as st
# import os
# from pathlib import Path

# # --- Config ---
# CODE_DIR = "code_blocks"
# EXECUTION_HISTORY = "execution_history.txt"

# # --- Helper Functions ---
# def load_code_block(filename):
#     """Load code from a file in `code_blocks/`"""
#     full_path = Path(CODE_DIR) / filename
#     if not full_path.exists():
#         raise FileNotFoundError(f"Code block '{filename}' not found!")
#     with open(full_path, "r") as f:
#         return f.read()

# def execute_code(code):
#     """Execute Python code safely (with error handling)"""
#     try:
#         local_vars = {}
#         exec(code, globals(), local_vars)
#         return local_vars
#     except Exception as e:
#         st.error(f"Execution Error: {str(e)}")
#         return None

# # --- UI Components ---
# st.title("🔧 Code Execution Dashboard")

# # 1. Dropdown to select code block
# code_files = sorted([f for f in os.listdir(CODE_DIR) if f.endswith(".py")])
# selected_code = st.selectbox(
#     "Select a code block:",
#     code_files,
#     index=0
# )

# if selected_code:
#     # Load and display the code
#     with st.expander("Code Preview"):
#         code = load_code_block(selected_code)
#         st.code(code, language="python")

#     # Execute button
#     if st.button("⚡ Execute Code"):
#         try:
#             # Execute and capture output
#             result = execute_code(code)

#             # Save execution history (optional)
#             with open(EXECUTION_HISTORY, "a") as f:
#                 f.write(f"--- {selected_code} ---\n{code}\n\nOutput:\n{str(result)}\n\n{'='*50}\n")

#             st.success("✅ Code executed successfully!")
#             if result:
#                 st.json(result)  # Display output as JSON

#         except Exception as e:
#             st.error(f"Failed to execute: {e}")

# # --- Optional: Show execution history ---
# if os.path.exists(EXECUTION_HISTORY):
#     with st.expander("Execution History"):
#         try:
#             with open(EXECUTION_HISTORY, "r") as f:
#                 history = f.read()
#             st.text_area("History", history, height=300)
#         except Exception as e:
#             st.error(f"Failed to load history: {e}")

# # --- Initialize code blocks (if empty) ---
# if not os.path.exists(CODE_DIR):
#     os.makedirs(CODE_DIR)

# # Example: Add your function definition
# with open(Path(CODE_DIR) / "f(x1,x2).py", "w") as f:
#     f.write("""
# def f(x1, x2):
#     return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2
# """)

# # Example: Add plot code (if needed)
# with open(Path(CODE_DIR) / "plot_code_old.py", "w") as f:
#     f.write("""
# import numpy as np
# import matplotlib.pyplot as plt

# x1 = np.linspace(-5, 5, 100)
# x2 = np.linspace(-5, 5, 100)
# x1, x2 = np.meshgrid(x1, x2)

# z = f(x1, x2)  # Assuming 'f' is defined elsewhere

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='k', alpha=0.8)
# plt.show()
# """)


# import streamlit as st
# from src.executor import CodeExecutor

# def main():
#     st.title("Code Execution System")
#     st.markdown("Run Python code from files or execute inline code")

#     # Initialize executor
#     executor = CodeExecutor()

#     # Option 1: Run a specific page file
#     if st.button("Run pages/1_intro_to_optimization.py"):
#         with st.spinner("Running page..."):
#             try:
#                 output_text, plot_path = executor.run_page("pages/1_intro_to_optimization.py")

#                 if output_text:
#                     st.success("Page executed successfully!")
#                     st.text_area("Output:", value=output_text, height=200)

#                 if plot_path and os.path.exists(plot_path):
#                     with open(plot_path, 'rb') as f:
#                         img = Image(f.read())
#                     st.image(img, caption="Generated Plot", use_column_width=True)
#             except Exception as e:
#                 st.error(f"Error running page: {str(e)}")

#     # Option 2: Execute inline code
#     elif st.button("Execute Inline Code"):
#         with st.expander("Enter your Python code here:"):
#             code_snippet = st.text_area(
#                 "Paste your Python code here",
#                 height=300,
#                 value="""
# import numpy as np
# import matplotlib.pyplot as plt

# def f(x1, x2):
#     return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2

# x1 = np.linspace(-5, 5, 100)
# x2 = np.linspace(-5, 5, 100)
# x1, x2 = np.meshgrid(x1, x2)

# z = f(x1, x2)

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='k', alpha=0.8)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('f(x1, x2)')
# ax.set_title('3D Surface Plot of f(x1, x2)')
# plt.show()
# """
#             )

#         if st.button("Execute"):
#             with st.spinner("Executing code..."):
#                 try:
#                     output_text, plot_path = executor.execute_code(code_snippet)

#                     if output_text:
#                         st.success("Code executed successfully!")
#                         st.text_area("Output:", value=output_text, height=200)

#                     if plot_path and os.path.exists(plot_path):
#                         with open(plot_path, 'rb') as f:
#                             img = Image(f.read())
#                         st.image(img, caption="Generated Plot", use_column_width=True)
#                 except Exception as e:
#                     st.error(f"Error executing code: {str(e)}")

# if __name__ == "__main__":
#     main()



# import streamlit as st
# from src.executor import CodeExecutor

# # Initialize executor
# executor = CodeExecutor()

# st.title("Code Executor")
# code_snippet = st.text_area("Paste your Python code here:")

# if st.button("Execute"):
#     with st.spinner("Running..."):
#         result, output_path = executor.execute(code_snippet)
#         st.success(f"Executed! Output saved to: {output_path}")
#         st.image(output_path)  # Show plot if applicable
