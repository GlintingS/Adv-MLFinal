from pathlib import Path
import runpy

# Standard Streamlit Cloud entrypoint.
runpy.run_path(
    str(Path(__file__).resolve().parent / "streamlit_04NeuralNetworks_app.py"),
    run_name="__main__",
)
