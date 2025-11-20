import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from plot.entropy import plot_entropy_multirun
    from plot.loader import results_loader
    return plot_entropy_multirun, results_loader


@app.cell
def _(plot_entropy_multirun, results_loader):
    multirun_path = "output/multi_pattern_mhn/multirun/2025-11-13/17-01-22"
    results = results_loader(root=multirun_path)
    plot_entropy_multirun(results, memory=results[0]["weight"])
    return


@app.cell
def _(mo):
    mo.md(f"""
    # Hello World!
    This is a simple markdown cell.
    """)
    return


@app.cell
def _():
    print("Hello from a Python cell!")
    return


if __name__ == "__main__":
    app.run()
