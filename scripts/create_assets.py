from pathlib import Path
from sweep_results import sweep_dataframe, load_sweep_runs


if __name__ == "__main__":

    df = sweep_dataframe(sweep_dir=Path("outputs/kalman_ml_sweeps/linear_sweep"))
    print(df[[
        "name", 
        "test.ade_center_px",
        "test.fde_center_px",
        "test.miou",
        "test.kalman_ade_center_px",
        "test.kalman_fde_center_px",
        "test.kalman_miou",
        ]])

    # runs = load_sweep_runs(sweep_dir=Path("outputs/kalman_ml_sweeps/example"))
    # print(runs[0].config)
    # print(runs[0].test_results)