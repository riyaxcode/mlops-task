import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import yaml


REQUIRED_CONFIG_FIELDS = {"seed", "window", "version"}


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Config must be a YAML dictionary")

        missing = REQUIRED_CONFIG_FIELDS - set(config.keys())

        if missing:
            raise ValueError(
                f"Missing required config fields: {sorted(missing)}"
            )

        if not isinstance(config["window"], int):
            raise ValueError("window must be an integer")

        if config["window"] <= 0:
            raise ValueError("window must be greater than 0")

        np.random.seed(config["seed"])

        return config

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML format: {e}"
        )


def load_dataset(input_path):
    try:
        with open(input_path, "r", encoding="utf-8") as f:

            for i in range(3):
                print(repr(f.readline()))

        df = pd.read_csv(input_path)

        if len(df.columns) == 1:
            split_cols = df.columns[0].split(",")

            df = pd.read_csv(
                input_path,
                header=None,
                names=split_cols
            )

            df = df.iloc[1:].reset_index(drop=True)

        df.columns = df.columns.str.strip().str.lower()

        print("Columns found:", df.columns.tolist())

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")

    except Exception as e:
        raise ValueError(
            f"Invalid CSV format: {e}"
        )

    if df.empty:
        raise ValueError(
            "Dataset contains no rows"
        )

    if "close" not in df.columns:
        raise ValueError(
            "Required column 'close' not found"
        )

    return df


def compute_signals(df, window):
    df = df.copy()

    logging.info("Computing rolling mean")

    df["rolling_mean"] = (
        df["close"]
        .rolling(window=window, min_periods=window)
        .mean()
    )

    logging.info("Generating signals")

    df["signal"] = (
        df["close"] > df["rolling_mean"]
    ).fillna(False).astype(int)

    return df


def generate_metrics(df, config, latency_ms):
    return {
        "version": config["version"],
        "rows_processed": int(len(df)),
        "metric": "signal_rate",
        "value": round(float(df["signal"].mean()), 4),
        "latency_ms": int(latency_ms),
        "seed": config["seed"],
        "status": "success"
    }


def generate_error_metrics(version, error_message):
    return {
        "version": version,
        "status": "error",
        "error_message": error_message
    }


def write_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--log-file",
        required=True
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    config = {"version": "v1"}

    try:
        setup_logging(args.log_file)

        logging.info("Job started")

        config = load_config(args.config)

        logging.info(
            f"Config loaded and validated | "
            f"seed={config['seed']} "
            f"window={config['window']} "
            f"version={config['version']}"
        )

        df = load_dataset(args.input)

        logging.info(
            f"Rows loaded: {len(df)}"
        )

        df = compute_signals(
            df,
            config["window"]
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = generate_metrics(
            df,
            config,
            latency_ms
        )

        write_metrics(
            metrics,
            args.output
        )

        logging.info(
            f"Metrics summary | "
            f"rows_processed={metrics['rows_processed']} "
            f"signal_rate={metrics['value']} "
            f"latency_ms={metrics['latency_ms']}"
        )

        logging.info(
            "Job completed successfully"
        )

        print(
            json.dumps(
                metrics,
                indent=2
            )
        )

        sys.exit(0)

    except Exception as e:

        logging.exception(
            f"Pipeline failed: {e}"
        )

        error_metrics = generate_error_metrics(
            config.get("version", "v1"),
            str(e)
        )

        try:
            write_metrics(
                error_metrics,
                args.output
            )
        except Exception:
            pass

        print(
            json.dumps(
                error_metrics,
                indent=2
            )
        )

        sys.exit(1)


if __name__ == "__main__":
    main()