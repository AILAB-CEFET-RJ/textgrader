from datasets import load_dataset


def get_datasets(data_dir, suffix):
    try:
        d = load_dataset(
            "parquet",
            data_files=f"{data_dir}/train_{suffix}.parquet",
        )
        d_test = load_dataset(
            "parquet",
            data_files=f"{data_dir}/test_{suffix}.parquet",
        )
        d_eval = load_dataset(
            "parquet",
            data_files=f"{data_dir}/eval_{suffix}.parquet",

        )
    except Exception as e:
        print(f"Não foi possível usar o parquet >> {e}")
        d = load_dataset(
            "csv",
            data_files=f"{data_dir}/train_{suffix}.csv",
        )
        d_test = load_dataset(
            "csv",
            data_files=f"{data_dir}/test_{suffix}.csv",
        )
        d_eval = load_dataset(
            "csv",
            data_files=f"{data_dir}/eval_{suffix}.csv",
        )

    return d, d_test, d_eval
