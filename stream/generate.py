import os
import random
import numpy as np
import polars as pl
from tqdm import tqdm
import yaml
from stream.config.config import load_config, TrainConfig, Config
from ollama import Client


def get_fictive(
    ref: pl.DataFrame,
    n: int = 1,
    dp: str = None,
    das: str = None,
    n_das: int = None,
    max_das: int = 5,
    los: int = None,
    max_los: int = 10,
) -> pl.DataFrame:

    ref = ref.filter(~pl.col("concept_code").is_null()).select(
        "concept_code", "concept_name", "aut_mco"
    )

    df = pl.DataFrame(
        schema={
            "visit_occurence_id": pl.Int64,
            "los": pl.Int32,
            "condition_status_source_value": pl.String,
            "concept_code": pl.String,
            "concept_name": pl.String,
        }
    )

    for _ in tqdm(range(n), desc="Generate ICD-10-coded hospital stays"):

        # Sélection d'un DP
        if dp is None:
            df_dp = (
                ref.filter(pl.col("aut_mco").is_in(0))
                .sample(1)
                .with_columns(pl.lit("DP").alias("condition_status_source_value"))
            )

        else:  # Si une liste de DP est fournie
            df_dp = (
                ref.filter(
                    (pl.col("concept_code").str.contains(dp))
                    & (pl.col("aut_mco").is_in(0))
                )
                .sample(1)
                .with_columns(pl.lit("DP").alias("condition_status_source_value"))
            )

        # Sélection d'un DR
        if df_dp.filter(
            ~pl.col("concept_code").str.contains("Z08|Z511|Z5101")
        ).is_empty():
            df_dr = (
                ref.filter(
                    (pl.col("concept_code").str.contains("C"))
                    & (pl.col("aut_mco").is_in([0, 4]))
                )
                .sample(1)
                .with_columns(pl.lit("df_").alias("condition_status_source_value"))
            )
            los = 0

        elif df_dp.filter(~pl.col("concept_code").str.contains("Z515")).is_empty():
            df_dr = (
                ref.filter(
                    (pl.col("concept_code").str.contains("C"))
                    & (pl.col("aut_mco").is_in([0, 4]))
                )
                .sample(1)
                .with_columns(pl.lit("DR").alias("condition_status_source_value"))
            )

        elif df_dp.filter(~pl.col("concept_code").str.contains("Z")).is_empty():
            df_dr = (
                ref.filter(
                    (~pl.col("concept_code").str.contains("C"))
                    & (pl.col("aut_mco").is_in([0, 4]))
                )
                .sample(1)
                .with_columns(pl.lit("DR").alias("condition_status_source_value"))
            )

        else:
            df_dr = pl.DataFrame(
                {
                    "concept_code": None,
                    "concept_name": None,
                    "aut_mco": None,
                    "condition_status_source_value": None,
                }
            )

        # compléter les règles de codage du GM + FG

        # Sélection des DAS
        if n_das is None:
            n_das_r = random.randint(1, max_das)
            df_das = (
                ref.filter(~pl.col("aut_mco").is_in([3]))
                .sample(n_das_r)
                .with_columns(pl.lit("DAS").alias("condition_status_source_value"))
            )
        else:
            df_das = (
                ref.filter(~pl.col("aut_mco").is_in([3]))
                .sample(n_das)
                .with_columns(pl.lit("DAS").alias("condition_status_source_value"))
            )

        # Concaténation du DP, DR et des DAS
        temp = pl.concat([df_dp, df_dr, df_das]).filter(
            pl.col("concept_name").is_not_null()
        )

        # Création de la durée de séjour
        if los is None:
            los_r = random.randint(0, max_los)
            temp = temp.with_columns(pl.lit(los_r).alias("los"))

        else:
            temp = temp.with_columns(pl.lit(los).alias("los"))

        # Création du visit_occurence_id
        temp = temp.with_columns(
            pl.lit(np.int64(random.randint(0, np.iinfo(np.int64).max))).alias(
                "visit_occurence_id"
            )
        )

        temp = temp.select(
            [
                "visit_occurence_id",
                "los",
                "condition_status_source_value",
                "concept_code",
                "concept_name",
            ]
        )

        df = pl.concat([df, temp])

    print(df)

    return df


def get_scenario(df: pl.DataFrame) -> pl.DataFrame:

    df = (
        df.filter(pl.col("concept_name").is_not_null())
        .with_columns(
            pl.format("""{} ({})""", "concept_name", "concept_code").alias("diag")
        )
        .group_by(["visit_occurence_id", "los", "condition_status_source_value"])
        .agg(pl.col("diag").str.join(", "))
        .pivot(
            "condition_status_source_value",
            index=["visit_occurence_id", "los"],
            values="diag",
        )
        .fill_null("Aucun")
    )

    df = df.with_columns(
        pl.format("""Diagnostics CIM-10 :\n- Diagnostic principal : {}.""", "DP").alias(
            "scenario"
        )
    )

    if "DR" in df.columns:
        df = df.with_columns(
            pl.concat_str(
                [pl.col("scenario"), pl.format("\n- Diagnostic relié : {}.", "DR")]
            ).alias("scenario")
        )

    else:
        df = df.with_columns(
            pl.concat_str(
                [pl.col("scenario"), pl.lit("\n- Diagnostic relié : Aucun.")]
            ).alias("scenario")
        )

    if "DAS" in df.columns:
        df = df.with_columns(
            pl.concat_str(
                [pl.col("scenario"), pl.format("\n- Diagnostics associés : {}.", "DAS")]
            ).alias("scenario")
        )

    else:
        df = df.with_columns(
            pl.concat_str(
                [pl.col("scenario"), pl.lit("\n- Diagnostics associés : Aucun.")]
            ).alias("scenario")
        )

    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("scenario"),
                pl.format("\n- Durée de l'hospitalisation : {} jours.", "los"),
            ]
        ).alias("scenario")
    ).select("visit_occurence_id", "scenario")

    print("Generate scenario : Done.")

    return df


def get_report(
    scenario: pl.DataFrame, prompt: dict, client, model: str
) -> pl.DataFrame:
    """ """
    df = pl.DataFrame(
        schema={
            "visit_occurence_id": pl.Int64,
            "report": pl.String,
        }
    )

    for row in scenario.iter_rows():

        temp = pl.DataFrame([row], schema=scenario.columns, orient="row")

        chat_response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt["generate"]["prompt_system_intructions"],
                },
                {
                    "role": "system",
                    "content": prompt["generate"]["prompt_exemple_specifique"],
                },
                {
                    "role": "user",
                    "content": row[1],
                },
            ],
        )

        temp = temp.select("visit_occurence_id").with_columns(
            pl.lit(chat_response["message"]["content"])
            .str.replace(r"Diagnostics CIM-10 :\n?.+\n?.+\n?.+\n?.+", "")
            .alias("report")
        )

        df = pl.concat([df, temp])

    return df


def get_standarized(
    scenario: pl.DataFrame, report: pl.DataFrame, prompt: dict
) -> pl.DataFrame:

    settings = load_config("./stream/config/stream_config.yaml", TrainConfig)

    df = (
        scenario.join(report, on="visit_occurence_id")
        .rename({"scenario": "output", "report": "input"})
        .with_columns(
            pl.lit(prompt["finetune"]["prompt_system_instruction"]).alias("instruction")
        )
        .with_columns(
            pl.format(
                """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""",
                "instruction",
                "input",
                "output",
            ).alias("text")
        )
        .select("visit_occurence_id", "instruction", "input", "output", "text")
    )

    # os.makedirs("stream/output", exist_ok=True)

    if not os.path.exists(settings.path.train_dataset):
        db = pl.DataFrame(
            schema={
                "visit_occurence_id": pl.Int64,
                "instruction": pl.String,
                "input": pl.String,
                "output": pl.String,
                "text": pl.String,
            }
        )

        db.write_csv(settings.path.train_dataset, separator=";", include_bom=True)

    db = pl.read_csv(
        settings.path.train_dataset,
        separator=";",
        encoding="utf-8",
        schema_overrides={
            "visit_occurence_id": pl.Int64,
            "instruction": pl.String,
            "input": pl.String,
            "output": pl.String,
            "text": pl.String,
        },
    )
    df = pl.concat([db, df])

    df.write_csv(settings.path.train_dataset, separator=";", include_bom=True)

    return df


def generate_MR(fictive: bool, n: int):

    # fictive = input("Do you want to generate fictive hospital stay ? [y/n]")
    # n = input("How many do you want ? [int]")

    settings = load_config("./stream/config/stream_config.yaml", TrainConfig)

    if fictive:  # Get fictive hospital stay
        # Import referential
        PATH_ICD = "./stream/referentials/"
        df = pl.read_csv(
            PATH_ICD + "cim_10_atih_2019.tsv",
            separator="\t",
            new_columns=[
                "concept_code",
                "aut_mco",
                "pos",
                "aut_ssr",
                "lib_court",
                "concept_name",
            ],
            encoding="utf-8",
        )
        df = df.with_columns(pl.col("concept_code").str.replace(r"\.|\s+", ""))

        sample = get_fictive(df, int(n))

    elif not fictive:  # Get real hospital stay
        df = pl.read_csv(settings.path.pmsi_dataset, separator=";", encoding="utf-8").unique()
        _temp = df.select("visit_occurence_id").unique().sample(int(n), shuffle=True)

        sample = df.join(_temp, on="visit_occurence_id")

    else:
        print("╰┈➤Please respond by [y/n].")
        sys.exit(1)

    # Import prompt
    with open("stream/prompt.yaml", "r") as file:
        prompt = yaml.safe_load(file)

    # Set Model
    setting = load_config("./stream/config/config.yaml", Config)
    model = setting.ollama.model
    client = Client(host=setting.ollama.host)

    # Get scenario
    scenario = get_scenario(sample)

    for row in tqdm(
        scenario.iter_rows(), total=scenario.shape[0], desc="Generate Hospital report"
    ):

        # Get hospital report
        report = get_report(
            pl.DataFrame([row], schema=scenario.columns, orient="row"),
            prompt,
            client,
            model,
        )

        # Prepare data for finetunning
        dataset = get_standarized(
            pl.DataFrame([row], schema=scenario.columns, orient="row"), report, prompt
        )
    print(f"╰┈➤ Job save in {settings.path.train_dataset}'")
