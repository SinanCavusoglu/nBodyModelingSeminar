from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/forbes_billionaires_JOINED_connections_industries_v2.csv")

OUTPUT_MINIMAL_PATH = Path("data/forbes_billionaires_simulation_minimal_v3.csv")
OUTPUT_RICH_PATH = Path("data/forbes_billionaires_simulation_rich_v3.csv")
OUTPUT_EDGES_PATH = Path("data/forbes_billionaires_edges.csv")

# This scale is applied once during preprocessing.
# In config.py, INITIAL_VELOCITY_SCALE should usually stay 1.0.
CONNECTION_VELOCITY_SCALE = 0.05

# Used only for billionaires without valid connections.
# This is not the main velocity model.
FALLBACK_VELOCITY_SCALE = 0.005

RANDOM_SEED = 42


def parse_connections(value) -> list[int]:
    if pd.isna(value):
        return []

    tokens = str(value).split("/")
    ids = []

    for token in tokens:
        token = token.strip()

        if token == "":
            continue

        try:
            ids.append(int(float(token)))
        except ValueError:
            continue

    return ids


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH}\n"
            "Put forbes_billionaires_JOINED_connections_industries_v2.csv "
            "into the data folder."
        )

    print("Loading joined Forbes dataset...")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "ID",
        "Name",
        "NetWorth",
        "x",
        "y",
        "z",
        "connections",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["ID"] = safe_numeric(df["ID"]).astype("Int64")
    df["NetWorth"] = safe_numeric(df["NetWorth"])
    df["x"] = safe_numeric(df["x"])
    df["y"] = safe_numeric(df["y"])
    df["z"] = safe_numeric(df["z"])

    df["is_valid"] = (
        df["ID"].notna()
        & df["Name"].notna()
        & df["NetWorth"].notna()
        & (df["NetWorth"] > 0)
        & df["x"].notna()
        & df["y"].notna()
        & df["z"].notna()
    )

    valid_df = df[df["is_valid"]].copy()

    valid_ids = set(valid_df["ID"].astype(int).tolist())

    print(f"Rows: {len(df)}")
    print(f"Valid simulation rows: {len(valid_df)}")
    print(f"Invalid rows removed from minimal simulation file: {len(df) - len(valid_df)}")

    id_to_position = {
        int(row.ID): np.array([row.x, row.y, row.z], dtype=float)
        for row in valid_df.itertuples(index=False)
    }

    id_to_mass = {
        int(row.ID): float(row.NetWorth) / 1e9
        for row in valid_df.itertuples(index=False)
    }

    id_to_name = {
        int(row.ID): str(row.Name)
        for row in valid_df.itertuples(index=False)
    }

    rng = np.random.default_rng(RANDOM_SEED)

    clean_connections_col = []
    connection_count_col = []

    vx_col = []
    vy_col = []
    vz_col = []
    velocity_strength_col = []

    edge_rows = []

    raw_connection_tokens = 0
    clean_connection_tokens = 0
    fallback_rows = 0

    for row in valid_df.itertuples(index=False):
        current_id = int(row.ID)
        current_position = id_to_position[current_id]

        raw_connections = parse_connections(getattr(row, "connections"))
        raw_connection_tokens += len(raw_connections)

        clean_connections = []
        seen = set()

        for connected_id in raw_connections:
            if connected_id == current_id:
                continue

            if connected_id not in valid_ids:
                continue

            if connected_id in seen:
                continue

            seen.add(connected_id)
            clean_connections.append(connected_id)

        weighted_directions = []
        weights = []

        for connected_id in clean_connections:
            connected_position = id_to_position[connected_id]
            connected_mass = id_to_mass[connected_id]

            direction = connected_position - current_position
            distance = np.linalg.norm(direction)

            if distance == 0:
                continue

            unit_direction = direction / distance

            weighted_directions.append(unit_direction * connected_mass)
            weights.append(connected_mass)

            edge_rows.append(
                {
                    "source_id": current_id,
                    "target_id": connected_id,
                    "source_name": id_to_name.get(current_id, ""),
                    "target_name": id_to_name.get(connected_id, ""),
                    "source_mass": id_to_mass.get(current_id, 0.0),
                    "target_mass": id_to_mass.get(connected_id, 0.0),
                }
            )

        clean_connection_tokens += len(clean_connections)

        if weighted_directions and sum(weights) > 0:
            velocity = np.sum(weighted_directions, axis=0) / sum(weights)
            velocity = velocity * CONNECTION_VELOCITY_SCALE

        else:
            fallback_rows += 1

            # Fallback is used only during preprocessing for isolated particles.
            # Main simulation does not use INWARD_BIAS anymore.
            inward_direction = normalize_vector(-current_position)

            random_direction = rng.normal(size=3)
            random_direction = normalize_vector(random_direction)

            velocity = 0.7 * inward_direction + 0.3 * random_direction
            velocity = normalize_vector(velocity)
            velocity = velocity * FALLBACK_VELOCITY_SCALE

            clean_connections = []

        clean_connections_col.append("/".join(str(x) for x in clean_connections))
        connection_count_col.append(len(clean_connections))

        vx_col.append(float(velocity[0]))
        vy_col.append(float(velocity[1]))
        vz_col.append(float(velocity[2]))
        velocity_strength_col.append(float(np.linalg.norm(velocity)))

    valid_df["connections"] = clean_connections_col
    valid_df["connection_count"] = connection_count_col

    valid_df["vx"] = vx_col
    valid_df["vy"] = vy_col
    valid_df["vz"] = vz_col
    valid_df["velocity_strength"] = velocity_strength_col

    valid_df["NetWorth_Billions"] = valid_df["NetWorth"] / 1e9

    minimal_columns = [
        "ID",
        "Name",
        "NetWorth",
        "NetWorth_Billions",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "velocity_strength",
        "industries",
        "connections",
        "connection_count",
    ]

    existing_minimal_columns = [
        column for column in minimal_columns if column in valid_df.columns
    ]

    minimal_df = valid_df[existing_minimal_columns].copy()
    rich_df = valid_df.copy()

    edges_df = pd.DataFrame(edge_rows)

    if not edges_df.empty:
        edges_df = edges_df.drop_duplicates(
            subset=["source_id", "target_id"]
        )

    OUTPUT_MINIMAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    minimal_df.to_csv(OUTPUT_MINIMAL_PATH, index=False)
    rich_df.to_csv(OUTPUT_RICH_PATH, index=False)
    edges_df.to_csv(OUTPUT_EDGES_PATH, index=False)

    print("Velocity generation complete.")
    print(f"Raw connection tokens: {raw_connection_tokens}")
    print(f"Clean connection tokens: {clean_connection_tokens}")
    print(f"Rows using fallback velocity: {fallback_rows}")
    print(f"Saved minimal simulation file: {OUTPUT_MINIMAL_PATH}")
    print(f"Saved rich simulation file: {OUTPUT_RICH_PATH}")
    print(f"Saved edge file: {OUTPUT_EDGES_PATH}")


if __name__ == "__main__":
    main()