from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_name: str = "Watchtower ML Detection API"
    api_version: str = "1.0.0"
    model1_path: str = str(Path("src") / "models" / "random_forest.pkl")
    model2_path: str = str(Path("src") / "models" / "xgboost_model.pkl")
    model3_path: str = str(Path("src") / "models" / "isolation_forest.pkl")
    pipeline_path: str = str(Path("src") / "models" / "preprocessing_pipeline.pkl")
    default_test_csv_path: str = str(Path("src") / "data" / "ddos_unlabeled.csv")
    expected_feature_count: int = 70


settings = Settings()
