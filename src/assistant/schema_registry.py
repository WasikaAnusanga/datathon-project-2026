import os
import yaml
from typing import Dict, Any, List, Set, Optional, Tuple


class SchemaRegistry:
    """
    Semantic Schema Registry for NYC Urban Flow Taxi Dataset.
    Provides physical column lookup, synonym resolution, alias distinction,
    and dataset-time boundary policy tracking.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default path relative to workspace root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config", "mobility_schema.yaml")

        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Schema configuration file not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.dataset_meta: Dict[str, Any] = data.get("dataset_metadata", {})
        self.date_policy: Dict[str, Any] = data.get("date_policy", {})
        self.raw_columns: Dict[str, Dict[str, Any]] = data.get("columns", {})

        self.physical_columns: Set[str] = set()
        self.column_lower_to_canonical: Dict[str, str] = {}
        self.synonym_to_canonical: Dict[str, str] = {}
        self.column_types: Dict[str, str] = {}

        for col_name, info in self.raw_columns.items():
            self.physical_columns.add(col_name)
            self.column_lower_to_canonical[col_name.lower()] = col_name
            self.column_types[col_name] = info.get("type", "VARCHAR")

            # Map synonyms
            for syn in info.get("synonyms", []):
                self.synonym_to_canonical[syn.lower()] = col_name

    def is_physical_column(self, name: str) -> bool:
        """Check if name is an exact or case-insensitive physical dataset column."""
        return name.lower() in self.column_lower_to_canonical

    def resolve_column(self, name: str) -> Optional[str]:
        """
        Resolves a column name or synonym to its canonical physical column name.
        Returns None if not matched.
        """
        low = name.lower()
        if low in self.column_lower_to_canonical:
            return self.column_lower_to_canonical[low]
        if low in self.synonym_to_canonical:
            return self.synonym_to_canonical[low]
        return None

    def validate_column_or_alias(self, column_name: str, query_aliases: Set[str]) -> bool:
        """
        Refinement #3: Validates whether a column reference is EITHER:
        1. A physical dataset column (or known synonym), OR
        2. A valid query expression alias generated in the SELECT clause (e.g. trip_count).
        """
        if self.is_physical_column(column_name):
            return True
        if self.resolve_column(column_name) is not None:
            return True
        # Check case-insensitive against query aliases
        aliases_lower = {a.lower() for a in query_aliases}
        return column_name.lower() in aliases_lower

    def get_column_type(self, column_name: str) -> Optional[str]:
        """Returns the stored data type of a physical column."""
        canonical = self.resolve_column(column_name)
        if canonical:
            return self.column_types.get(canonical)
        return None

    def get_dataset_temporal_bounds(self) -> Dict[str, Any]:
        """
        Refinement #6: Returns dataset-time bounds:
        min_date, max_date, available years/months for relative time interpretation.
        """
        return {
            "dataset_min_date": self.dataset_meta.get("dataset_min_date", "2025-04-01"),
            "dataset_max_date": self.dataset_meta.get("dataset_max_date", "2026-03-31"),
            "available_years": self.dataset_meta.get("available_years", [2025, 2026]),
            "available_months": self.dataset_meta.get("available_months", list(range(1, 13))),
            "default_anchor_date": self.date_policy.get("default_anchor_date", "2026-03-31"),
        }

    def get_table_name(self) -> str:
        """Return canonical DuckDB relation name."""
        return self.dataset_meta.get("table_name", "taxi_trips")

    def get_data_path(self) -> str:
        """Return relative path to full parquet dataset."""
        return self.dataset_meta.get("data_path", "data/processed/Urban_Flow_Analytics_Taxi_Clean_Enriched_12Month.parquet")

    def get_sample_fixture_path(self) -> str:
        """Return relative path to test sample parquet fixture."""
        return self.dataset_meta.get("sample_fixture_path", "tests/fixtures/taxi_sample.parquet")
