from __future__ import annotations
import pandas as pd

# Exception 
class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to its expected schema"""
    pass


# Schema definitions 
ColumnSpec = tuple[str, str, bool, tuple[float, float] | None]

_SCHEMAS: dict[str, list[ColumnSpec]] = {

    "raw": [
        ("customerID", "string",  False, None),
        ("gender", "string",  False, None),
        ("SeniorCitizen", "numeric", False, (0, 1)),
        ("tenure", "numeric", False, (0, 200)),
        ("MonthlyCharges", "numeric", False, (0, 10_000)),
        ("TotalCharges", "any", True, None),   # string in raw file
        ("Contract", "string", False, None),
        ("PaymentMethod", "string", False, None),
        ("Churn", "string", False, None),],

    "clean": [
        ("customerID", "string", False, None),
        ("tenure", "numeric", False, (0, 200)),
        ("MonthlyCharges", "numeric", False, (0, 10_000)),
        ("TotalCharges", "numeric", False, (0, 1_000_000)),
        ("Contract", "string",  False, None),
        ("PaymentMethod", "string", False, None),
        ("Churn", "string", False, None),
        ("Churn_bin", "numeric", False, (0, 1)),],

    "features": [
        ("customerID", "string",  False, None),
        ("tenure", "numeric", False, (0, 200)),
        ("MonthlyCharges", "numeric", False, (0, 10_000)),
        ("TotalCharges", "numeric", False, (0, 1_000_000)),
        ("Contract", "string", False, None),
        ("PaymentMethod", "string",  False, None),
        ("num_services", "numeric", False, (0, 9)),
        ("num_services_protective", "numeric", False, (0, 4)),
        ("num_services_entertainment", "numeric", False, (0, 2)),
        ("is_internet_no", "bool", False, None),
        ("is_internet_dsl", "bool", False, None),
        ("is_internet_fiber", "bool", False, None),
        ("auto_payment", "bool", False, None),
        ("is_mailed_check", "bool", False, None),
        ("is_electronic_check", "bool", False, None),
        ("long_contract", "bool", False, None),
        ("E0", "numeric", False, (0.0, 1.0)),
        ("E_eq", "numeric", False, (0.0, 1.0)),
        ("gamma", "numeric", False, (0.0, 1.0)),
        ("tau", "numeric", False, (0.0, 1_000)),
        ("resilience", "string", False, None),
        ("physical_state", "string", False, None),
        ("intrinsic_risk", "string", False, None),
        ("estimated_perturbation", "numeric", False, (-2.0, 3.0)),],

    "final": [
        ("customerID", "string",  False, None),
        ("E0", "numeric", False, (0.0, 1.0)),
        ("E_eq", "numeric", False, (0.0, 1.0)),
        ("gamma", "numeric", False, (0.0, 1.0)),
        ("tau", "numeric", False, (0.0, 1_000)),
        ("prob_churn", "numeric", False, (0.0, 1.0)),
        ("prob_churn_calibrated", "numeric", False, (0.0, 1.0)),
        ("sigma_prob", "numeric", False, (0.0, 1.0)),
        ("risk_level", "string", False, None),
        ("segment", "string", False, None),
        ("action", "string", False, None),
        ("priority", "numeric", False, (1, 4)),],}

# Valid values for categorical columns (checked separately from dtype)
_CATEGORICALS: dict[str, dict[str, list[str]]] = {
    "clean": {
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["Yes", "No"],},
    "features": {
        "Contract": ["Month-to-month", "One year", "Two year"],
        "resilience": ["High", "Medium", "Low"],
        "physical_state": ["Stable", "At risk", "Critical"],
        "intrinsic_risk": ["Yes", "No"],},
    "final": {
        "risk_level": ["HIGH", "MEDIUM", "LOW", "UNCERTAIN"],
        "segment": [
            "High Risk / Resilient",
            "High Risk / Fragile",
            "High Risk / Medium resilience",
            "Medium Risk / Resilient",
            "Medium Risk / Fragile",
            "Structural Risk",
            "Stable",
            "Uncertain",],},}


# Dtype helpers 
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_string(series: pd.Series) -> bool:
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype))


def _is_bool(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series) or (_is_numeric(series) and series.dropna().isin([0, 1]).all())


_DTYPE_CHECKS = {
    "numeric": _is_numeric,
    "string": _is_string,
    "bool": _is_bool,
    "any": lambda s: True,}


# Core validator 
def validate(df: pd.DataFrame, stage: str, min_rows: int = 1, extra_checks: bool = True,) -> None:
    """Validates a DataFrame against the schema for the given pipeline stage"""
    if stage not in _SCHEMAS:
        valid = sorted(_SCHEMAS.keys())
        raise KeyError(f"Unknown schema stage {stage!r}. Valid stages: {valid}")

    errors: list[str] = []

    # Row count 
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows; expected at least {min_rows}.")

    schema = _SCHEMAS[stage]
    cats = _CATEGORICALS.get(stage, {})
    present = set(df.columns)

    # Column presence 
    required_cols = {spec[0] for spec in schema}
    missing = sorted(required_cols - present)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Per-column checks 
    for col, dtype_family, nullable, value_range in schema:
        if col not in present:
            continue  

        series = df[col]

        # dtype
        check_fn = _DTYPE_CHECKS[dtype_family]
        if not check_fn(series):
            actual = str(series.dtype)
            errors.append(
                f"Column {col!r}: expected dtype family {dtype_family!r}, "
                f"got {actual!r}.")

        # nullability
        n_null = series.isna().sum()
        if not nullable and n_null > 0:
            errors.append(f"Column {col!r}: {n_null} null value(s) in non-nullable column.")

        if not extra_checks:
            continue

        # value range (numeric only)
        if value_range is not None and _is_numeric(series):
            lo, hi = value_range
            out_of_range = series.dropna()
            n_low = (out_of_range < lo).sum()
            n_high = (out_of_range > hi).sum()
            if n_low > 0:
                min_val = out_of_range.min()
                errors.append(
                    f"Column {col!r}: {n_low} value(s) below minimum {lo} "
                    f"(min found: {min_val:.4g}).")
            if n_high > 0:
                max_val = out_of_range.max()
                errors.append(
                    f"Column {col!r}: {n_high} value(s) above maximum {hi} "
                    f"(max found: {max_val:.4g}).")

    # Categorical value checks 
    if extra_checks:
        for col, valid_values in cats.items():
            if col not in present:
                continue
            actual_values = set(df[col].dropna().unique())
            unexpected = sorted(actual_values - set(valid_values))
            if unexpected:
                errors.append(
                    f"Column {col!r}: unexpected values {unexpected}. "
                    f"Valid: {sorted(valid_values)}.")

    # Raise if any errors 
    if errors:
        n = len(errors)
        bullet_list = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise SchemaError(
            f"Schema validation failed for stage {stage!r} "
            f"({n} error{'s' if n > 1 else ''}):\n{bullet_list}")


def validate_or_warn(df: pd.DataFrame, stage: str, logger=None, **kwargs,) -> bool:
    """Like validate() but logs a warning instead of raising"""
    try:
        validate(df, stage, **kwargs)
        return True
    except SchemaError as e:
        msg = str(e)
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
        return False


def schema_report(stage: str) -> str:
    """ Returns a human-readable description of the schema for a given stage"""
    if stage not in _SCHEMAS:
        raise KeyError(f"Unknown stage {stage!r}. Valid: {sorted(_SCHEMAS.keys())}")

    lines = [
        f"Schema: {stage!r}",
        "",
        f"  {'Column':<30} {'Type':<10} {'Nullable':<10} {'Range'}",
        "  ",]
    for col, dtype, nullable, rng in _SCHEMAS[stage]:
        rng_str = f"[{rng[0]}, {rng[1]}]" if rng else "-"
        null_str = "yes" if nullable else "no"
        lines.append(f"  {col:<30} {dtype:<10} {null_str:<10} {rng_str}")

    cats = _CATEGORICALS.get(stage, {})
    if cats:
        lines.append("")
        lines.append("  Categorical constraints:")
        for col, valid in cats.items():
            lines.append(f" {col}: {valid}")

    return "\n".join(lines)
