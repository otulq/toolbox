"""
profile_df.py   column-profiling utilities

• ColumnProfile hierarchy (FloatProfile, IntProfile, BoolProfile,
  DateProfile, DateStrProfile, DatetimeProfile, DatetimeStrProfile, StringProfile)
• profile_column()    build a profile from an existing DataFrame column
• DataFrameProfile    convenience wrapper to profile an entire DataFrame
"""

# region: imports and helpers

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta, datetime
from typing import Optional, Union, Dict, Literal, List, Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype, is_integer_dtype, is_float_dtype,
    is_bool_dtype, is_datetime64_any_dtype, is_string_dtype,
)


def _convert_numpy_to_python(value: Any) -> Any:
    """Convert numpy types to native Python types for OneValueProfile."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    elif isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    else:
        return value


def _format_float_for_script(value: float, other_value: Optional[float] = None) -> str:
    """
    Format a float for script generation with intelligent rounding.
    
    Args:
        value: The float to format
        other_value: Another value to ensure we don't round to the same thing
        
    Returns:
        Clean string representation of the float
    """
    if pd.isna(value) or value is None:
        return str(value)
    
    # Try different precision levels
    for precision in [2, 3, 4, 5, 6]:
        rounded = round(value, precision)
        if other_value is None:
            return str(rounded)
        
        # Make sure rounding doesn't make values equal
        other_rounded = round(other_value, precision)
        if rounded != other_rounded:
            return str(rounded)
    
    # If all else fails, use the original value
    return str(value)


# ────────────────────────────────────────────────────────────────────────────
# mix-in for “maybe missing” logic
# ────────────────────────────────────────────────────────────────────────────
class _SampleMixin:
    def _maybe_missing(self) -> bool:
        # FIXED: self.missing_prob was named missing_probability in code – unify
        return random.random() < self.missing_prob

# endregion: imports and helpers

# region: ColumnProfile and subclasses

# ────────────────────────────────────────────────────────────────────────────
# base class
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ColumnProfile(_SampleMixin):
    missing_prob: float

    # repr / dict helpers
    def __repr__(self) -> str:
        return str(self)

    def has_nulls(self) -> bool:
        return self.missing_prob > 0.0

    def to_dict(self) -> Dict[str, Union[float, int, str, List[str]]]:
        return asdict(self)

    # sampling helpers
    def sample(self) -> Optional[Union[float, int, str, date, datetime]]:
        """Subclasses implement."""
        raise NotImplementedError

    def samples(self, n: int) -> List[Optional[Union[float, int, str, date, datetime]]]:
        return [self.sample() for _ in range(n)]

    def samples_script(self, n: int, indent: str = '    ', single_line: bool = True, 
                      use_variable: bool = False, variable_name: str = "n_values") -> str:
        # Choose the range expression based on use_variable parameter
        range_expr = variable_name if use_variable else str(n)
        
        if not self.has_nulls():
            if single_line:
                return f"[{self.sample_script()} for _ in range({range_expr})]"
            else:
                return f"[\n{indent}{self.sample_script()}\n{indent}for _ in range({range_expr})\n]"
        
        # Format the probability cleanly
        prob_str = _format_float_for_script(1 - self.missing_prob)
        
        if single_line:
            return (
                f'[{self.sample_script(ignore_missing=True)} '
                f'if random.random() < {prob_str} else None '
                f'for _ in range({range_expr})]'
            )
        else:
            return (
                f'[\n{indent}{self.sample_script(ignore_missing=True)} '
                f'if random.random() < {prob_str} else None'
                f'\n{indent}for _ in range({range_expr})\n]'
            )

    def _contains_single_value(self, value: Union[float, int, str, date, datetime]) -> bool:
        """Subclasses implement."""
        raise NotImplementedError

    def __contains__(self, value: Union[float, int, str, date, datetime]) -> bool:
        if isinstance(value, (list, set, tuple, pd.Series)):
            return all(self.__contains__(v) for v in value)
        if value is None or pd.isna(value):
            return self.has_nulls()
        return self._contains_single_value(value)

    contains = __contains__

    @classmethod
    def type_check(cls, value: Union[float, int, str, date, datetime, pd.Series]) -> bool:
        """Check if the value is of the correct type for this profile."""
        return cls._type_check(value)

# ────────────────────────────────────────────────────────────────────────────
# special profiles
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NullProfile(ColumnProfile):
    type: Literal['null'] = 'null'
    missing_prob: float = 1.0

    def __str__(self) -> str:
        return f"NullProfile()"

    def sample(self) -> None:
        return None

    def sample_script(self, ignore_missing: bool = False) -> str:
        return "None"
    
    def samples_script(self, n: int, indent: str = '    ', single_line: bool = True,
                      use_variable: bool = False, variable_name: str = "n_values") -> str:
        # Override to always return simple None list since we're always null
        # Choose the range expression based on use_variable parameter
        range_expr = variable_name if use_variable else str(n)
        
        if single_line:
            return f"[None for _ in range({range_expr})]"
        else:
            return f"[\n{indent}None\n{indent}for _ in range({range_expr})\n]"

    @classmethod
    def _type_check(cls, value: Union[None, pd.Series]) -> bool:
        return value is None

    def _contains_single_value(self, value: None) -> bool:
        return value is None

    def has_nulls(self) -> bool:
        return True

@dataclass(frozen=True)
class OneValueProfile(ColumnProfile):
    missing_prob: float
    value: Union[float, int, str, bool, date, datetime]
    type: Literal['single_value'] = 'single_value'
    
    def __str__(self) -> str:
        return f"OneValueProfile(value={self.value}, missing_prob={self.missing_prob})"
    
    def sample(self) -> Optional[Union[float, int, str, bool, date, datetime]]:
        if self._maybe_missing():
            return None
        return self.value
    
    def sample_script(self, ignore_missing: bool = False) -> str:
        if ignore_missing or self.missing_prob == 0.0:
            if isinstance(self.value, str):
                return f"'{self.value}'"
            elif isinstance(self.value, (float, int, bool)):
                if isinstance(self.value, float):
                    return _format_float_for_script(self.value)
                else:
                    return f"{self.value}"
            elif isinstance(self.value, datetime):
                return f"datetime.fromisoformat('{self.value.isoformat()}')"
            elif isinstance(self.value, date):
                return f"date.fromisoformat('{self.value.isoformat()}')"
            else:
                raise ValueError(f"Invalid value type: {type(self.value)}")
        else:
            missing_prob_str = _format_float_for_script(1 - self.missing_prob)
            if isinstance(self.value, str):
                return f"'{self.value}' if random.random() < {missing_prob_str} else None"
            elif isinstance(self.value, (float, int, bool)):
                if isinstance(self.value, float):
                    value_str = _format_float_for_script(self.value)
                    return f"{value_str} if random.random() < {missing_prob_str} else None"
                else:
                    return f"{self.value} if random.random() < {missing_prob_str} else None"
            elif isinstance(self.value, datetime):
                return f"datetime.fromisoformat('{self.value.isoformat()}') if random.random() < {missing_prob_str} else None"
            elif isinstance(self.value, date):
                return f"date.fromisoformat('{self.value.isoformat()}') if random.random() < {missing_prob_str} else None"
            else:
                raise ValueError(f"Invalid value type: {type(self.value)}")
    
    @classmethod
    def _type_check(cls, value: Union[float, int, str, bool, date, datetime, pd.Series]) -> bool:
        return isinstance(value, (float, int, str, bool, datetime, date))
    
    def _contains_single_value(self, value: Union[float, int, str, bool, date, datetime]) -> bool:
        return value == self.value




# ────────────────────────────────────────────────────────────────────────────
# numeric profiles
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FloatProfile(ColumnProfile):
    min: float
    max: float
    type: Literal['float'] = 'float'

    def __str__(self) -> str:
        return (f"FloatProfile(min={self.min}, max={self.max}, "
                f"missing_prob={self.missing_prob})")

    def sample(self) -> Optional[float]:
        if self._maybe_missing() or self.min is None or self.max is None:
            return None
        if pd.isna(self.min) or pd.isna(self.max):
            return None
        return random.uniform(self.min, self.max)

    def sample_script(self, ignore_missing: bool = False) -> str:
        min_str = _format_float_for_script(self.min, self.max)
        max_str = _format_float_for_script(self.max, self.min)
        
        if not self.has_nulls() or ignore_missing:
            return f"random.uniform({min_str}, {max_str})"
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"random.uniform({min_str}, {max_str}) if random.random() < {prob_str} else None"

    # incorrect_sample kept exactly as you wrote
    def incorrect_sample(self, margin: float) -> float:
        """
        Generate a sample guaranteed *outside* the min–max range.
        Useful for testing.
        """
        if pd.isna(self.min) or pd.isna(self.max) or self.min is None or self.max is None:
            raise ValueError("Cannot generate incorrect sample with NaN or None min/max")

        # if gap to zero is small, place outside on upper end
        if abs(self.min) < margin:
            return self.max + margin
        return self.min - margin

    @classmethod
    def _type_check(cls, value: Union[float, pd.Series]) -> bool:
        # check if value is a float or a numpy float64 or a pd.Series of floats
        if isinstance(value, (float, np.float64)):
            return True
        if isinstance(value, pd.Series) and value.dtype == 'float64':
            return True
        return False

    def _contains_single_value(self, value: float) -> bool:
        if not isinstance(value, (float, np.float64)):
            return False
        return self.min <= value <= self.max


@dataclass(frozen=True)
class IntProfile(ColumnProfile):
    min: int
    max: int
    type: Literal['int'] = 'int'

    def __str__(self) -> str:
        return (f"IntProfile(min={self.min}, max={self.max}, "
                f"missing_prob={self.missing_prob})")

    def sample(self) -> Optional[int]:
        if self._maybe_missing() or self.min is None or self.max is None:
            return None
        # pd.isna on ints returns False; safe but redundant – kept for parity
        if pd.isna(self.min) or pd.isna(self.max):
            return None
        return random.randint(self.min, self.max)

    def sample_script(self, ignore_missing: bool = False) -> str:
        if not self.has_nulls() or ignore_missing:
            return f"random.randint({self.min}, {self.max})"
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"random.randint({self.min}, {self.max}) if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[int, pd.Series]) -> bool:
        # check if value is an int or a numpy int64 or a pd.Series of ints
        if isinstance(value, (int, np.int64)):
            return True
        if isinstance(value, pd.Series) and value.dtype == 'int64':
            return True
        return False

    def _contains_single_value(self, value: int) -> bool:
        if not isinstance(value, (int, np.int64)):
            return False
        return self.min <= value <= self.max


# ────────────────────────────────────────────────────────────────────────────
# boolean profile
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BoolProfile(ColumnProfile):
    true_prob: float
    type: Literal['bool'] = 'bool'

    def __str__(self) -> str:
        return (
            f"BoolProfile(true_prob={self.true_prob}, "
            f"missing_prob={self.missing_prob})"
        )

    def sample(self) -> Optional[bool]:
        if self._maybe_missing():
            return None
        # FIXED: pd.isnan → np.isnan (avoids warning on float)
        return random.random() < self.true_prob if not np.isnan(self.true_prob) else None

    def sample_script(self, ignore_missing: bool = False) -> str:
        prob_str = _format_float_for_script(self.true_prob)
        
        if not self.has_nulls() or ignore_missing:
            return f"random.random() < {prob_str}"
        
        missing_prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"random.random() < {prob_str} if random.random() < {missing_prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[bool, pd.Series]) -> bool:
        # check if value is a bool or a numpy bool_ or a pd.Series of bools
        if isinstance(value, (bool, np.bool_)):
            return True
        if isinstance(value, pd.Series):
            if value.dtype == 'bool':
                return True
            if value.dtype == 'object':
                # Check if all non-null values are strictly bool (not str)
                non_null_values = value.dropna()
                return not non_null_values.empty and all(type(val) is bool for val in non_null_values)
        return False

    def _contains_single_value(self, value: bool) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            return False
        return value == self.true_prob


# ────────────────────────────────────────────────────────────────────────────
# date profiles
# ────────────────────────────────────────────────────────────────────────────
def _rand_date(d0: date, d1: date) -> date:
    delta = (d1 - d0).days
    return d0 + timedelta(days=random.randint(0, delta))


def _rand_datetime(dt0: datetime, dt1: datetime) -> datetime:
    """Generate a random datetime between dt0 and dt1 (inclusive)."""
    delta = dt1 - dt0
    total_seconds = int(delta.total_seconds())
    random_seconds = random.randint(0, total_seconds)
    return dt0 + timedelta(seconds=random_seconds)


@dataclass(frozen=True)
class DateProfile(ColumnProfile):
    min: date
    max: date
    type: Literal['date'] = 'date'

    def __post_init__(self):
        # accept ISO strings, convert to date
        if isinstance(self.min, str):
            object.__setattr__(self, 'min', date.fromisoformat(self.min))  # type: ignore[arg-type]
        if isinstance(self.max, str):
            object.__setattr__(self, 'max', date.fromisoformat(self.max))  # type: ignore[arg-type]

    def __str__(self) -> str:
        if pd.isna(self.min) or pd.isna(self.max):
            return (f"DateProfile(min=None, max=None, "
                    f"missing_prob={self.missing_prob})")
        return (f"DateProfile(min='{self.min.isoformat()}', "
                f"max='{self.max.isoformat()}', missing_prob={self.missing_prob})")

    def sample(self) -> Optional[date]:
        if self._maybe_missing():
            return None
        if pd.isna(self.min) or pd.isna(self.max):
            return None
        return _rand_date(self.min, self.max)

    def sample_script(self, ignore_missing: bool = False) -> str:
        min_iso = self.min.isoformat()
        max_iso = self.max.isoformat()
        base_script = f"date.fromisoformat('{min_iso}') + timedelta(days=random.randint(0, (date.fromisoformat('{max_iso}') - date.fromisoformat('{min_iso}')).days))"
        if not self.has_nulls() or ignore_missing:
            return base_script
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"{base_script} if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[date, pd.Series]) -> bool:
        # check if value is a date or a pd.Series of dates
        if isinstance(value, date):
            return True
        if isinstance(value, pd.Series) and value.dtype == 'object':
            # Check if all non-null values are date objects
            non_null_values = value.dropna()
            return not non_null_values.empty and all(isinstance(val, date) for val in non_null_values)
        return False

    def _contains_single_value(self, value: date) -> bool:
        if not isinstance(value, (date, np.datetime64)):
            return False
        return self.min <= value <= self.max


@dataclass(frozen=True)
class DateStrProfile(ColumnProfile):
    min: str  # ISO-8601
    max: str
    type: Literal['date_str'] = 'date_str'

    @property
    def min_datetime(self) -> datetime:
        return datetime.fromisoformat(self.min)
    
    @property
    def max_datetime(self) -> datetime:
        return datetime.fromisoformat(self.max)

    def __post_init__(self):
        # allow date objects on construction
        if isinstance(self.min, (date, datetime)):
            object.__setattr__(self, 'min', self.min.isoformat())  # type: ignore[arg-type]
        if isinstance(self.max, (date, datetime)):
            object.__setattr__(self, 'max', self.max.isoformat())  # type: ignore[arg-type]

    def __str__(self) -> str:
        if self.min is None or self.max is None or pd.isna(self.min) or pd.isna(self.max):
            return (f"DateStrProfile(min=None, max=None, "
                    f"missing_prob={self.missing_prob})")
        return (f"DateStrProfile(min='{self.min}', max='{self.max}', "
                f"missing_prob={self.missing_prob})")

    def sample(self) -> Optional[str]:
        if self._maybe_missing() or not self.min or not self.max:
            return None
        d0 = date.fromisoformat(self.min)
        d1 = date.fromisoformat(self.max)
        sampled_date = _rand_date(d0, d1)
        return sampled_date.isoformat()

    def sample_script(self, ignore_missing: bool = False) -> str:
        base_script = f"(date.fromisoformat('{self.min}') + timedelta(days=random.randint(0, (date.fromisoformat('{self.max}') - date.fromisoformat('{self.min}')).days))).isoformat()"
        if not self.has_nulls() or ignore_missing:
            return base_script
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"{base_script} if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[str, pd.Series]) -> bool:
        # check if value is a string or a pd.Series of strings that can be parsed as dates
        if isinstance(value, str):
            try:
                date.fromisoformat(value)
                return True
            except ValueError:
                return False
        if isinstance(value, pd.Series) and value.dtype == 'object':
            # Check if all non-null values are strings that can be parsed as dates
            non_null_values = value.dropna()
            if non_null_values.empty:
                return False
            # First check if any values are already date objects - if so, this is NOT a DateStrProfile
            if any(isinstance(val, date) for val in non_null_values):
                return False
            # Check if all non-null values are strings
            if not all(isinstance(val, str) for val in non_null_values):
                return False
            # Then check if all values are strings that can be parsed as dates
            try:
                pd.to_datetime(non_null_values, format='%Y-%m-%d', errors="coerce")
                return True
            except (ValueError, TypeError):
                return False
        return False

    def _contains_single_value(self, value: str) -> bool:
        if not isinstance(value, (str, np.str_)):
            return False
        value_date = date.fromisoformat(value)
        return self.min_datetime.date() <= value_date <= self.max_datetime.date()


@dataclass(frozen=True)
class DatetimeProfile(ColumnProfile):
    min: datetime
    max: datetime
    type: Literal['datetime'] = 'datetime'

    def __post_init__(self):
        # accept ISO strings, convert to datetime
        if isinstance(self.min, str):
            object.__setattr__(self, 'min', datetime.fromisoformat(self.min))  # type: ignore[arg-type]
        if isinstance(self.max, str):
            object.__setattr__(self, 'max', datetime.fromisoformat(self.max))  # type: ignore[arg-type]

    def __str__(self) -> str:
        if pd.isna(self.min) or pd.isna(self.max):
            return (f"DatetimeProfile(min=None, max=None, "
                    f"missing_prob={self.missing_prob})")
        return (f"DatetimeProfile(min='{self.min.isoformat()}', "
                f"max='{self.max.isoformat()}', missing_prob={self.missing_prob})")

    def sample(self) -> Optional[datetime]:
        if self._maybe_missing():
            return None
        if pd.isna(self.min) or pd.isna(self.max):
            return None
        return _rand_datetime(self.min, self.max)

    def sample_script(self, ignore_missing: bool = False) -> str:
        min_iso = self.min.isoformat()
        max_iso = self.max.isoformat()
        base_script = f"datetime.fromisoformat('{min_iso}') + timedelta(seconds=random.randint(0, int((datetime.fromisoformat('{max_iso}') - datetime.fromisoformat('{min_iso}')).total_seconds())))"
        if not self.has_nulls() or ignore_missing:
            return base_script
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"{base_script} if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[datetime, pd.Series]) -> bool:
        # check if value is a datetime or a pd.Series of datetimes
        if isinstance(value, datetime):
            return True
        if isinstance(value, pd.Series):
            if is_datetime64_any_dtype(value):
                return True
            if value.dtype == 'object':
                # Check if all non-null values are datetime objects
                non_null_values = value.dropna()
                return not non_null_values.empty and all(isinstance(val, datetime) for val in non_null_values)
        return False

    def _contains_single_value(self, value: datetime) -> bool:
        if not isinstance(value, (datetime, pd.Timestamp)):
            return False
        return self.min <= value <= self.max


@dataclass(frozen=True)
class DatetimeStrProfile(ColumnProfile):
    min: str  # ISO-8601 with time
    max: str
    type: Literal['datetime_str'] = 'datetime_str'

    @property
    def min_datetime(self) -> datetime:
        return datetime.fromisoformat(self.min)
    
    @property
    def max_datetime(self) -> datetime:
        return datetime.fromisoformat(self.max)

    def __post_init__(self):
        # allow datetime objects on construction
        if isinstance(self.min, datetime):
            object.__setattr__(self, 'min', self.min.isoformat())  # type: ignore[arg-type]
        if isinstance(self.max, datetime):
            object.__setattr__(self, 'max', self.max.isoformat())  # type: ignore[arg-type]

    def __str__(self) -> str:
        if self.min is None or self.max is None or pd.isna(self.min) or pd.isna(self.max):
            return (f"DatetimeStrProfile(min=None, max=None, "
                    f"missing_prob={self.missing_prob})")
        return (f"DatetimeStrProfile(min='{self.min}', max='{self.max}', "
                f"missing_prob={self.missing_prob})")

    def sample(self) -> Optional[str]:
        if self._maybe_missing() or not self.min or not self.max:
            return None
        dt0 = datetime.fromisoformat(self.min)
        dt1 = datetime.fromisoformat(self.max)
        sampled_datetime = _rand_datetime(dt0, dt1)
        return sampled_datetime.isoformat()

    def sample_script(self, ignore_missing: bool = False) -> str:
        base_script = f"(datetime.fromisoformat('{self.min}') + timedelta(seconds=random.randint(0, int((datetime.fromisoformat('{self.max}') - datetime.fromisoformat('{self.min}')).total_seconds())))).isoformat()"
        if not self.has_nulls() or ignore_missing:
            return base_script
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"{base_script} if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[str, pd.Series]) -> bool:
        # check if value is a string or a pd.Series of strings that can be parsed as datetimes with time
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                # Check if it has time components (not just date)
                return dt.time() != datetime.min.time() or 'T' in value or ' ' in value
            except ValueError:
                return False
        if isinstance(value, pd.Series) and value.dtype == 'object':
            # Check if all non-null values are strings that can be parsed as datetimes with time
            non_null_values = value.dropna()
            if non_null_values.empty:
                return False
            # First check if any values are already datetime objects - if so, this is NOT a DatetimeStrProfile
            if any(isinstance(val, (datetime, date)) for val in non_null_values):
                return False
            # Check if all non-null values are strings
            if not all(isinstance(val, str) for val in non_null_values):
                return False
            # Then check if all values are strings that can be parsed as datetimes with time components
            try:
                for val in non_null_values:
                    dt = datetime.fromisoformat(val)
                    # Must have time components or explicit time markers
                    if dt.time() == datetime.min.time() and 'T' not in val and ' ' not in val:
                        return False
                return True
            except (ValueError, TypeError):
                return False
        return False

    def _contains_single_value(self, value: str) -> bool:
        if not isinstance(value, (str, np.str_)):
            return False
        value_datetime = datetime.fromisoformat(value)
        return self.min_datetime <= value_datetime <= self.max_datetime


# ────────────────────────────────────────────────────────────────────────────
# string / categorical
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StringProfile(ColumnProfile):
    values: List[str]
    type: Literal['string'] = 'string'

    def __str__(self) -> str:
        vals = ', '.join(f"'{v}'" for v in self.values)
        return (f"StringProfile(values=[{vals}], "
                f"missing_prob={self.missing_prob})")

    def sample(self) -> Optional[str]:
        if self._maybe_missing() or not self.values:
            return None
        return random.choice(self.values)

    def sample_script(self, ignore_missing: bool = False) -> str:
        values_repr = repr(self.values)
        base_script = f"random.choice({values_repr})"
        if not self.has_nulls() or ignore_missing:
            return base_script
        
        prob_str = _format_float_for_script(1 - self.missing_prob)
        return f"{base_script} if random.random() < {prob_str} else None"

    @classmethod
    def _type_check(cls, value: Union[str, pd.Series]) -> bool:
        # check if value is a string or a pd.Series of strings
        if isinstance(value, str):
            return True
        if isinstance(value, pd.Series) and value.dtype == 'object':
            # Check if all non-null values are strings
            non_null_values = value.dropna()
            return not non_null_values.empty and all(isinstance(val, str) for val in non_null_values)
        return False

    def _contains_single_value(self, value: str) -> bool:
        if not isinstance(value, (str, np.str_)):
            return False
        return str(value) in self.values

# endregion: ColumnProfile and subclasses

# region: profiling helper and DataFrameProfile

# ────────────────────────────────────────────────────────────────────────────
# profiling helper
# ────────────────────────────────────────────────────────────────────────────
def profile_column(
    df: pd.DataFrame,
    *,
    column: str,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Union[
    FloatProfile, IntProfile, DateStrProfile, DatetimeProfile, DatetimeStrProfile,
    StringProfile, DateProfile, BoolProfile,
    NullProfile, OneValueProfile,
]:
    """
    Build a ColumnProfile for *df[column]*:
      • trims numeric outliers to [q_low , q_high] quantiles
      • recognises bool / int / float / datetime / datetime-string / date / date-string / generic string
    """
    s = df[column]
    missing_prob = float(round(s.isna().mean(), 2))

    # null ------------------------------------------------------------------
    if s.isna().all():
        return NullProfile()
    
    # single value ------------------------------------------------------------------
    if s.nunique() == 1:
        # Get the actual non-null unique value, not just iloc[0] which might be None
        non_null_values = s.dropna()
        if non_null_values.empty:
            # This should be caught by the null check above, but just in case
            return NullProfile()
        raw_value = non_null_values.iloc[0]
        python_value = _convert_numpy_to_python(raw_value)
        return OneValueProfile(missing_prob=missing_prob, value=python_value)

    # bool  ------------------------------------------------------------------
    if BoolProfile.type_check(s):
        true_prob = round(s.dropna().mean(), 2) if s.notna().any() else 0.0
        return BoolProfile(missing_prob=missing_prob, true_prob=true_prob)                  # type: ignore[arg-type]


    # numeric ----------------------------------------------------------------
    if FloatProfile.type_check(s) or IntProfile.type_check(s):
        clean = s.dropna()
        if clean.empty:
            if is_integer_dtype(s):
                return IntProfile(missing_prob=missing_prob, min=np.nan, max=np.nan)  # type: ignore[arg-type]
            return FloatProfile(missing_prob=missing_prob, min=np.nan, max=np.nan)    # type: ignore[arg-type]
        lo, hi = clean.quantile([q_low, q_high])
        
        # Check if quantiles created min=max (effectively single value)
        if lo == hi:
            python_value = _convert_numpy_to_python(lo)
            return OneValueProfile(missing_prob=missing_prob, value=python_value)       # type: ignore[arg-type]
        
        # Only use IntProfile if the original dtype is actually integer
        if is_integer_dtype(s):
            return IntProfile(missing_prob=missing_prob, min=int(clean.min()), max=int(clean.max()))    # type: ignore[arg-type]
        return FloatProfile(missing_prob=missing_prob, min=float(lo), max=float(hi))  # type: ignore[arg-type]

    # datetime ---------------------------------------------------------------
    if DatetimeProfile.type_check(s):
        clean = s.dropna()
        if clean.empty:
            return DatetimeProfile(missing_prob=missing_prob, min=pd.NaT, max=pd.NaT)  # type: ignore[arg-type]
        min_datetime = clean.min()
        max_datetime = clean.max()
        # Convert pandas Timestamp to datetime if needed
        if hasattr(min_datetime, 'to_pydatetime'):
            min_datetime = min_datetime.to_pydatetime()
        if hasattr(max_datetime, 'to_pydatetime'):
            max_datetime = max_datetime.to_pydatetime()
        return DatetimeProfile(missing_prob=missing_prob, min=min_datetime, max=max_datetime)  # type: ignore[arg-type]

    # date -------------------------------------------------------------------
    if DateProfile.type_check(s):
        clean = s.dropna()
        if clean.empty:
            return DateProfile(missing_prob=missing_prob, min=pd.NaT, max=pd.NaT)     # type: ignore[arg-type]
        min_date = clean.min()
        max_date = clean.max()
        # Handle both date and datetime objects
        if hasattr(min_date, 'date'):
            min_date = min_date.date()
        if hasattr(max_date, 'date'):
            max_date = max_date.date()
        return DateProfile(missing_prob=missing_prob, min=min_date, max=max_date)     # type: ignore[arg-type]

    # datetime_str -----------------------------------------------------------
    if DatetimeStrProfile.type_check(s):
        # Try to parse as datetimes with time components
        try:
            parsed = pd.to_datetime(s.dropna(), errors="coerce")
            if parsed.notna().any():
                min_datetime = parsed.min().isoformat()
                max_datetime = parsed.max().isoformat()
                return DatetimeStrProfile(missing_prob=missing_prob, min=min_datetime, max=max_datetime)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            pass
        # otherwise treat as categorical
        values = sorted(s.dropna().unique().tolist())
        return StringProfile(missing_prob=missing_prob, values=values)                   # type: ignore[arg-type]

    # date_str ---------------------------------------------------------------
    if DateStrProfile.type_check(s):
        # Try to parse as dates first with explicit format
        try:
            parsed = pd.to_datetime(s.dropna(), format='%Y-%m-%d', errors="coerce")
            if parsed.notna().any():
                min_date = parsed.min().date().isoformat()
                max_date = parsed.max().date().isoformat()
                return DateStrProfile(missing_prob=missing_prob, min=min_date, max=max_date)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            # If explicit format fails, try flexible parsing
            parsed = pd.to_datetime(s.dropna(), errors="coerce")
            if parsed.notna().any():
                min_date = parsed.min().date().isoformat()
                max_date = parsed.max().date().isoformat()
                return DateStrProfile(missing_prob=missing_prob, min=min_date, max=max_date)  # type: ignore[arg-type]
        # otherwise treat as categorical
        values = sorted(s.dropna().unique().tolist())
        return StringProfile(missing_prob=missing_prob, values=values)                   # type: ignore[arg-type]

    # fallback as strings
    values = sorted(s.dropna().astype(str).unique().tolist())
    return StringProfile(missing_prob=missing_prob, values=values)                       # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────────────
# DataFrame-level wrapper
# ────────────────────────────────────────────────────────────────────────────
class DataFrameProfile:
    def __init__(
        self,
        column_profiles: Dict[str, ColumnProfile],
        q_low: float = 0.10,
        q_high: float = 0.90,
    ):
        if not isinstance(column_profiles, dict):
            raise ValueError("column_profiles must be a dictionary")
        if not all(isinstance(profile, ColumnProfile) for profile in column_profiles.values()):
            raise ValueError("column_profiles must be a dictionary of ColumnProfile objects")
        
        self.column_profiles = column_profiles
        self.q_low = q_low
        self.q_high = q_high

    def __getitem__(self, key: str) -> ColumnProfile:
        return self.column_profiles[key]

    @property
    def columns(self) -> List[str]:
        return list(self.column_profiles.keys())

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        q_low: float = 0.10,
        q_high: float = 0.90,
    ) -> 'DataFrameProfile':
        profiles = {
            column: profile_column(df, column=column, q_low=q_low, q_high=q_high)
            for column in df.columns
        }
        return cls(profiles, q_low, q_high)

    # pretty-print so you can copy-paste into a script
    def __str__(self) -> str:
        profiles_str = ',\n  '.join(
            f"'{col}': {profile}" for col, profile in self.column_profiles.items()
        )
        return f"{{\n  {profiles_str}\n}}"

    __repr__ = __str__

    # generate synthetic DataFrame
    def generate_df(self, num_rows: int) -> pd.DataFrame:
        data = {
            column: profile.samples(num_rows)
            for column, profile in self.column_profiles.items()
        }
        return pd.DataFrame(data)

    def generate_dict(self) -> Dict[str, Any]:
        dictionary = {
            column: profile.sample()
            for column, profile in self.column_profiles.items()
        }
        return dictionary
    
    def generate_row(self) -> pd.Series:
        return pd.Series(self.generate_dict())

    def generate_df_script(self, num_rows: int, indent: str = '    ', 
                          use_variable: bool = False, variable_name: str = "n_values") -> str:
        """
        Generate a script to generate a DataFrame with the same structure as the profile.
        
        Args:
            num_rows: Number of rows to generate (used when use_variable=False)
            indent: Indentation string for multi-line formatting
            use_variable: If True, use variable_name instead of hard-coding num_rows
            variable_name: Variable name to use when use_variable=True
            
        Returns:
            String containing executable Python code to generate a DataFrame
            
        Example with use_variable=False (default):
            pd.DataFrame({
                'int_col': [random.randint(0, 10) for _ in range(100)],
                'str_col': [random.choice(['a', 'b']) for _ in range(100)]
            })
            
        Example with use_variable=True:
            pd.DataFrame({
                'int_col': [random.randint(0, 10) for _ in range(n_values)],
                'str_col': [random.choice(['a', 'b']) for _ in range(n_values)]
            })
        """
        middle_part = ",\n".join(
            f"{indent}'{col}': {profile.samples_script(num_rows, single_line=True, use_variable=use_variable, variable_name=variable_name)}"
            for col, profile in self.column_profiles.items()
        )
        return (
            "pd.DataFrame({\n" + middle_part + "\n})"
        )

# endregion: profiling helper and DataFrameProfile
