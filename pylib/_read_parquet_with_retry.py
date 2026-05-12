import logging
import time
import typing

import pandas as pd

logger = logging.getLogger(__name__)


def _read_with_retry(
    reader: typing.Callable,
    path: str,
    *args: typing.Any,
    retries: int = 10,
    backoff: float = 2.0,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """Wrapper around a pandas read function with exponential-backoff retries."""
    for attempt in range(retries + 1):
        try:
            return reader(path, *args, **kwargs)
        except Exception:
            if attempt >= retries:
                raise
            wait = backoff ** (attempt + 1)
            logger.warning(
                "%s attempt %d/%d failed for %s, " "retrying in %.1fs...",
                reader.__name__,
                attempt + 1,
                retries,
                path,
                wait,
            )
            time.sleep(wait)


def read_parquet_with_retry(
    path: str,
    *args: typing.Any,
    retries: int = 10,
    backoff: float = 2.0,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """Wrapper around pd.read_parquet with exponential-backoff retries."""
    return _read_with_retry(
        pd.read_parquet,
        path,
        *args,
        retries=retries,
        backoff=backoff,
        **kwargs,
    )


def read_csv_with_retry(
    path: str,
    *args: typing.Any,
    retries: int = 10,
    backoff: float = 2.0,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """Wrapper around pd.read_csv with exponential-backoff retries."""
    return _read_with_retry(
        pd.read_csv, path, *args, retries=retries, backoff=backoff, **kwargs
    )
