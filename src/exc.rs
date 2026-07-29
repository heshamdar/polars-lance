//! Python exceptions.
//!
//! Lance reports every failure through one error enum, and turning all of it into
//! `RuntimeError` leaves a caller unable to tell a missing dataset from a write that refused
//! to clobber one. Variants with an exact Python counterpart are raised as that builtin; the
//! rest keep their message and arrive as [`PolarsLanceError`].
//!
//! `PolarsLanceError` derives from `RuntimeError` because that is what every failure used to
//! raise, so code catching `RuntimeError` keeps working across this change.

use lance::Error as LanceError;
use pyo3::create_exception;
use pyo3::exceptions::{
    PyFileExistsError, PyFileNotFoundError, PyNotImplementedError, PyOSError, PyRuntimeError,
    PyTimeoutError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

create_exception!(
    _polars_lance,
    PolarsLanceError,
    PyRuntimeError,
    "Base class for polars-lance errors that have no exact Python counterpart."
);

create_exception!(
    _polars_lance,
    CommitConflictError,
    PolarsLanceError,
    "A concurrent write won the race to commit."
);

/// Map a Lance error to the closest Python exception.
///
/// Only variants whose meaning matches a builtin exactly are mapped to one, so a caller is
/// never misled about what failed. Two omissions are deliberate:
///
/// - `Error::Arrow` is not a `ValueError`. It carries Lance's internal Arrow failures, which
///   include an error raised by the caller's own query on its way out of a streaming write —
///   nothing to do with a bad argument. Our own argument checks live in `LanceWriterError::Arrow`
///   and are mapped separately, in `py.rs`.
/// - `Error::CorruptFile` is not an `OSError`. The read succeeded; the bytes were wrong.
pub fn lance_error_to_pyerr(err: &LanceError) -> PyErr {
    let message = err.to_string();

    match err {
        LanceError::DatasetNotFound { .. }
        | LanceError::NotFound { .. }
        | LanceError::VersionNotFound { .. } => PyFileNotFoundError::new_err(message),

        LanceError::DatasetAlreadyExists { .. } => PyFileExistsError::new_err(message),

        LanceError::InvalidInput { .. }
        | LanceError::Schema { .. }
        | LanceError::SchemaMismatch { .. } => PyValueError::new_err(message),

        LanceError::IO { .. } => PyOSError::new_err(message),

        LanceError::NotSupported { .. } => PyNotImplementedError::new_err(message),

        LanceError::Timeout { .. } => PyTimeoutError::new_err(message),

        LanceError::CommitConflict { .. } | LanceError::RetryableCommitConflict { .. } => {
            CommitConflictError::new_err(message)
        }

        _ => PolarsLanceError::new_err(message),
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("PolarsLanceError", m.py().get_type::<PolarsLanceError>())?;
    m.add(
        "CommitConflictError",
        m.py().get_type::<CommitConflictError>(),
    )?;
    Ok(())
}
