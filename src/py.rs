use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use polars::prelude::DataFrame;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySchema};

use crate::blob::wrap_blob_v2_columns;
use crate::exc::lance_error_to_pyerr;
use crate::io::StorageOptions;
use crate::{
    arrow_schema_for_write, df_to_record_batches, resolve_data_storage_version,
    write_lance_dataset, write_lance_dataset_from_df, LanceReader, LanceScanner, LanceScannerError,
    LanceScannerOptions, LanceWriterError, PolarsLanceWriteMode,
};

// The `Arrow` arm carries this crate's own argument checks — `resolve_data_storage_version`
// and `mark_blob_columns` both report a bad `data_storage_version` or `blob_columns` through
// it — so it maps to `ValueError`. Lance's *internal* Arrow failures travel in
// `LanceError::Arrow` instead and are deliberately not mapped that way; see `exc.rs`.
impl From<LanceScannerError> for PyErr {
    fn from(err: LanceScannerError) -> Self {
        match err {
            LanceScannerError::Polars(err) => PyErr::from(PyPolarsErr::from(err)),
            LanceScannerError::Lance(err) => lance_error_to_pyerr(&err),
            LanceScannerError::Arrow(err) => PyValueError::new_err(err.to_string()),
        }
    }
}

impl From<LanceWriterError> for PyErr {
    fn from(err: LanceWriterError) -> Self {
        match err {
            LanceWriterError::Polars(err) => PyErr::from(PyPolarsErr::from(err)),
            LanceWriterError::Lance(err) => lance_error_to_pyerr(&err),
            LanceWriterError::Arrow(err) => PyValueError::new_err(err.to_string()),
        }
    }
}

#[pyclass(name = "LanceReader")]
pub struct PyLanceReader(LanceReader);

#[pymethods]
impl PyLanceReader {
    // Opening loads the dataset's manifest, which for a remote dataset is a network round
    // trip. Holding the GIL across it would stop every other Python thread for its duration.
    #[new]
    #[pyo3(signature = (uri, storage_options=None))]
    fn new(py: Python, uri: String, storage_options: Option<StorageOptions>) -> PyResult<Self> {
        py.detach(|| LanceReader::open(&uri, storage_options))
            .map(Self)
            .map_err(PyErr::from)
    }

    fn schema(&self, py: Python) -> PyResult<PySchema> {
        py.detach(|| self.0.schema())
            .map(|schema| PySchema(Arc::new(schema)))
            .map_err(PyErr::from)
    }

    #[pyo3(signature = (with_columns=None, filter=None, n_rows=None, batch_size=None))]
    fn scanner(
        &self,
        with_columns: Option<Vec<String>>,
        filter: Option<String>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> PyLanceScanner {
        PyLanceScanner(Mutex::new(self.0.scanner(LanceScannerOptions {
            with_columns,
            filter,
            n_rows,
            batch_size,
        })))
    }
}

/// The scan is behind a mutex so that the object can be shared between threads: the
/// streaming engine may drive a scan from a worker thread, and Lance's record batch stream
/// is `Send` but not `Sync`.
#[pyclass(name = "LanceScanner")]
pub struct PyLanceScanner(Mutex<LanceScanner>);

#[pymethods]
impl PyLanceScanner {
    /// Reading a batch is the slowest thing this extension does, and for a remote dataset it
    /// is mostly waiting on the network, so the GIL is released for the duration. The
    /// streaming engine drives scans from worker threads, which cannot proceed otherwise.
    fn next(&self, py: Python) -> PyResult<Option<PyDataFrame>> {
        let df = py.detach(|| {
            self.0
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Lance scanner is poisoned"))?
                .next()
                .map_err(PyErr::from)
        })?;

        Ok(df.map(PyDataFrame))
    }
}

fn parse_write_mode(mode: &str) -> PyResult<PolarsLanceWriteMode> {
    match mode {
        "error" => Ok(PolarsLanceWriteMode::Error),
        "append" => Ok(PolarsLanceWriteMode::Append),
        "overwrite" => Ok(PolarsLanceWriteMode::Overwrite),
        _ => Err(PyValueError::new_err(
            "`mode` must be one of: 'error', 'append', 'overwrite'",
        )),
    }
}

/// A record batch reader that pulls dataframes from a Python iterator.
///
/// Lance reads one batch at a time, so a generator that produces dataframes lazily (such as
/// `LazyFrame.collect_batches`) is written without ever holding the whole frame in memory.
struct PyDataFrameBatchReader {
    dataframes: Py<PyAny>,
    schema: SchemaRef,
    /// Batches of the dataframe read most recently, which may have several chunks.
    pending: VecDeque<RecordBatch>,
    done: bool,
}

impl PyDataFrameBatchReader {
    fn new(dataframes: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            dataframes,
            schema,
            pending: VecDeque::new(),
            done: false,
        }
    }

    /// Pull the next dataframe, returning `false` once the iterator is exhausted.
    fn fill(&mut self) -> Result<bool, ArrowError> {
        // Called from the runtime driving the write, so the GIL has to be taken here.
        let df = Python::attach(|py| -> PyResult<Option<DataFrame>> {
            let next = self.dataframes.bind(py).call_method0("__next__");
            match next {
                Ok(df) => Ok(Some(df.extract::<PyDataFrame>()?.into())),
                Err(err) if err.is_instance_of::<PyStopIteration>(py) => Ok(None),
                Err(err) => Err(err),
            }
        })
        .map_err(|err| ArrowError::ExternalError(Box::new(err)))?;

        let Some(df) = df else {
            self.done = true;
            return Ok(false);
        };

        for batch in df_to_record_batches(df).map_err(to_arrow_error)? {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }

            // A reader has to hand out batches matching the schema it advertises, which also
            // means wrapping a blob v2 column in the struct that schema declares.
            self.pending
                .push_back(wrap_blob_v2_columns(&batch, &self.schema)?);
        }

        Ok(true)
    }
}

fn to_arrow_error(err: LanceWriterError) -> ArrowError {
    ArrowError::ExternalError(Box::new(err))
}

impl Iterator for PyDataFrameBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return Some(Ok(batch));
            }

            if self.done {
                return None;
            }

            // An empty dataframe yields no batches, so keep pulling until one has rows.
            match self.fill() {
                Ok(_) => continue,
                Err(err) => {
                    self.done = true;
                    return Some(Err(err));
                }
            }
        }
    }
}

impl RecordBatchReader for PyDataFrameBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (
    df,
    target,
    *,
    mode = "error",
    storage_options = None,
    max_rows_per_file = None,
    max_bytes_per_file = None,
    data_storage_version = None,
    blob_columns = None
))]
fn write_lance(
    py: Python,
    df: PyDataFrame,
    target: String,
    mode: &str,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: Option<String>,
    blob_columns: Option<Vec<String>>,
) -> PyResult<()> {
    let mode = parse_write_mode(mode)?;
    let df: DataFrame = df.into();
    let blob_columns = blob_columns.unwrap_or_default();

    // Writing is pure Rust once the dataframe is in hand, so nothing here needs the GIL.
    py.detach(|| {
        write_lance_dataset_from_df(
            df,
            &target,
            mode,
            storage_options,
            max_rows_per_file,
            max_bytes_per_file,
            data_storage_version.as_deref(),
            &blob_columns,
        )
    })
    .map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (
    dataframes,
    schema,
    target,
    *,
    mode = "error",
    storage_options = None,
    max_rows_per_file = None,
    max_bytes_per_file = None,
    data_storage_version = None,
    blob_columns = None
))]
fn write_lance_stream(
    py: Python,
    dataframes: Py<PyAny>,
    schema: PyDataFrame,
    target: String,
    mode: &str,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: Option<String>,
    blob_columns: Option<Vec<String>>,
) -> PyResult<()> {
    let mode = parse_write_mode(mode)?;
    let blob_columns = blob_columns.unwrap_or_default();
    let version = resolve_data_storage_version(data_storage_version.as_deref(), &blob_columns)
        .map_err(PyErr::from)?;

    let schema: DataFrame = schema.into();
    let schema = Arc::new(arrow_schema_for_write(&schema, &blob_columns).map_err(PyErr::from)?);
    let reader = PyDataFrameBatchReader::new(dataframes, schema);

    // The reader takes the GIL to pull each dataframe, so it cannot be held here.
    py.detach(|| {
        write_lance_dataset(
            reader,
            &target,
            mode,
            storage_options,
            max_rows_per_file,
            max_bytes_per_file,
            version,
        )
    })
    .map_err(PyErr::from)
}

#[pymodule]
fn _polars_lance(m: &Bound<PyModule>) -> PyResult<()> {
    // Lance guards parts of its 2.1 encoder with `debug_assert!`s that a nullable nested column
    // can trip (lance-format/lance#8032, #8033). They vanish in a release build, so which data
    // this extension can write depends on how it was compiled, and the tests need to know.
    m.add("_debug_assertions", cfg!(debug_assertions))?;
    crate::exc::register(m)?;
    m.add_class::<PyLanceReader>()?;
    m.add_class::<PyLanceScanner>()?;
    m.add_function(wrap_pyfunction!(write_lance, m)?)?;
    m.add_function(wrap_pyfunction!(write_lance_stream, m)?)?;
    Ok(())
}
