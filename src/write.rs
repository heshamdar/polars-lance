use std::sync::Arc;

use arrow::datatypes::Schema as ArrowSchema;
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use lance::dataset::{Dataset, WriteMode as LanceWriteMode, WriteParams};
use lance::io::ObjectStoreParams;
use polars::frame::chunk_df_for_writing;
use polars::prelude::{CompatLevel, DataFrame, SchemaExt};

use crate::arrow::{ArrowBridgeError, PolarsArrowRecordBatchExt, PolarsArrowSchemaExt};
use crate::err::LanceWriterError;
use crate::io::StorageOptions;
use crate::sync::TOKIO_RUNTIME;

const LANCE_ARROW_COMPAT_LEVEL: CompatLevel = CompatLevel::oldest();

/// Data storage version to write new datasets with, unless the caller picks another.
///
/// Version 2.0 does not record the validity of a struct itself, so a null struct read back
/// from it is a valid struct holding filler values. 2.1 preserves it, which keeps what is
/// written, stored, and read consistent. Lance still defaults to 2.0, so this is set
/// explicitly. Appending to an existing dataset keeps that dataset's version regardless.
///
/// Named by its string form because Lance does not re-export the version type.
pub const DEFAULT_DATA_STORAGE_VERSION: &str = "2.1";

pub enum PolarsLanceWriteMode {
    Error,
    Append,
    Overwrite,
}

impl From<PolarsLanceWriteMode> for LanceWriteMode {
    fn from(mode: PolarsLanceWriteMode) -> Self {
        match mode {
            PolarsLanceWriteMode::Error => Self::Create,
            PolarsLanceWriteMode::Append => Self::Append,
            PolarsLanceWriteMode::Overwrite => Self::Overwrite,
        }
    }
}

fn chunk_df_for_lance_write(mut df: DataFrame) -> Result<DataFrame, LanceWriterError> {
    // 512 * 512 matches chunk size used internally by Polars.
    Ok(chunk_df_for_writing(&mut df, 512 * 512)?.into_owned())
}

fn maybe_build_object_store_params(
    storage_options: Option<StorageOptions>,
) -> Option<ObjectStoreParams> {
    storage_options.map(|storage_options| ObjectStoreParams {
        storage_options: Some(storage_options),
        ..ObjectStoreParams::default()
    })
}

fn build_write_params(
    mode: PolarsLanceWriteMode,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: Option<&str>,
) -> Result<WriteParams, LanceWriterError> {
    let data_storage_version = data_storage_version.unwrap_or(DEFAULT_DATA_STORAGE_VERSION);

    let mut write_params = WriteParams {
        mode: mode.into(),
        store_params: maybe_build_object_store_params(storage_options),
        data_storage_version: Some(data_storage_version.parse()?),
        ..WriteParams::default()
    };

    if let Some(max_rows_per_file) = max_rows_per_file {
        write_params.max_rows_per_file = max_rows_per_file;
    }
    if let Some(max_bytes_per_file) = max_bytes_per_file {
        write_params.max_bytes_per_file = max_bytes_per_file;
    }

    Ok(write_params)
}

/// Convert a dataframe into the record batches to write, one per chunk.
pub fn df_to_record_batches(
    df: DataFrame,
) -> Result<Vec<Result<RecordBatch, ArrowError>>, LanceWriterError> {
    let mut df = chunk_df_for_lance_write(df)?;

    Ok(df
        .split_chunks()
        .map(|df| {
            let mut batches = df.iter_chunks(LANCE_ARROW_COMPAT_LEVEL, false);

            let batch = batches
                .next()
                .expect("chunk dataframe should yield one record batch");
            assert!(
                batches.next().is_none(),
                "chunk dataframe should yield exactly one record batch"
            );

            batch.to_arrow_record_batch().map_err(|err| match err {
                ArrowBridgeError::Arrow(err) => err,
                ArrowBridgeError::Polars(err) => ArrowError::ExternalError(Box::new(err)),
            })
        })
        .collect())
}

/// The Arrow schema Lance is given for a write, derived from a dataframe's schema.
pub fn arrow_schema_for_write(df: &DataFrame) -> Result<ArrowSchema, LanceWriterError> {
    Ok(df
        .schema()
        .to_arrow(LANCE_ARROW_COMPAT_LEVEL)
        .to_arrow_schema()?)
}

/// Write record batches to a Lance dataset as they are produced.
///
/// The batches are pulled one at a time, so a reader that produces them lazily is never
/// fully held in memory.
pub fn write_lance_dataset<R>(
    batches: R,
    uri: &str,
    mode: PolarsLanceWriteMode,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: Option<&str>,
) -> Result<(), LanceWriterError>
where
    R: RecordBatchReader + Send + 'static,
{
    let write_params = build_write_params(
        mode,
        storage_options,
        max_rows_per_file,
        max_bytes_per_file,
        data_storage_version,
    )?;

    TOKIO_RUNTIME.block_on(Dataset::write(batches, uri, Some(write_params)))?;
    Ok(())
}

/// Write a single dataframe to a Lance dataset.
pub fn write_lance_dataset_from_df(
    df: DataFrame,
    uri: &str,
    mode: PolarsLanceWriteMode,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: Option<&str>,
) -> Result<(), LanceWriterError> {
    let schema = Arc::new(arrow_schema_for_write(&df)?);
    let batches = df_to_record_batches(df)?;

    write_lance_dataset(
        RecordBatchIterator::new(batches.into_iter(), schema),
        uri,
        mode,
        storage_options,
        max_rows_per_file,
        max_bytes_per_file,
        data_storage_version,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{build_write_params, PolarsLanceWriteMode};
    use lance::dataset::{WriteMode as LanceWriteMode, WriteParams};

    #[test]
    fn build_write_params_falls_back_to_defaults() {
        let write_params =
            build_write_params(PolarsLanceWriteMode::Error, None, None, None, None).unwrap();

        let default_write_params = WriteParams::default();
        assert_eq!(
            write_params.max_rows_per_file,
            default_write_params.max_rows_per_file
        );
        assert_eq!(
            write_params.max_rows_per_group,
            default_write_params.max_rows_per_group
        );
        assert_eq!(
            write_params.max_bytes_per_file,
            default_write_params.max_bytes_per_file
        );
    }

    /// New datasets are written with the version that preserves struct validity, which is
    /// not the Lance default.
    #[test]
    fn build_write_params_sets_the_data_storage_version() {
        let write_params =
            build_write_params(PolarsLanceWriteMode::Error, None, None, None, None).unwrap();

        let version = write_params
            .data_storage_version
            .expect("data storage version should be set");
        assert_eq!(version.to_string(), "2.1");
        assert_ne!(Some(version), WriteParams::default().data_storage_version);
    }

    #[test]
    fn build_write_params_applies_overrides() {
        let storage_options = HashMap::from([("aws_region".to_owned(), "us-east-1".to_owned())]);
        let max_rows_per_file = 100;
        let max_bytes_per_file = 2048;

        let write_params = build_write_params(
            PolarsLanceWriteMode::Append,
            Some(storage_options),
            Some(max_rows_per_file),
            Some(max_bytes_per_file),
            None,
        )
        .unwrap();

        assert!(matches!(write_params.mode, LanceWriteMode::Append));
        assert_eq!(
            write_params
                .store_params
                .expect("store params should be set")
                .storage_options
                .expect("storage options should be set")
                .get("aws_region"),
            Some(&"us-east-1".to_owned())
        );
        assert_eq!(write_params.max_rows_per_file, max_rows_per_file);
        assert_eq!(write_params.max_bytes_per_file, max_bytes_per_file);
    }
}
