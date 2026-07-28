use std::sync::Arc;

use arrow::datatypes::Schema as ArrowSchema;
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use lance::dataset::{Dataset, WriteMode as LanceWriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance_file::version::LanceFileVersion;
use polars::frame::chunk_df_for_writing;
use polars::prelude::{CompatLevel, DataFrame, SchemaExt};

use crate::arrow::{ArrowBridgeError, PolarsArrowRecordBatchExt, PolarsArrowSchemaExt};
use crate::blob::{mark_blob_columns, wrap_blob_v2_columns, BlobLayout, BlobNullability};
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

/// The version a blob column is written with, unless the caller picks another.
///
/// Blob columns are the one case where 2.1 is not the better choice: its blob layout loses a
/// null when fragments that disagree about nullability are compacted, which 2.2 fixes by
/// describing the column as an extension type instead (lance-format/lance#7955).
pub const DEFAULT_BLOB_DATA_STORAGE_VERSION: &str = "2.2";

/// The version to write with, and how that version wants a blob column described.
pub fn resolve_data_storage_version(
    data_storage_version: Option<&str>,
    blob_columns: &[String],
) -> Result<(LanceFileVersion, BlobLayout), LanceWriterError> {
    let version = data_storage_version.unwrap_or(if blob_columns.is_empty() {
        DEFAULT_DATA_STORAGE_VERSION
    } else {
        DEFAULT_BLOB_DATA_STORAGE_VERSION
    });

    let version: LanceFileVersion = version.parse()?;
    Ok((version, BlobLayout::for_version(version)))
}

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
        // Lance 9 reads storage options through an accessor rather than a plain map.
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            storage_options,
        ))),
        ..ObjectStoreParams::default()
    })
}

fn build_write_params(
    mode: PolarsLanceWriteMode,
    storage_options: Option<StorageOptions>,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
    data_storage_version: LanceFileVersion,
) -> Result<WriteParams, LanceWriterError> {
    let mut write_params = WriteParams {
        mode: mode.into(),
        store_params: maybe_build_object_store_params(storage_options),
        data_storage_version: Some(data_storage_version),
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
pub fn arrow_schema_for_write(
    df: &DataFrame,
    blob_columns: &[String],
    layout: BlobLayout,
) -> Result<ArrowSchema, LanceWriterError> {
    let schema = df
        .schema()
        .to_arrow(LANCE_ARROW_COMPAT_LEVEL)
        .to_arrow_schema()?;

    mark_blob_columns(schema, blob_columns, layout).map_err(LanceWriterError::Arrow)
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
    data_storage_version: LanceFileVersion,
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
    blob_columns: &[String],
) -> Result<(), LanceWriterError> {
    let (version, layout) = resolve_data_storage_version(data_storage_version, blob_columns)?;
    let schema = Arc::new(arrow_schema_for_write(&df, blob_columns, layout)?);
    let batches = df_to_record_batches(df)?;

    // The legacy blob layout needs every batch of a write to agree about nullability; the
    // extension type records it per value, so it has no such constraint.
    let mut blob_nullability = match layout {
        BlobLayout::Legacy => BlobNullability::new(blob_columns),
        BlobLayout::V2 => BlobNullability::new(&[]),
    };

    let batches = batches
        .into_iter()
        .map(|batch| {
            let batch = batch?;
            blob_nullability.check(&batch)?;
            wrap_blob_v2_columns(&batch, &schema)
        })
        .collect::<Result<Vec<_>, ArrowError>>()
        .map_err(LanceWriterError::Arrow)?;

    write_lance_dataset(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        uri,
        mode,
        storage_options,
        max_rows_per_file,
        max_bytes_per_file,
        version,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        build_write_params, resolve_data_storage_version, BlobLayout, LanceFileVersion,
        PolarsLanceWriteMode,
    };
    use lance::dataset::{WriteMode as LanceWriteMode, WriteParams};

    fn blob(name: &str) -> Vec<String> {
        vec![name.to_owned()]
    }

    /// New datasets are written with the version that preserves struct validity, which is not
    /// the Lance default.
    #[test]
    fn resolve_data_storage_version_defaults_to_2_1() {
        let (version, layout) = resolve_data_storage_version(None, &[]).unwrap();

        assert_eq!(version.to_string(), "2.1");
        assert_eq!(layout, BlobLayout::Legacy);
        assert_ne!(
            Some(version),
            WriteParams::default().data_storage_version,
            "should not be Lance's own default"
        );
    }

    /// A blob column is the one case where 2.1 is not the better choice, because its blob
    /// layout loses a null when disagreeing fragments are compacted.
    #[test]
    fn resolve_data_storage_version_defaults_a_blob_column_to_2_2() {
        let (version, layout) = resolve_data_storage_version(None, &blob("blob")).unwrap();

        assert_eq!(version.to_string(), "2.2");
        assert_eq!(layout, BlobLayout::V2);
    }

    /// The version the caller asked for decides the layout, since neither version accepts the
    /// other's.
    #[test]
    fn resolve_data_storage_version_follows_an_explicit_version() {
        for (asked, expected, layout) in [
            ("2.0", "2.0", BlobLayout::Legacy),
            ("2.1", "2.1", BlobLayout::Legacy),
            ("stable", "stable", BlobLayout::Legacy),
            ("2.2", "2.2", BlobLayout::V2),
            ("2.3", "2.3", BlobLayout::V2),
        ] {
            let (version, resolved) =
                resolve_data_storage_version(Some(asked), &blob("blob")).unwrap();

            assert_eq!(version.to_string(), expected, "for {asked}");
            assert_eq!(resolved, layout, "for {asked}");
        }
    }

    #[test]
    fn resolve_data_storage_version_rejects_an_unknown_version() {
        assert!(resolve_data_storage_version(Some("9.9"), &[]).is_err());
    }

    #[test]
    fn build_write_params_falls_back_to_defaults() {
        let write_params = build_write_params(
            PolarsLanceWriteMode::Error,
            None,
            None,
            None,
            LanceFileVersion::V2_1,
        )
        .unwrap();

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
            LanceFileVersion::V2_1,
        )
        .unwrap();

        assert!(matches!(write_params.mode, LanceWriteMode::Append));
        let store_params = write_params
            .store_params
            .expect("store params should be set");
        assert_eq!(
            store_params
                .storage_options()
                .expect("storage options should be set")
                .get("aws_region"),
            Some(&"us-east-1".to_owned())
        );
        assert_eq!(write_params.max_rows_per_file, max_rows_per_file);
        assert_eq!(write_params.max_bytes_per_file, max_bytes_per_file);
    }
}
