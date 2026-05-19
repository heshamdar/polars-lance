use std::sync::Arc;

use arrow::error::ArrowError;
use arrow::record_batch::RecordBatchIterator;
use lance::dataset::{Dataset, WriteMode as LanceWriteMode, WriteParams};
use lance::io::ObjectStoreParams;
use polars::frame::chunk_df_for_writing;
use polars::prelude::{CompatLevel, DataFrame, SchemaExt};

use crate::arrow::{ArrowBridgeError, PolarsArrowRecordBatchExt, PolarsArrowSchemaExt};
use crate::err::LanceWriterError;
use crate::io::StorageOptions;
use crate::sync::TOKIO_RUNTIME;

const LANCE_ARROW_COMPAT_LEVEL: CompatLevel = CompatLevel::oldest();

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

fn maybe_build_object_store_params(storage_options: StorageOptions) -> Option<ObjectStoreParams> {
    storage_options.map(|storage_options| ObjectStoreParams {
        storage_options: Some(storage_options),
        ..ObjectStoreParams::default()
    })
}

fn build_write_params(
    mode: PolarsLanceWriteMode,
    storage_options: StorageOptions,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
) -> WriteParams {
    let mut write_params = WriteParams {
        mode: mode.into(),
        store_params: maybe_build_object_store_params(storage_options),
        ..WriteParams::default()
    };

    if let Some(max_rows_per_file) = max_rows_per_file {
        write_params.max_rows_per_file = max_rows_per_file;
    }
    if let Some(max_bytes_per_file) = max_bytes_per_file {
        write_params.max_bytes_per_file = max_bytes_per_file;
    }

    write_params
}

pub fn write_lance_dataset(
    df: DataFrame,
    uri: &str,
    mode: PolarsLanceWriteMode,
    storage_options: StorageOptions,
    max_rows_per_file: Option<usize>,
    max_bytes_per_file: Option<usize>,
) -> Result<(), LanceWriterError> {
    let mut df = chunk_df_for_lance_write(df)?;

    let dfs = df.split_chunks().collect::<Vec<_>>();

    let batches = dfs.into_iter().map(|df| {
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
    });

    let schema = df
        .schema()
        .to_arrow(LANCE_ARROW_COMPAT_LEVEL)
        .to_arrow_schema()?;

    let batch_iterator = RecordBatchIterator::new(batches, Arc::new(schema));

    let write_params =
        build_write_params(mode, storage_options, max_rows_per_file, max_bytes_per_file);

    TOKIO_RUNTIME.block_on(Dataset::write(batch_iterator, uri, Some(write_params)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{build_write_params, PolarsLanceWriteMode};
    use lance::dataset::{WriteMode as LanceWriteMode, WriteParams};

    #[test]
    fn build_write_params_falls_back_to_defaults() {
        let write_params = build_write_params(PolarsLanceWriteMode::Error, None, None, None);

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
        );

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
