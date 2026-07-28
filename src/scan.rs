use arrow::datatypes::Schema as ArrowSchema;
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance::datatypes::BlobHandling;
use lance::{Dataset, Error as LanceError};
use polars::prelude::{DataFrame, Schema, SchemaExt};

use crate::arrow::{ArrowRecordBatchExt, ArrowSchemaExt};
use crate::blob::{unwrap_blob_v2_batch, unwrap_blob_v2_fields};
use crate::err::LanceScannerError;
use crate::io::StorageOptions;
use crate::sync::TOKIO_RUNTIME;

#[derive(Clone, Default)]
pub struct LanceScannerOptions {
    pub with_columns: Option<Vec<String>>,
    /// A Lance SQL filter, translated from the Polars predicate by the caller.
    pub filter: Option<String>,
    pub n_rows: Option<usize>,
    pub batch_size: Option<usize>,
}

/// An opened Lance dataset, used to read its schema and to create scanners for it.
///
/// Opening the dataset once and reusing it avoids loading its manifest again for every
/// scan.
pub struct LanceReader {
    dataset: Dataset,
}

impl LanceReader {
    pub fn open(
        uri: &str,
        storage_options: Option<StorageOptions>,
    ) -> Result<Self, LanceScannerError> {
        let dataset = LanceScanner::open_dataset(uri, storage_options)?;
        Ok(Self { dataset })
    }

    pub fn schema(&self) -> Result<Schema, LanceScannerError> {
        // A blob v2 column is described by an extension type that Polars cannot represent, and
        // that a scan does not return anyway: asking for the bytes yields plain binary.
        let arrow_schema = unwrap_blob_v2_fields(ArrowSchema::from(self.dataset.schema()));
        let polars_arrow_schema = arrow_schema.to_polars_arrow_schema()?;
        Ok(Schema::from_arrow_schema(&polars_arrow_schema))
    }

    pub fn scanner(&self, options: LanceScannerOptions) -> LanceScanner {
        LanceScanner::new(self.dataset.clone(), options)
    }
}

pub struct LanceScanner {
    dataset: Dataset,
    options: LanceScannerOptions,
    stream: Option<DatasetRecordBatchStream>,
}

impl LanceScanner {
    pub fn new(dataset: Dataset, options: LanceScannerOptions) -> Self {
        Self {
            dataset,
            options,
            stream: None,
        }
    }

    pub fn next(&mut self) -> Result<Option<DataFrame>, LanceScannerError> {
        if matches!(self.options.n_rows, Some(0)) {
            return Ok(None);
        }

        let stream = self.get_or_init_stream()?;
        let next_batch = Self::next_batch(stream)?;

        let Some(batch) = next_batch else {
            return Ok(None);
        };

        let batch = unwrap_blob_v2_batch(batch).map_err(LanceScannerError::Arrow)?;
        Ok(Some(DataFrame::from(batch.to_polars_arrow_record_batch()?)))
    }

    fn open_dataset(
        uri: &str,
        storage_options: Option<StorageOptions>,
    ) -> Result<Dataset, LanceError> {
        TOKIO_RUNTIME.block_on(Self::build_lance_dataset_builder(uri, storage_options).load())
    }

    fn build_lance_dataset_builder(
        uri: &str,
        storage_options: Option<StorageOptions>,
    ) -> DatasetBuilder {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(storage_options) = storage_options {
            builder = builder.with_storage_options(storage_options);
        }
        builder
    }

    fn get_or_init_stream(&mut self) -> Result<&mut DatasetRecordBatchStream, LanceError> {
        if self.stream.is_none() {
            let scanner = self.build_lance_scanner()?;
            let stream = TOKIO_RUNTIME.block_on(scanner.try_into_stream())?;
            self.stream = Some(stream);
        }

        Ok(self
            .stream
            .as_mut()
            .expect("stream should be initialized by get_or_init_stream"))
    }

    fn build_lance_scanner(&self) -> Result<Scanner, LanceError> {
        let mut scanner = self.dataset.scan();

        // Without this a blob column yields a position and size rather than the bytes the
        // dataset's schema advertises. Projection pushdown keeps the cost off queries that do
        // not select the column.
        scanner.blob_handling(BlobHandling::AllBinary);

        if let Some(columns) = self.options.with_columns.as_deref() {
            scanner.project(columns)?;
        }

        if let Some(filter) = self.options.filter.as_deref() {
            scanner.filter(filter)?;
        }

        if let Some(n_rows) = self.options.n_rows {
            scanner.limit(Some(n_rows as i64), None)?;
        }

        if let Some(batch_size) = self.options.batch_size.filter(|batch_size| *batch_size > 0) {
            scanner.batch_size(batch_size);
        }

        Ok(scanner)
    }

    fn next_batch(
        stream: &mut DatasetRecordBatchStream,
    ) -> Result<Option<RecordBatch>, LanceError> {
        TOKIO_RUNTIME.block_on(stream.next()).transpose()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int32Array, StringArray};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use lance::Dataset;
    use polars::prelude::{DataFrame, DataType, NamedFrom, Schema, Series};
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    use super::{LanceReader, LanceScanner, LanceScannerOptions, TOKIO_RUNTIME};

    fn new_scanner(uri: &str, options: LanceScannerOptions) -> LanceScanner {
        LanceReader::open(uri, None).unwrap().scanner(options)
    }

    struct TestDataset {
        uri: String,
        int32_values: Vec<Option<i32>>,
        utf8_values: Vec<&'static str>,
        _temp_dir: TempDir, // Include to keep temp dir alive for duration of fixture.
    }

    #[fixture]
    fn test_dataset() -> TestDataset {
        // Create batches to write to Lance dataset.
        let int32_values = vec![Some(1), None, Some(3)];
        let utf8_values = vec!["a", "b", "c"];
        let batch = RecordBatch::try_from_iter(vec![
            (
                "my_int32_field",
                Arc::new(Int32Array::from(int32_values.clone())) as ArrayRef,
            ),
            (
                "my_utf8_field",
                Arc::new(StringArray::from(utf8_values.clone())) as ArrayRef,
            ),
        ])
        .unwrap();
        let batch_schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), batch_schema);

        // Create temp dir to write Lance dataset to.
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = temp_dir
            .path()
            .join("test_dataset.lance")
            .to_str()
            .unwrap()
            .to_owned();

        // Write Lance dataset to temp dir.
        TOKIO_RUNTIME
            .block_on(Dataset::write(batches, &uri, None))
            .unwrap();

        TestDataset {
            _temp_dir: temp_dir,
            uri,
            int32_values,
            utf8_values,
        }
    }

    #[rstest]
    fn lance_reader_scanner(test_dataset: TestDataset) {
        let options = LanceScannerOptions {
            with_columns: Some(vec!["my_int32_field".to_owned()]),
            filter: Some("my_int32_field > 1".to_owned()),
            n_rows: Some(3),
            batch_size: Some(128),
        };

        let scanner = new_scanner(&test_dataset.uri, options.clone());

        assert_eq!(scanner.options.with_columns, options.with_columns);
        assert_eq!(scanner.options.filter, options.filter);
        assert_eq!(scanner.options.n_rows, options.n_rows);
        assert_eq!(scanner.options.batch_size, options.batch_size);
        assert!(scanner.stream.is_none());
    }

    /// The dataset is opened once, so scanners for the same reader share it.
    #[rstest]
    fn lance_reader_scanners_share_the_dataset(test_dataset: TestDataset) {
        let reader = LanceReader::open(&test_dataset.uri, None).unwrap();

        let first = reader.scanner(LanceScannerOptions::default());
        let second = reader.scanner(LanceScannerOptions::default());

        assert_eq!(
            first.dataset.manifest().version,
            second.dataset.manifest().version
        );
    }

    #[rstest]
    fn lance_scanner_next(test_dataset: TestDataset) {
        fn new_expected_dataframe(int32_values: &[Option<i32>], utf8_values: &[&str]) -> DataFrame {
            DataFrame::new_infer_height(vec![
                Series::new("my_int32_field".into(), int32_values).into(),
                Series::new("my_utf8_field".into(), utf8_values).into(),
            ])
            .unwrap()
        }

        let batch_size = 2;
        let mut scanner = new_scanner(
            &test_dataset.uri,
            LanceScannerOptions {
                batch_size: Some(batch_size),
                ..Default::default()
            },
        );

        // First next().
        let df_0 = scanner.next().unwrap().unwrap();
        assert!(scanner.stream.is_some());
        let expected_dataframe_0 = new_expected_dataframe(
            &test_dataset.int32_values[..batch_size],
            &test_dataset.utf8_values[..batch_size],
        );
        assert_eq!(df_0, expected_dataframe_0);

        // Second next().
        let df_1 = scanner.next().unwrap().unwrap();
        let expected_dataframe_1 = new_expected_dataframe(
            &test_dataset.int32_values[batch_size..],
            &test_dataset.utf8_values[batch_size..],
        );
        assert_eq!(df_1, expected_dataframe_1);

        // Third (and final) next().
        assert_eq!(scanner.next().unwrap(), None);
    }

    /// Asking for no rows yields nothing, without opening a scan. Polars never sends this — it
    /// resolves a zero-row limit itself — so this holds the Rust API's own behaviour, where
    /// `Scanner::limit(0)` would otherwise decide it.
    #[rstest]
    fn lance_scanner_with_zero_rows(test_dataset: TestDataset) {
        let mut scanner = new_scanner(
            &test_dataset.uri,
            LanceScannerOptions {
                n_rows: Some(0),
                ..Default::default()
            },
        );

        assert_eq!(scanner.next().unwrap(), None);
    }

    /// The filter must reach the Lance scanner, so that Lance reads fewer rows.
    #[rstest]
    fn lance_scanner_pushes_filter_down(test_dataset: TestDataset) {
        let scanner = new_scanner(
            &test_dataset.uri,
            LanceScannerOptions {
                filter: Some("my_int32_field > 1".to_owned()),
                ..Default::default()
            },
        );

        let plan = TOKIO_RUNTIME
            .block_on(scanner.build_lance_scanner().unwrap().explain_plan(false))
            .unwrap();

        assert!(
            plan.contains("my_int32_field > Int32(1)"),
            "expected the filter in the Lance plan, got: {plan}"
        );
    }

    /// Without a filter, Lance scans everything.
    #[rstest]
    fn lance_scanner_without_filter(test_dataset: TestDataset) {
        let scanner = new_scanner(&test_dataset.uri, LanceScannerOptions::default());

        let plan = TOKIO_RUNTIME
            .block_on(scanner.build_lance_scanner().unwrap().explain_plan(false))
            .unwrap();

        // Lance renders an absent filter as `--`.
        assert!(
            plan.contains("full_filter=--"),
            "expected no filter in the Lance plan, got: {plan}"
        );
    }

    /// The pushed filter reduces the rows the scanner returns.
    #[rstest]
    fn lance_scanner_next_returns_filtered_rows(test_dataset: TestDataset) {
        let mut scanner = new_scanner(
            &test_dataset.uri,
            LanceScannerOptions {
                filter: Some("my_int32_field > 1".to_owned()),
                ..Default::default()
            },
        );

        let df = scanner.next().unwrap().unwrap();

        let expected_dataframe = DataFrame::new_infer_height(vec![
            Series::new("my_int32_field".into(), [3i32]).into(),
            Series::new("my_utf8_field".into(), ["c"]).into(),
        ])
        .unwrap();
        assert_eq!(df, expected_dataframe);
    }

    /// An invalid filter is reported instead of being silently ignored.
    #[rstest]
    fn lance_scanner_rejects_invalid_filter(test_dataset: TestDataset) {
        let mut scanner = new_scanner(
            &test_dataset.uri,
            LanceScannerOptions {
                filter: Some("my_int32_field >> 1".to_owned()),
                ..Default::default()
            },
        );

        assert!(scanner.next().is_err());
    }

    #[rstest]
    fn lance_reader_schema(test_dataset: TestDataset) {
        let schema = LanceReader::open(&test_dataset.uri, None)
            .unwrap()
            .schema()
            .unwrap();

        let expected_schema = Schema::from_iter([
            ("my_int32_field".into(), DataType::Int32),
            ("my_utf8_field".into(), DataType::String),
        ]);
        assert_eq!(schema, expected_schema);
    }
}
