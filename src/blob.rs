//! Blob columns.
//!
//! Lance stores a large binary value out of line when its field is marked with
//! `lance-encoding:blob`, which keeps the value out of the way until something asks for it.
//! The marker lives in Arrow *field metadata*, and Polars carries no per-field metadata, so a
//! column cannot say for itself that it is a blob. Writing one is therefore requested by name.
//!
//! Reading needs no marker: a scan asks Lance for the bytes, so a blob column arrives as the
//! binary column the dataset's schema already advertises. Without that request Lance returns a
//! position and size instead, which would not match the schema.

use std::collections::HashSet;

use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

/// Field metadata key Lance uses to mark a blob column.
const BLOB_META_KEY: &str = "lance-encoding:blob";

/// Refuses a write whose batches disagree about whether a blob column holds nulls.
///
/// Lance's blob encoder derives how it interprets nullability from each batch's data rather than
/// from the field, and then assumes every later batch agrees
/// (<https://github.com/lance-format/lance/issues/8033>). When they disagree it trips a debug
/// assertion in a debug build, and in a release build the outcome is luck: one arrangement
/// writes the nulls, another writes them as empty values. Since nothing downstream can tell
/// those apart, the write is refused rather than left to chance.
///
/// Lance decides by null count, so this does too, which keeps the two in step.
pub struct BlobNullability {
    /// Column name, paired with whether the first batch holding it had any nulls.
    columns: Vec<(String, Option<bool>)>,
}

impl BlobNullability {
    pub fn new(blob_columns: &[String]) -> Self {
        Self {
            columns: blob_columns
                .iter()
                .map(|name| (name.clone(), None))
                .collect(),
        }
    }

    /// Record this batch, failing if it disagrees with one already seen.
    pub fn check(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        for (name, seen) in &mut self.columns {
            let Some(column) = batch.column_by_name(name) else {
                continue;
            };

            let holds_nulls = column.null_count() > 0;
            match seen {
                None => *seen = Some(holds_nulls),
                Some(first) if *first != holds_nulls => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "blob column {name:?} has nulls in some batches of this write but not \
                         others, which Lance cannot encode consistently: it may store those \
                         nulls as empty values. Write the frame with `write_lance`, or choose a \
                         `chunk_size` that keeps the column's nulls from splitting across batches."
                    )));
                }
                Some(_) => {}
            }
        }

        Ok(())
    }
}

/// Mark the named columns so Lance stores them out of line.
///
/// A name that is missing, or a column that is not binary, is refused rather than written as an
/// ordinary column, which would quietly ignore what the caller asked for.
pub fn mark_blob_columns(
    schema: ArrowSchema,
    blob_columns: &[String],
) -> Result<ArrowSchema, ArrowError> {
    if blob_columns.is_empty() {
        return Ok(schema);
    }

    let requested: HashSet<&str> = blob_columns.iter().map(String::as_str).collect();

    let unknown = requested
        .iter()
        .filter(|name| schema.field_with_name(name).is_err())
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "blob_columns names a column that is not in the frame: {unknown:?}"
        )));
    }

    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if !requested.contains(field.name().as_str()) {
                return Ok(field.clone());
            }

            if !matches!(
                field.data_type(),
                ArrowDataType::Binary | ArrowDataType::LargeBinary
            ) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "column {:?} cannot be a blob column because its type is {:?}; \
                     a blob column has to be binary",
                    field.name(),
                    field.data_type()
                )));
            }

            let mut metadata = field.metadata().clone();
            metadata.insert(BLOB_META_KEY.to_owned(), "true".to_owned());
            Ok(ArrowField::clone(field).with_metadata(metadata).into())
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ArrowSchema::new(fields).with_metadata(schema.metadata().clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::LargeBinaryArray;
    use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;

    use super::{mark_blob_columns, BlobNullability, BLOB_META_KEY};

    fn blob_batch(values: Vec<Option<&[u8]>>) -> RecordBatch {
        let schema = ArrowSchema::new(vec![ArrowField::new(
            "blob",
            ArrowDataType::LargeBinary,
            true,
        )]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(LargeBinaryArray::from(values))],
        )
        .unwrap()
    }

    #[test]
    fn blob_nullability_accepts_batches_that_agree() {
        for batches in [
            vec![vec![Some(&b"a"[..])], vec![Some(&b"b"[..])]],
            vec![vec![None], vec![None]],
            vec![vec![Some(&b"a"[..]), None], vec![None, Some(&b"b"[..])]],
        ] {
            let mut nullability = BlobNullability::new(&["blob".to_owned()]);

            for batch in batches {
                nullability.check(&blob_batch(batch)).unwrap();
            }
        }
    }

    #[test]
    fn blob_nullability_refuses_batches_that_disagree() {
        let mut nullability = BlobNullability::new(&["blob".to_owned()]);
        nullability
            .check(&blob_batch(vec![Some(&b"a"[..])]))
            .unwrap();

        let error = nullability.check(&blob_batch(vec![None])).unwrap_err();

        assert!(
            error.to_string().contains("has nulls in some batches"),
            "{error}"
        );
    }

    /// A column that was not asked to be a blob is Lance's ordinary encoder's problem, which
    /// handles a mix of batches.
    #[test]
    fn blob_nullability_ignores_columns_that_are_not_blobs() {
        let mut nullability = BlobNullability::new(&[]);

        nullability
            .check(&blob_batch(vec![Some(&b"a"[..])]))
            .unwrap();
        nullability.check(&blob_batch(vec![None])).unwrap();
    }

    fn schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("blob", ArrowDataType::LargeBinary, true),
        ])
    }

    #[test]
    fn marks_only_the_named_column() {
        let marked = mark_blob_columns(schema(), &["blob".to_owned()]).unwrap();

        assert_eq!(
            marked
                .field_with_name("blob")
                .unwrap()
                .metadata()
                .get(BLOB_META_KEY),
            Some(&"true".to_owned())
        );
        assert!(marked
            .field_with_name("id")
            .unwrap()
            .metadata()
            .get(BLOB_META_KEY)
            .is_none());
    }

    #[test]
    fn leaves_the_schema_alone_when_nothing_is_requested() {
        let marked = mark_blob_columns(schema(), &[]).unwrap();

        assert_eq!(marked, schema());
    }

    /// Naming a column that is not binary is a mistake worth reporting, because Lance would
    /// otherwise store it as an ordinary column.
    #[test]
    fn refuses_a_column_that_is_not_binary() {
        let error = mark_blob_columns(schema(), &["id".to_owned()]).unwrap_err();

        assert!(error.to_string().contains("has to be binary"), "{error}");
    }

    #[test]
    fn refuses_a_column_that_does_not_exist() {
        let error = mark_blob_columns(schema(), &["missing".to_owned()]).unwrap_err();

        assert!(error.to_string().contains("not in the frame"), "{error}");
    }
}
