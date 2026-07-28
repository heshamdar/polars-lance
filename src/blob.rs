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

/// Field metadata key Lance uses to mark a blob column.
const BLOB_META_KEY: &str = "lance-encoding:blob";

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
    use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    use super::{mark_blob_columns, BLOB_META_KEY};

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
