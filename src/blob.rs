//! Blob columns.
//!
//! Lance stores a large binary value out of line when its field says it is a blob, which keeps
//! the value out of the way until something asks for it. A Polars schema carries neither
//! per-field metadata nor extension types, so a column cannot say this for itself; it is
//! requested by name and applied to the Arrow schema on the way out.
//!
//! Lance describes such a field with the `lance.blob.v2` extension type over
//! `struct<data, uri>`, which records each value's nullability. An older layout using
//! `lance-encoding:blob` metadata exists but cannot keep a null, so it is not written; see
//! `MIN_BLOB_VERSION`.
//!
//! Reading needs no request: a scan asks Lance for the bytes, so the column arrives as plain
//! binary either way. What it does need is for the extension type to be taken back off, since
//! Polars cannot represent one — the dataset's schema always names it, and a filtered scan
//! repeats it on the batch.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, RecordBatch, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    SchemaRef as ArrowSchemaRef,
};
use arrow::error::ArrowError;

/// Arrow field metadata key naming an extension type.
const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// Arrow field metadata key holding an extension type's parameters. Lance writes it empty.
const ARROW_EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";

/// Extension type Lance uses for a blob column from version 2.2.
const BLOB_V2_EXTENSION_NAME: &str = "lance.blob.v2";

/// The struct behind the blob v2 extension type.
///
/// Lance also accepts `position` and `size` fields, which reference bytes held outside the
/// dataset. A write from a dataframe always carries its own bytes, so they are left out.
fn blob_v2_storage_fields() -> ArrowFields {
    ArrowFields::from(vec![
        ArrowField::new("data", ArrowDataType::LargeBinary, true),
        ArrowField::new("uri", ArrowDataType::Utf8, true),
    ])
}

fn is_blob_v2(field: &ArrowField) -> bool {
    field
        .metadata()
        .get(ARROW_EXTENSION_NAME_KEY)
        .map(String::as_str)
        == Some(BLOB_V2_EXTENSION_NAME)
}

/// Wrap the blob v2 columns of `batch` in the struct Lance expects, and give the batch the
/// schema being written.
///
/// Columns that are not blob v2 are passed through, so this also covers the legacy layout,
/// where only the schema differs. Re-stating the schema matters on its own: a batch built from
/// one morsel can call a column non-nullable just because that morsel holds no nulls.
///
/// Columns are matched to fields by name. Pairing them positionally would wrap whichever
/// column happened to sit at the blob field's index, which for two columns of the same type
/// swaps their contents without any error being raised.
pub fn wrap_blob_v2_columns(
    batch: &RecordBatch,
    schema: &ArrowSchemaRef,
) -> Result<RecordBatch, ArrowError> {
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let column = batch.column_by_name(field.name()).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "the batch is missing column {:?}, which the schema being written declares",
                    field.name()
                ))
            })?;

            if !is_blob_v2(field) {
                return Ok(Arc::clone(column));
            }

            let fields = blob_v2_storage_fields();
            // The blob's own validity becomes the struct's, so a null blob stays null rather
            // than becoming a present blob holding no bytes.
            let validity = column.nulls().cloned();
            let data = cast(column.as_ref(), &ArrowDataType::LargeBinary)?;
            let uri = new_null_array(&ArrowDataType::Utf8, column.len());

            Ok(Arc::new(StructArray::try_new(fields, vec![data, uri], validity)?) as _)
        })
        .collect::<Result<Vec<_>, ArrowError>>()?;

    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// Strip the blob v2 extension type from a batch a scan returned.
///
/// The bytes come back as a plain `LargeBinary` column, but the field describing it still names
/// the extension type on some scans (a filtered one does, an unfiltered one does not). Polars
/// cannot build a series from an extension type at all, so the name has to go before the batch
/// crosses over.
pub fn unwrap_blob_v2_batch(batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
    if !batch
        .schema()
        .fields()
        .iter()
        .any(|field| is_blob_v2(field))
    {
        return Ok(batch);
    }

    let schema = Arc::new(unwrap_blob_v2_fields(batch.schema().as_ref().clone()));
    RecordBatch::try_new(schema, batch.columns().to_vec())
}

/// Present a blob v2 column as the binary column a scan actually returns.
///
/// A scan asks Lance for the bytes, so it hands back a `LargeBinary` column even though the
/// dataset's schema describes the field as the extension type. Polars cannot represent an
/// extension type at all, so the schema has to describe what the scan delivers.
pub fn unwrap_blob_v2_fields(schema: ArrowSchema) -> ArrowSchema {
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if !is_blob_v2(field) {
                return Arc::clone(field);
            }

            let mut metadata = field.metadata().clone();
            metadata.remove(ARROW_EXTENSION_NAME_KEY);
            metadata.remove(ARROW_EXTENSION_METADATA_KEY);

            Arc::new(
                ArrowField::new(
                    field.name(),
                    ArrowDataType::LargeBinary,
                    field.is_nullable(),
                )
                .with_metadata(metadata),
            )
        })
        .collect::<Vec<_>>();

    ArrowSchema::new(fields).with_metadata(schema.metadata().clone())
}

/// Describe the named columns as blob columns, so Lance stores them out of line.
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
            metadata.insert(
                ARROW_EXTENSION_NAME_KEY.to_owned(),
                BLOB_V2_EXTENSION_NAME.to_owned(),
            );

            Ok(ArrowField::new(
                field.name(),
                ArrowDataType::Struct(blob_v2_storage_fields()),
                field.is_nullable(),
            )
            .with_metadata(metadata)
            .into())
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

    use super::{
        blob_v2_storage_fields, mark_blob_columns, unwrap_blob_v2_fields, wrap_blob_v2_columns,
        ARROW_EXTENSION_NAME_KEY, BLOB_V2_EXTENSION_NAME,
    };

    fn schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("blob", ArrowDataType::LargeBinary, true),
        ])
    }

    /// From 2.2 the column is described by an extension type over a struct, not by metadata on
    /// a binary column.
    #[test]
    fn marks_a_blob_v2_column_with_the_extension_type() {
        let marked = mark_blob_columns(schema(), &["blob".to_owned()]).unwrap();

        let field = marked.field_with_name("blob").unwrap();
        assert_eq!(
            field.metadata().get(ARROW_EXTENSION_NAME_KEY),
            Some(&BLOB_V2_EXTENSION_NAME.to_owned())
        );
        assert_eq!(
            field.data_type(),
            &ArrowDataType::Struct(blob_v2_storage_fields())
        );
        assert_eq!(marked.field_with_name("id").unwrap().metadata().len(), 0);
    }

    /// A scan returns the bytes, so the schema has to say binary however the dataset describes
    /// the field.
    #[test]
    fn unwraps_a_blob_v2_column_back_to_binary() {
        let marked = mark_blob_columns(schema(), &["blob".to_owned()]).unwrap();

        let unwrapped = unwrap_blob_v2_fields(marked);

        let field = unwrapped.field_with_name("blob").unwrap();
        assert_eq!(field.data_type(), &ArrowDataType::LargeBinary);
        assert!(field.metadata().get(ARROW_EXTENSION_NAME_KEY).is_none());
        assert!(field.is_nullable());
    }

    /// A null blob has to stay null once wrapped, rather than becoming a present value holding
    /// no bytes.
    #[test]
    fn wrapping_a_blob_v2_column_keeps_its_nulls() {
        let schema = Arc::new(mark_blob_columns(schema(), &["blob".to_owned()]).unwrap());
        let batch = RecordBatch::try_new(
            Arc::new(super::ArrowSchema::new(vec![
                ArrowField::new("id", ArrowDataType::Int64, false),
                ArrowField::new("blob", ArrowDataType::LargeBinary, true),
            ])),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
                Arc::new(LargeBinaryArray::from(vec![Some(&b"a"[..]), None])),
            ],
        )
        .unwrap();

        let wrapped = wrap_blob_v2_columns(&batch, &schema).unwrap();

        let column = wrapped.column_by_name("blob").unwrap();
        assert_eq!(
            column.data_type(),
            &ArrowDataType::Struct(blob_v2_storage_fields())
        );
        assert_eq!(column.null_count(), 1);
        assert!(column.is_null(1));
    }

    /// Columns are paired to fields by name, not by position.
    ///
    /// The streaming writer hands over a batch built from the query's own schema, which is
    /// only incidentally in the same order as the schema being written. With two columns of
    /// the same type, pairing by position wraps the wrong one and swaps their contents, and
    /// every downstream check still passes because the types line up.
    #[test]
    fn wraps_the_named_column_when_the_batch_is_in_another_order() {
        let schema = Arc::new(
            mark_blob_columns(
                ArrowSchema::new(vec![
                    ArrowField::new("blob", ArrowDataType::LargeBinary, true),
                    ArrowField::new("other", ArrowDataType::LargeBinary, true),
                ]),
                &["blob".to_owned()],
            )
            .unwrap(),
        );

        // Reversed relative to the schema, and both columns are binary.
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("other", ArrowDataType::LargeBinary, true),
                ArrowField::new("blob", ArrowDataType::LargeBinary, true),
            ])),
            vec![
                Arc::new(LargeBinaryArray::from(vec![Some(&b"other"[..])])),
                Arc::new(LargeBinaryArray::from(vec![Some(&b"blob"[..])])),
            ],
        )
        .unwrap();

        let wrapped = wrap_blob_v2_columns(&batch, &schema).unwrap();

        let blob = wrapped.column_by_name("blob").unwrap();
        let data = blob
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .expect("the blob column should have been wrapped in the extension struct")
            .column_by_name("data")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap()
            .value(0)
            .to_vec();
        assert_eq!(data, b"blob".to_vec(), "wrapped the wrong column");

        let other = wrapped.column_by_name("other").unwrap();
        assert_eq!(other.data_type(), &ArrowDataType::LargeBinary);
        assert_eq!(
            other
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .value(0),
            b"other"
        );
    }

    /// A field with no matching column is reported rather than silently dropped, which is
    /// what zipping the two sequences used to do.
    #[test]
    fn refuses_a_batch_missing_a_column_the_schema_declares() {
        let schema = Arc::new(schema());
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1]))],
        )
        .unwrap();

        let error = wrap_blob_v2_columns(&batch, &schema).unwrap_err();

        assert!(error.to_string().contains("missing column"), "{error}");
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
