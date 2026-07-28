use std::error::Error;
use std::fmt;
use std::mem::transmute;
use std::sync::Arc;

use arrow::array::{
    make_array, ArrayData as ArrowArrayData, ArrayRef as ArrowArrayRef,
    FixedSizeListArray as ArrowFixedSizeListArray, LargeListArray as ArrowLargeListArray,
    NullArray as ArrowNullArray, StructArray as ArrowStructArray,
};
use arrow::buffer::{NullBuffer as ArrowNullBuffer, OffsetBuffer as ArrowOffsetBuffer};
use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use arrow::error::ArrowError;
use arrow::ffi::{
    from_ffi_and_data_type, FFI_ArrowArray as CArrowArray, FFI_ArrowSchema as CArrowSchema,
};
use arrow::record_batch::RecordBatch as ArrowRecordBatch;
use polars::prelude::{
    ArrayRef as PolarsArrowArrayRef, ArrowDataType as PolarsArrowDataType,
    ArrowField as PolarsArrowField, ArrowSchema as PolarsArrowSchema, PolarsError,
};
use polars_core::utils::arrow::{
    array::{
        Array as PolarsArrowArray, FixedSizeListArray as PolarsArrowFixedSizeListArray,
        ListArray as PolarsArrowListArray, NullArray as PolarsArrowNullArray,
        StructArray as PolarsArrowStructArray,
    },
    bitmap::Bitmap as PolarsArrowBitmap,
    ffi::{
        export_array_to_c, export_field_to_c, import_array_from_c, import_field_from_c,
        ArrowArray as PolarsCArrowArray, ArrowSchema as PolarsCArrowSchema,
    },
    offset::OffsetsBuffer as PolarsArrowOffsetsBuffer,
    record_batch::RecordBatch as PolarsArrowRecordBatch,
};

pub(crate) type ArrowBridgeResult<T> = Result<T, ArrowBridgeError>;

#[derive(Debug)]
pub(crate) enum ArrowBridgeError {
    Arrow(ArrowError),
    Polars(PolarsError),
}

impl fmt::Display for ArrowBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arrow(err) => err.fmt(f),
            Self::Polars(err) => err.fmt(f),
        }
    }
}

impl Error for ArrowBridgeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Arrow(err) => Some(err),
            Self::Polars(err) => Some(err),
        }
    }
}

impl From<ArrowError> for ArrowBridgeError {
    fn from(err: ArrowError) -> Self {
        Self::Arrow(err)
    }
}

impl From<PolarsError> for ArrowBridgeError {
    fn from(err: PolarsError) -> Self {
        Self::Polars(err)
    }
}

trait AsPolarsCArrowSchema {
    unsafe fn as_polars_c_arrow_schema(&self) -> &PolarsCArrowSchema;
}

impl AsPolarsCArrowSchema for CArrowSchema {
    unsafe fn as_polars_c_arrow_schema(&self) -> &PolarsCArrowSchema {
        &*(self as *const CArrowSchema).cast::<PolarsCArrowSchema>()
    }
}

trait AsCArrowSchema {
    unsafe fn as_c_arrow_schema(&self) -> &CArrowSchema;
}

impl AsCArrowSchema for PolarsCArrowSchema {
    unsafe fn as_c_arrow_schema(&self) -> &CArrowSchema {
        &*(self as *const PolarsCArrowSchema).cast::<CArrowSchema>()
    }
}

trait IntoPolarsCArrowArray {
    unsafe fn into_polars_c_arrow_array(self) -> PolarsCArrowArray;
}

impl IntoPolarsCArrowArray for CArrowArray {
    unsafe fn into_polars_c_arrow_array(self) -> PolarsCArrowArray {
        transmute::<CArrowArray, PolarsCArrowArray>(self)
    }
}

trait IntoCArrowArray {
    unsafe fn into_c_arrow_array(self) -> CArrowArray;
}

impl IntoCArrowArray for PolarsCArrowArray {
    unsafe fn into_c_arrow_array(self) -> CArrowArray {
        transmute::<PolarsCArrowArray, CArrowArray>(self)
    }
}

pub(crate) trait ArrowDataTypeExt {
    fn to_polars_arrow_data_type(&self) -> ArrowBridgeResult<PolarsArrowDataType>;
}

pub(crate) trait ArrowFieldExt {
    fn to_polars_arrow_field(&self) -> ArrowBridgeResult<PolarsArrowField>;
}

pub(crate) trait ArrowSchemaExt {
    fn to_polars_arrow_schema(&self) -> ArrowBridgeResult<PolarsArrowSchema>;
}

pub(crate) trait ArrowArrayRefExt {
    fn to_polars_arrow_array_ref(&self) -> ArrowBridgeResult<PolarsArrowArrayRef>;
}

pub(crate) trait ArrowRecordBatchExt {
    fn to_polars_arrow_record_batch(&self) -> ArrowBridgeResult<PolarsArrowRecordBatch>;
}

pub(crate) trait PolarsArrowDataTypeExt {
    fn to_arrow_data_type(&self) -> ArrowBridgeResult<ArrowDataType>;
}

pub(crate) trait PolarsArrowFieldExt {
    fn to_arrow_field(&self) -> ArrowBridgeResult<ArrowField>;
}

pub(crate) trait PolarsArrowSchemaExt {
    fn to_arrow_schema(&self) -> ArrowBridgeResult<ArrowSchema>;
}

pub(crate) trait PolarsArrowArrayRefExt {
    fn to_arrow_array_ref(&self) -> ArrowBridgeResult<ArrowArrayRef>;
}

pub(crate) trait PolarsArrowRecordBatchExt {
    fn to_arrow_record_batch(&self) -> ArrowBridgeResult<ArrowRecordBatch>;
}

fn import_polars_arrow_field(c_arrow_schema: &CArrowSchema) -> ArrowBridgeResult<PolarsArrowField> {
    unsafe {
        let polars_c_arrow_schema = c_arrow_schema.as_polars_c_arrow_schema();
        import_field_from_c(polars_c_arrow_schema)
    }
    .map_err(Into::into)
}

impl ArrowDataTypeExt for ArrowDataType {
    fn to_polars_arrow_data_type(&self) -> ArrowBridgeResult<PolarsArrowDataType> {
        let c_arrow_schema = CArrowSchema::try_from(self)?;
        import_polars_arrow_field(&c_arrow_schema).map(|field| field.dtype().clone())
    }
}

impl ArrowFieldExt for ArrowField {
    fn to_polars_arrow_field(&self) -> ArrowBridgeResult<PolarsArrowField> {
        // Convert from Arrow to Polars Arrow via Arrow C data interface.
        let c_arrow_schema = CArrowSchema::try_from(self)?;
        import_polars_arrow_field(&c_arrow_schema)
    }
}

impl ArrowSchemaExt for ArrowSchema {
    fn to_polars_arrow_schema(&self) -> ArrowBridgeResult<PolarsArrowSchema> {
        self.fields
            .iter()
            .map(|field_ref| field_ref.to_polars_arrow_field())
            .collect()
    }
}

impl ArrowArrayRefExt for ArrowArrayRef {
    fn to_polars_arrow_array_ref(&self) -> ArrowBridgeResult<PolarsArrowArrayRef> {
        // Convert from Arrow to Polars Arrow via Arrow C data interface.
        let c_arrow_array = CArrowArray::new(&self.to_data());
        let polars_arrow_data_type = self.data_type().to_polars_arrow_data_type()?;

        if contains_null_dtype(&polars_arrow_data_type) {
            return self.to_polars_arrow_array_ref_without_ffi(&polars_arrow_data_type);
        }

        unsafe {
            let polars_c_arrow_array = c_arrow_array.into_polars_c_arrow_array();
            import_array_from_c(polars_c_arrow_array, polars_arrow_data_type)
        }
        .map_err(Into::into)
    }
}

trait ArrowArrayRefWithoutFfiExt: ArrowArrayRefExt {
    /// Rebuild an array that holds a null type, whose children are converted one by one.
    fn to_polars_arrow_array_ref_without_ffi(
        &self,
        dtype: &PolarsArrowDataType,
    ) -> ArrowBridgeResult<PolarsArrowArrayRef>;

    /// Convert a child, using the C data interface unless it holds a null type.
    fn to_polars_arrow_array_ref_for(
        &self,
        dtype: &PolarsArrowDataType,
    ) -> ArrowBridgeResult<PolarsArrowArrayRef> {
        if contains_null_dtype(dtype) {
            self.to_polars_arrow_array_ref_without_ffi(dtype)
        } else {
            self.to_polars_arrow_array_ref()
        }
    }
}

impl ArrowArrayRefWithoutFfiExt for ArrowArrayRef {
    fn to_polars_arrow_array_ref_without_ffi(
        &self,
        dtype: &PolarsArrowDataType,
    ) -> ArrowBridgeResult<PolarsArrowArrayRef> {
        let unsupported = || {
            ArrowBridgeError::Arrow(ArrowError::NotYetImplemented(format!(
                "converting a {:?} array holding a null type is not supported",
                self.data_type()
            )))
        };

        if matches!(dtype, PolarsArrowDataType::Null) {
            return Ok(Box::new(PolarsArrowNullArray::new(
                dtype.clone(),
                self.len(),
            )));
        }

        let validity = self.nulls().map(|nulls| nulls.iter().collect());

        match dtype {
            PolarsArrowDataType::Struct(fields) => {
                let struct_array = self
                    .as_any()
                    .downcast_ref::<ArrowStructArray>()
                    .ok_or_else(unsupported)?;

                let values = fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        struct_array
                            .column(index)
                            .to_polars_arrow_array_ref_for(field.dtype())
                    })
                    .collect::<ArrowBridgeResult<Vec<_>>>()?;

                Ok(Box::new(PolarsArrowStructArray::new(
                    dtype.clone(),
                    self.len(),
                    values,
                    validity,
                )))
            }
            PolarsArrowDataType::LargeList(field) => {
                let list_array = self
                    .as_any()
                    .downcast_ref::<ArrowLargeListArray>()
                    .ok_or_else(unsupported)?;

                let offsets = PolarsArrowOffsetsBuffer::try_from(
                    list_array.offsets().iter().copied().collect::<Vec<_>>(),
                )?;

                Ok(Box::new(PolarsArrowListArray::<i64>::new(
                    dtype.clone(),
                    offsets,
                    list_array
                        .values()
                        .to_polars_arrow_array_ref_for(field.dtype())?,
                    validity,
                )))
            }
            PolarsArrowDataType::FixedSizeList(field, _) => {
                let list_array = self
                    .as_any()
                    .downcast_ref::<ArrowFixedSizeListArray>()
                    .ok_or_else(unsupported)?;

                Ok(Box::new(PolarsArrowFixedSizeListArray::new(
                    dtype.clone(),
                    self.len(),
                    list_array
                        .values()
                        .to_polars_arrow_array_ref_for(field.dtype())?,
                    validity,
                )))
            }
            _ => Err(unsupported()),
        }
    }
}

impl ArrowRecordBatchExt for ArrowRecordBatch {
    fn to_polars_arrow_record_batch(&self) -> ArrowBridgeResult<PolarsArrowRecordBatch> {
        let schema = self.schema().as_ref().to_polars_arrow_schema()?;

        let arrays = self
            .columns()
            .iter()
            .map(ArrowArrayRefExt::to_polars_arrow_array_ref)
            .collect::<ArrowBridgeResult<Vec<_>>>()?;

        PolarsArrowRecordBatch::try_new(self.num_rows(), Arc::new(schema), arrays)
            .map_err(Into::into)
    }
}

impl PolarsArrowDataTypeExt for PolarsArrowDataType {
    fn to_arrow_data_type(&self) -> ArrowBridgeResult<ArrowDataType> {
        PolarsArrowField::new("dummy".into(), self.clone(), true)
            .to_arrow_field()
            .map(|field| field.data_type().clone())
    }
}

impl PolarsArrowFieldExt for PolarsArrowField {
    fn to_arrow_field(&self) -> ArrowBridgeResult<ArrowField> {
        // Convert from Polars Arrow to Arrow via Arrow C data interface.
        let polars_c_arrow_schema = export_field_to_c(self);
        let c_arrow_schema = unsafe { polars_c_arrow_schema.as_c_arrow_schema() };
        ArrowField::try_from(c_arrow_schema).map_err(Into::into)
    }
}

impl PolarsArrowSchemaExt for PolarsArrowSchema {
    fn to_arrow_schema(&self) -> ArrowBridgeResult<ArrowSchema> {
        self.iter_values()
            .map(PolarsArrowFieldExt::to_arrow_field)
            .collect::<ArrowBridgeResult<Vec<_>>>()
            .map(ArrowSchema::new)
    }
}

/// Whether a struct in `data` reports an offset that its children cannot be sliced by.
///
/// Slicing a struct array in Polars slices its children but leaves the validity bitmap
/// pointing into the original buffer at an offset. Exporting that over the C data interface
/// reports the offset on the struct even though the children are already sliced, so Arrow
/// applies it a second time and reads past the end of a child. Polars documents the
/// discrepancy in `polars_arrow::array::struct_::ffi`; only structs are affected.
///
/// This checks the condition Arrow itself asserts on, so an array is only copied to remove
/// the offset when it would otherwise panic.
fn struct_offset_exceeds_children(data: &ArrowArrayData) -> bool {
    let exceeds_own_children = matches!(data.data_type(), ArrowDataType::Struct(_))
        && data
            .child_data()
            .iter()
            .any(|child| data.offset() + data.len() > child.len());

    exceeds_own_children || data.child_data().iter().any(struct_offset_exceeds_children)
}

/// Rebuild the validity of every struct in `array` so that none of them carries an offset.
fn compact_struct_validity(array: &dyn PolarsArrowArray) -> Box<dyn PolarsArrowArray> {
    let Some(struct_array) = array.as_any().downcast_ref::<PolarsArrowStructArray>() else {
        return array.to_boxed();
    };

    let values = struct_array
        .values()
        .iter()
        .map(|child| compact_struct_validity(child.as_ref()))
        .collect();

    // Collecting the bits into a new bitmap starts it at offset zero.
    let validity = struct_array
        .validity()
        .map(|validity| validity.iter().collect::<PolarsArrowBitmap>());

    Box::new(PolarsArrowStructArray::new(
        struct_array.dtype().clone(),
        struct_array.len(),
        values,
        validity,
    ))
}

/// Whether `dtype` is, or contains, the null type — the one shape the C data interface cannot
/// carry, so an array of this type is rebuilt child by child instead.
///
/// Polars exports a null array with one (null) buffer for compatibility with older C++
/// implementations, while Arrow rejects any buffer for that type. A null array on its own is
/// built directly; one nested inside a struct or list has to be reached by rebuilding the array
/// that holds it.
fn contains_null_dtype(dtype: &PolarsArrowDataType) -> bool {
    match dtype {
        PolarsArrowDataType::Null => true,
        PolarsArrowDataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_null_dtype(field.dtype())),
        PolarsArrowDataType::List(field)
        | PolarsArrowDataType::LargeList(field)
        | PolarsArrowDataType::FixedSizeList(field, _) => contains_null_dtype(field.dtype()),
        _ => false,
    }
}

/// Convert a Polars validity bitmap into Arrow's representation.
fn to_arrow_nulls(validity: Option<&PolarsArrowBitmap>) -> Option<ArrowNullBuffer> {
    validity.map(|validity| validity.iter().collect())
}

impl PolarsArrowArrayRefExt for PolarsArrowArrayRef {
    fn to_arrow_array_ref(&self) -> ArrowBridgeResult<ArrowArrayRef> {
        if matches!(self.dtype(), PolarsArrowDataType::Null) {
            return Ok(Arc::new(ArrowNullArray::new(self.len())));
        }

        // A nested null type cannot cross the C data interface, so the array holding it is
        // rebuilt from children that are converted one by one.
        if contains_null_dtype(self.dtype()) {
            return self.to_arrow_array_ref_without_ffi();
        }

        let array_data = self.as_ref().to_boxed().to_arrow_array_data()?;

        if !struct_offset_exceeds_children(&array_data) {
            return Ok(make_array(array_data));
        }

        // Rebuilding the validity drops the offset, so the export is then well formed.
        let array_data = compact_struct_validity(self.as_ref()).to_arrow_array_data()?;
        if struct_offset_exceeds_children(&array_data) {
            return Err(ArrowBridgeError::Arrow(ArrowError::InvalidArgumentError(
                format!(
                    "cannot convert a sliced {:?} array: a struct inside it reports an offset \
                     its children cannot be sliced by",
                    self.dtype()
                ),
            )));
        }

        Ok(make_array(array_data))
    }
}

trait PolarsArrowArrayRefWithoutFfiExt {
    /// Rebuild an array that holds a null type, whose children are converted one by one.
    fn to_arrow_array_ref_without_ffi(&self) -> ArrowBridgeResult<ArrowArrayRef>;
}

impl PolarsArrowArrayRefWithoutFfiExt for PolarsArrowArrayRef {
    fn to_arrow_array_ref_without_ffi(&self) -> ArrowBridgeResult<ArrowArrayRef> {
        let unsupported = || {
            ArrowBridgeError::Arrow(ArrowError::NotYetImplemented(format!(
                "converting a {:?} array holding a null type is not supported",
                self.dtype()
            )))
        };

        let field = match self.dtype().to_arrow_data_type()? {
            ArrowDataType::Struct(fields) => {
                let struct_array = self
                    .as_any()
                    .downcast_ref::<PolarsArrowStructArray>()
                    .ok_or_else(unsupported)?;

                let children = struct_array
                    .values()
                    .iter()
                    .map(PolarsArrowArrayRefExt::to_arrow_array_ref)
                    .collect::<ArrowBridgeResult<Vec<_>>>()?;

                return Ok(Arc::new(ArrowStructArray::try_new_with_length(
                    fields,
                    children,
                    to_arrow_nulls(struct_array.validity()),
                    struct_array.len(),
                )?));
            }
            ArrowDataType::LargeList(field) => field,
            ArrowDataType::FixedSizeList(field, size) => {
                let list_array = self
                    .as_any()
                    .downcast_ref::<PolarsArrowFixedSizeListArray>()
                    .ok_or_else(unsupported)?;

                return Ok(Arc::new(ArrowFixedSizeListArray::try_new(
                    field,
                    size,
                    list_array.values().to_boxed().to_arrow_array_ref()?,
                    to_arrow_nulls(list_array.validity()),
                )?));
            }
            _ => return Err(unsupported()),
        };

        let list_array = self
            .as_any()
            .downcast_ref::<PolarsArrowListArray<i64>>()
            .ok_or_else(unsupported)?;

        let offsets = ArrowOffsetBuffer::new(list_array.offsets().iter().copied().collect());

        Ok(Arc::new(ArrowLargeListArray::try_new(
            field,
            offsets,
            list_array.values().to_boxed().to_arrow_array_ref()?,
            to_arrow_nulls(list_array.validity()),
        )?))
    }
}

trait PolarsArrowArrayBoxExt {
    /// Convert from Polars Arrow to Arrow via the Arrow C data interface.
    fn to_arrow_array_data(self) -> ArrowBridgeResult<ArrowArrayData>;
}

impl PolarsArrowArrayBoxExt for Box<dyn PolarsArrowArray> {
    fn to_arrow_array_data(self) -> ArrowBridgeResult<ArrowArrayData> {
        let arrow_data_type = self.dtype().to_arrow_data_type()?;
        let polars_c_arrow_array = export_array_to_c(self);

        unsafe {
            let c_arrow_array = polars_c_arrow_array.into_c_arrow_array();
            from_ffi_and_data_type(c_arrow_array, arrow_data_type)
        }
        .map_err(Into::into)
    }
}

impl PolarsArrowRecordBatchExt for PolarsArrowRecordBatch {
    fn to_arrow_record_batch(&self) -> ArrowBridgeResult<ArrowRecordBatch> {
        let schema = self.schema().to_arrow_schema()?;
        let arrays = self
            .columns()
            .iter()
            .map(|array_ref| array_ref.to_arrow_array_ref())
            .collect::<ArrowBridgeResult<Vec<_>>>()?;

        ArrowRecordBatch::try_new(Arc::new(schema), arrays).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        BooleanArray as ArrowBooleanArray, Int32Array as ArrowInt32Array,
        StringArray as ArrowStringArray, StructArray as ArrowStructArray,
    };
    use arrow::datatypes::DataType as ArrowDataType;
    use polars::prelude::ArrowDataType as PolarsArrowDataType;
    use polars_core::utils::arrow::array::{
        BooleanArray as PolarsBooleanArray, Int32Array as PolarsInt32Array,
        StructArray as PolarsStructArray, Utf8Array as PolarsUtf8Array,
    };

    use super::{
        ArrowArrayRef, ArrowArrayRefExt, ArrowDataTypeExt, ArrowField, ArrowFieldExt,
        ArrowRecordBatch, ArrowRecordBatchExt, ArrowSchema, ArrowSchemaExt, PolarsArrowArrayRefExt,
    };

    fn nullable_struct_array() -> PolarsStructArray {
        let dtype = PolarsArrowDataType::Struct(vec![super::PolarsArrowField::new(
            "x".into(),
            PolarsArrowDataType::Int32,
            true,
        )]);
        PolarsStructArray::new(
            dtype,
            3,
            vec![Box::new(PolarsInt32Array::from(vec![Some(1), Some(2), Some(3)])) as _],
            Some([true, false, true].into_iter().collect()),
        )
    }

    /// Slicing a struct array leaves its validity at an offset, which Polars reports over the
    /// C data interface even though the children are already sliced. Converting such an array
    /// used to panic inside Arrow.
    #[test]
    fn to_arrow_array_ref_converts_a_sliced_struct_with_nulls() {
        let array = nullable_struct_array();

        for (offset, length) in [(0, 3), (1, 1), (1, 2), (2, 1)] {
            let sliced: super::PolarsArrowArrayRef = array.clone().sliced(offset, length).boxed();

            let converted = sliced
                .to_arrow_array_ref()
                .expect("sliced struct should convert");

            assert_eq!(converted.len(), length);
            assert_eq!(
                converted.null_count(),
                sliced.null_count(),
                "nulls should survive slicing at offset {offset}"
            );
        }
    }

    #[test]
    fn to_polars_arrow_data_type() {
        let arrow_data_type = ArrowDataType::Int32;

        let polars_arrow_data_type = arrow_data_type.to_polars_arrow_data_type().unwrap();

        assert_eq!(polars_arrow_data_type, PolarsArrowDataType::Int32);
    }

    #[test]
    fn to_polars_arrow_field() {
        let arrow_field = ArrowField::new("my_int32_field", ArrowDataType::Int32, true);

        let polars_arrow_field = arrow_field.to_polars_arrow_field().unwrap();

        assert_eq!(polars_arrow_field.name(), &"my_int32_field");
        assert_eq!(polars_arrow_field.dtype(), &PolarsArrowDataType::Int32);
        assert_eq!(polars_arrow_field.is_nullable, true);
    }

    #[test]
    fn to_polars_arrow_schema() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("my_int32_field", ArrowDataType::Int32, true),
            ArrowField::new(
                "my_struct_field",
                ArrowDataType::Struct(
                    vec![
                        ArrowField::new("my_nested_boolean_field", ArrowDataType::Boolean, true),
                        ArrowField::new("my_nested_utf8_field", ArrowDataType::Utf8, false),
                    ]
                    .into(),
                ),
                false,
            ),
        ]);

        let polars_arrow_schema = arrow_schema.to_polars_arrow_schema().unwrap();

        assert_eq!(polars_arrow_schema.len(), 2);

        let int32_field = polars_arrow_schema.get("my_int32_field").unwrap();
        assert_eq!(int32_field.name(), &"my_int32_field");
        assert_eq!(int32_field.dtype(), &PolarsArrowDataType::Int32);
        assert_eq!(int32_field.is_nullable, true);

        let struct_field = polars_arrow_schema.get("my_struct_field").unwrap();
        assert_eq!(struct_field.name(), &"my_struct_field");
        assert_eq!(struct_field.is_nullable, false);
        match struct_field.dtype() {
            PolarsArrowDataType::Struct(nested_fields) => {
                assert_eq!(nested_fields.len(), 2);
                assert_eq!(nested_fields[0].name(), &"my_nested_boolean_field");
                assert_eq!(nested_fields[0].dtype(), &PolarsArrowDataType::Boolean);
                assert_eq!(nested_fields[0].is_nullable, true);
                assert_eq!(nested_fields[1].name(), &"my_nested_utf8_field");
                assert_eq!(nested_fields[1].dtype(), &PolarsArrowDataType::Utf8);
                assert_eq!(nested_fields[1].is_nullable, false);
            }
            _ => panic!("expected a struct field"),
        }
    }

    #[test]
    fn to_polars_arrow_array_ref() {
        let data = vec![Some(1), None, Some(3)];
        let arrow_array = ArrowInt32Array::from(data.clone());
        let arrow_array_ref = Arc::new(arrow_array) as ArrowArrayRef;

        let polars_arrow_array_ref = arrow_array_ref.to_polars_arrow_array_ref().unwrap();

        assert_eq!(polars_arrow_array_ref.dtype(), &PolarsArrowDataType::Int32);
        assert_eq!(polars_arrow_array_ref.len(), 3);
        assert_eq!(polars_arrow_array_ref.null_count(), 1);

        let polars_int32_array = polars_arrow_array_ref
            .as_any()
            .downcast_ref::<PolarsInt32Array>()
            .unwrap();
        assert_eq!(polars_int32_array, &PolarsInt32Array::from(data));
    }

    #[test]
    fn to_polars_arrow_record_batch() {
        let int32_data = vec![Some(1), None, Some(3)];
        let boolean_data = vec![Some(true), Some(false), None];
        let utf8_data = vec!["a", "b", "c"];

        let arrow_record_batch = ArrowRecordBatch::try_from_iter_with_nullable(vec![
            (
                "my_int32_field",
                Arc::new(ArrowInt32Array::from(int32_data.clone())) as ArrowArrayRef,
                true,
            ),
            (
                "my_struct_field",
                Arc::new(ArrowStructArray::from(vec![
                    (
                        Arc::new(ArrowField::new(
                            "my_nested_boolean_field",
                            ArrowDataType::Boolean,
                            true,
                        )),
                        Arc::new(ArrowBooleanArray::from(boolean_data.clone())) as ArrowArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new(
                            "my_nested_utf8_field",
                            ArrowDataType::Utf8,
                            false,
                        )),
                        Arc::new(ArrowStringArray::from(utf8_data.clone())) as ArrowArrayRef,
                    ),
                ])) as ArrowArrayRef,
                false,
            ),
        ])
        .unwrap();

        let polars_arrow_record_batch = arrow_record_batch.to_polars_arrow_record_batch().unwrap();

        assert_eq!(polars_arrow_record_batch.height(), 3);
        assert_eq!(polars_arrow_record_batch.width(), 2);

        let (field_0_name, field_0) = polars_arrow_record_batch.schema().get_at_index(0).unwrap();
        assert_eq!(field_0_name, &"my_int32_field");
        assert_eq!(field_0.is_nullable, true);
        assert_eq!(field_0.dtype(), &PolarsArrowDataType::Int32);

        let (field_1_name, field_1) = polars_arrow_record_batch.schema().get_at_index(1).unwrap();
        assert_eq!(field_1_name, &"my_struct_field");
        assert_eq!(field_1.is_nullable, false);
        match field_1.dtype() {
            PolarsArrowDataType::Struct(nested_fields) => {
                assert_eq!(nested_fields.len(), 2);
                assert_eq!(nested_fields[0].name(), &"my_nested_boolean_field");
                assert_eq!(nested_fields[0].dtype(), &PolarsArrowDataType::Boolean);
                assert_eq!(nested_fields[0].is_nullable, true);
                assert_eq!(nested_fields[1].name(), &"my_nested_utf8_field");
                assert_eq!(nested_fields[1].dtype(), &PolarsArrowDataType::Utf8);
                assert_eq!(nested_fields[1].is_nullable, false);
            }
            _ => panic!("expected a struct field"),
        }

        let polars_int32_array = polars_arrow_record_batch.columns()[0]
            .as_any()
            .downcast_ref::<PolarsInt32Array>()
            .unwrap();
        assert_eq!(polars_int32_array, &PolarsInt32Array::from(int32_data));

        let polars_struct_array = polars_arrow_record_batch.columns()[1]
            .as_any()
            .downcast_ref::<PolarsStructArray>()
            .unwrap();
        assert_eq!(polars_struct_array.len(), 3);
        assert_eq!(polars_struct_array.fields().len(), 2);

        let polars_boolean_array = polars_struct_array.values()[0]
            .as_any()
            .downcast_ref::<PolarsBooleanArray>()
            .unwrap();
        assert_eq!(
            polars_boolean_array,
            &PolarsBooleanArray::from(boolean_data)
        );

        let polars_nested_utf8_array = polars_struct_array.values()[1]
            .as_any()
            .downcast_ref::<PolarsUtf8Array<i32>>()
            .unwrap();
        assert_eq!(
            polars_nested_utf8_array,
            &PolarsUtf8Array::<i32>::from_slice(utf8_data)
        );
    }
}
