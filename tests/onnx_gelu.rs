#![cfg(feature = "onnx")]

use webnn_graph::ast::DataType;
use webnn_graph::emit_js::emit_builder_js;
use webnn_graph::onnx::convert::{ConvertOptions, OnnxConverter, OnnxError};
use webnn_graph::protos::onnx::{
    tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use webnn_graph::validate::validate_graph;

fn model(domain: &str, opset: i64, dtype: i32, approximate: Option<&str>) -> ModelProto {
    let tensor_type = TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: dtype,
            shape: Some(TensorShapeProto {
                dim: vec![tensor_shape_proto::Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(3)),
                    ..Default::default()
                }],
            }),
        })),
        ..Default::default()
    };
    let value = |name: &str| ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type.clone()),
        ..Default::default()
    };
    let attribute = approximate
        .map(|value| AttributeProto {
            name: "approximate".to_string(),
            r#type: 3,
            s: value.as_bytes().to_vec(),
            ..Default::default()
        })
        .into_iter()
        .collect();
    ModelProto {
        ir_version: 9,
        graph: Some(GraphProto {
            name: "gelu_regression".to_string(),
            input: vec![value("x")],
            output: vec![value("y")],
            node: vec![NodeProto {
                name: "activation".to_string(),
                op_type: "Gelu".to_string(),
                domain: domain.to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute,
                ..Default::default()
            }],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: domain.to_string(),
            version: opset,
        }],
        ..Default::default()
    }
}

#[test]
fn standard_gelu_opset20_still_fails_at_the_opset_guard() {
    // Standard-domain Gelu first appeared in opset 20. The handler regression
    // must not be mistaken for support of checker-valid opset-20 models.
    for approximate in [None, Some("none"), Some("tanh")] {
        for optimize in [false, true] {
            let error = OnnxConverter::new(model("", 20, 1, approximate))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap_err();
            assert!(matches!(
                error,
                OnnxError::UnsupportedOpset { version: 20, .. }
            ));
        }
    }
}

#[test]
fn legacy_exact_gelu_still_converts_validates_and_emits_js() {
    // The older com.microsoft Gelu is exact and has no approximate attribute.
    for (dtype, expected_type) in [(1, DataType::Float32), (10, DataType::Float16)] {
        for optimize in [false, true] {
            let graph = OnnxConverter::new(model("com.microsoft", 1, dtype, None))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap();
            validate_graph(&graph).unwrap();
            assert_eq!(graph.inputs["x"].data_type, expected_type);
            assert_eq!(graph.nodes.len(), 1);
            assert_eq!(graph.nodes[0].op, "gelu");
            assert_eq!(graph.nodes[0].inputs, ["x"]);
            assert!(graph.nodes[0].options.is_empty());
            assert!(emit_builder_js(&graph).contains("builder[\"gelu\"](env.get(\"x\"), {})"));
        }
    }
}

#[test]
fn legacy_gelu_rejects_an_unsupported_approximation_attribute() {
    // Deliberately malformed legacy nodes used to reach the handler and have
    // their approximation discarded. Optimizing must not hide this error.
    for approximate in ["tanh", "invalid"] {
        for optimize in [false, true] {
            let error = OnnxConverter::new(model("com.microsoft", 1, 1, Some(approximate)))
                .unwrap()
                .convert(&ConvertOptions {
                    optimize,
                    ..Default::default()
                })
                .unwrap_err();
            match approximate {
                "tanh" => assert!(matches!(error, OnnxError::UnsupportedOp { .. })),
                _ => assert!(matches!(error, OnnxError::InvalidAttribute { .. })),
            }
            assert!(error.to_string().contains("activation"));
        }
    }
}
