// Activation and unary math operators: Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use serde_json::Map;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu"
                | "Gelu"
                | "Tanh"
                | "Sigmoid"
                | "Sqrt"
                | "Exp"
                | "Log"
                | "Abs"
                | "Neg"
                | "Erf"
                | "Cos"
                | "Sin"
                | "Identity"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        if op_type == "Gelu" {
            Self::validate_gelu_approximation(node, &node_name)?;
        }

        // Map ONNX operator to WebNN operation name
        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            "Cos" => "cos",
            "Sin" => "sin",
            "Identity" => "identity",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        self.convert_unary(node, &node_name, webnn_op, context)
    }
}

impl ActivationHandler {
    fn validate_gelu_approximation(node: &NodeProto, node_name: &str) -> Result<(), OnnxError> {
        let invalid = |reason: &str| OnnxError::InvalidAttribute {
            attr: "approximate".to_string(),
            op: "Gelu".to_string(),
            node: node_name.to_string(),
            reason: reason.to_string(),
        };
        let mut attributes = node.attribute.iter().filter(|a| a.name == "approximate");
        let Some(attribute) = attributes.next() else {
            return Ok(());
        };
        if attributes.next().is_some() {
            return Err(invalid("attribute must not be repeated"));
        }
        if attribute.r#type != crate::protos::onnx::attribute_proto::AttributeType::String as i32 {
            return Err(invalid("expected a string"));
        }
        match attribute.s.as_slice() {
            b"none" => Ok(()),
            // WebNN gelu is the exact erf-based operation, not ONNX's tanh variant.
            // Fail closed until a semantics-preserving decomposition is available.
            b"tanh" => Err(OnnxError::UnsupportedOp {
                op: "Gelu (approximate=tanh)".to_string(),
                node: node_name.to_string(),
            }),
            _ => Err(invalid("expected 'none' or 'tanh'")),
        }
    }

    /// Convert ONNX unary/activation operation to WebNN
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let options = Map::new();

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: webnn_op.to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.output.as_slice().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(handler.supports("Cos"));
        assert!(handler.supports("Sin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "relu");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "sqrt");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "gelu");
    }

    fn convert_gelu_attributes(
        attributes: Vec<crate::protos::onnx::AttributeProto>,
    ) -> Result<ConversionResult, OnnxError> {
        let mut node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        node.attribute = attributes;
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        ActivationHandler.convert(
            &node,
            &ConversionContext {
                initializers: &initializers,
                value_shapes: &value_shapes,
                value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
                const_values: &const_values,
                value_ids: &value_ids,
                value_types: &value_types,
            },
        )
    }

    fn approximate(value: &[u8]) -> crate::protos::onnx::AttributeProto {
        crate::protos::onnx::AttributeProto {
            name: "approximate".to_string(),
            r#type: 3, // AttributeProto::STRING
            s: value.to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn test_gelu_default_and_none_preserve_exact_operation() {
        for attributes in [vec![], vec![approximate(b"none")]] {
            let result = convert_gelu_attributes(attributes).expect("exact GELU");
            assert_eq!(result.nodes.len(), 1);
            assert_eq!(result.nodes[0].op, "gelu");
            assert_eq!(result.nodes[0].inputs, ["x"]);
            assert!(result.nodes[0].options.is_empty());
            assert_eq!(result.output_mappings.get("y"), Some(&"y".to_string()));
        }
    }

    #[test]
    fn test_gelu_tanh_is_not_silently_replaced_with_exact_gelu() {
        let error = convert_gelu_attributes(vec![approximate(b"tanh")])
            .expect_err("tanh GELU requires a separate lowering");
        assert!(matches!(error, OnnxError::UnsupportedOp { .. }));
        let message = error.to_string();
        assert!(message.contains("Gelu"));
        assert!(message.contains("tanh"));
        assert!(message.contains("test_gelu"));
    }

    #[test]
    fn test_gelu_rejects_invalid_or_malformed_approximation() {
        let mut wrong_type = approximate(b"none");
        wrong_type.r#type = 2; // INT, even if the string field is populated.
        let mut missing_type = approximate(b"none");
        missing_type.r#type = 0;
        for attributes in [
            vec![approximate(b"invalid")],
            vec![approximate(b"")],
            vec![approximate(b"TANH")],
            vec![approximate(&[0xff])],
            vec![wrong_type],
            vec![missing_type],
            vec![approximate(b"none"), approximate(b"tanh")],
        ] {
            let error = convert_gelu_attributes(attributes)
                .expect_err("invalid approximation must not become exact GELU");
            assert!(matches!(
                &error,
                OnnxError::InvalidAttribute { attr, op, node, .. }
                    if attr == "approximate" && op == "Gelu" && node == "test_gelu"
            ));
            let message = error.to_string();
            assert!(message.contains("approximate"), "{message}");
            assert!(message.contains("Gelu"), "{message}");
            assert!(message.contains("test_gelu"), "{message}");
        }
    }

    #[test]
    fn test_convert_cos() {
        let handler = ActivationHandler;
        let node = create_test_node("Cos", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "cos");
    }

    #[test]
    fn test_convert_sin() {
        let handler = ActivationHandler;
        let node = create_test_node("Sin", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "sin");
    }
}
