//! Validated experimental WALNUTS options passed as JSON by the R layer.
use nuts_rs::WalnutsOptions;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Overrides {
    max_step_halvings: Option<u64>,
    max_error: Option<f64>,
    min_micro_steps: Option<u64>,
    energy_criterion: Option<nuts_rs::WalnutsEnergyCriterion>,
    target_macro_steps: Option<f64>,
    max_min_micro_steps: Option<u64>,
    max_leapfrog_steps: Option<u64>,
}

pub(crate) fn parse(json: &str) -> anyhow::Result<(WalnutsOptions, Option<u64>)> {
    let payload: serde_json::Value = serde_json::from_str(json)?;
    let fields = payload
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("walnuts options must be an object"))?;
    for (name, value) in fields {
        anyhow::ensure!(
            !value.is_null()
                || matches!(name.as_str(), "target_macro_steps" | "max_leapfrog_steps"),
            "walnuts {name} must not be null"
        );
    }
    let values: Overrides = serde_json::from_value(payload)?;
    for (name, value) in [
        ("min_micro_steps", values.min_micro_steps),
        ("max_min_micro_steps", values.max_min_micro_steps),
        ("max_leapfrog_steps", values.max_leapfrog_steps),
    ] {
        if let Some(value) = value {
            anyhow::ensure!(
                value <= i32::MAX as u64,
                "walnuts {name} exceeds the R integer range"
            );
        }
    }
    let mut options = WalnutsOptions::default();
    if let Some(value) = values.max_step_halvings {
        options.max_step_halvings = value;
    }
    if let Some(value) = values.max_error {
        options.max_error = value;
    }
    if let Some(value) = values.min_micro_steps {
        options.min_micro_steps = value;
    }
    if let Some(value) = values.energy_criterion {
        options.energy_criterion = value;
    }
    options.target_macro_steps = values.target_macro_steps;
    if let Some(value) = values.max_min_micro_steps {
        options.max_min_micro_steps = value;
    }
    anyhow::ensure!(
        options.max_error.is_finite() && options.max_error > 0.0,
        "walnuts max_error must be positive and finite"
    );
    anyhow::ensure!(
        options.min_micro_steps > 0,
        "walnuts min_micro_steps must be positive"
    );
    if let Some(target) = options.target_macro_steps {
        anyhow::ensure!(
            options.max_min_micro_steps >= options.min_micro_steps,
            "walnuts max_min_micro_steps must be at least min_micro_steps when adapting"
        );
        anyhow::ensure!(
            target.is_finite() && target > 0.0,
            "walnuts target_macro_steps must be positive and finite"
        );
    }
    // Check before converting or shifting: upstream casts the exponent to u32.
    anyhow::ensure!(
        options.max_step_halvings < 64,
        "walnuts max_step_halvings must be less than 64"
    );
    let max_micro_steps = if options.target_macro_steps.is_some() {
        options.max_min_micro_steps
    } else {
        options.min_micro_steps
    };
    anyhow::ensure!(
        max_micro_steps <= u64::MAX >> options.max_step_halvings,
        "walnuts microstep count overflows u64"
    );
    if let Some(limit) = values.max_leapfrog_steps {
        anyhow::ensure!(limit > 0, "walnuts max_leapfrog_steps must be positive");
    }
    Ok((options, values.max_leapfrog_steps))
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn empty_options_keep_upstream_defaults() {
        let (options, limit) = parse("{}").unwrap();
        assert_eq!(options.max_step_halvings, 8);
        assert_eq!(options.min_micro_steps, 1);
        assert!(options.target_macro_steps.is_none());
        assert!(limit.is_none());
    }

    #[test]
    fn malformed_options_and_unsafe_shifts_are_rejected() {
        for json in [
            r#"{"unknown":1}"#,
            r#"{"max_error":null}"#,
            r#"{"min_micro_steps":null}"#,
            r#"{"energy_criterion":null}"#,
            r#"{"max_leapfrog_steps":2147483648}"#,
            r#"{"min_micro_steps":0}"#,
            r#"{"min_micro_steps":1.5}"#,
            r#"{"max_error":0}"#,
            r#"{"max_step_halvings":4294967296}"#,
            r#"{"max_step_halvings":63,"min_micro_steps":2}"#,
            r#"{"target_macro_steps":0}"#,
            r#"{"target_macro_steps":15,"min_micro_steps":2,"max_min_micro_steps":1}"#,
            r#"{"max_leapfrog_steps":0}"#,
            r#"{"energy_criterion":"bogus"}"#,
        ] {
            assert!(parse(json).is_err(), "accepted {json}");
        }
    }

    #[test]
    fn adaptive_endpoint_options_are_preserved() {
        let (options, limit) = parse(
            r#"{"energy_criterion":"Endpoint","target_macro_steps":15,"max_leapfrog_steps":1000}"#,
        )
        .unwrap();
        assert!(matches!(
            options.energy_criterion,
            nuts_rs::WalnutsEnergyCriterion::Endpoint
        ));
        assert_eq!(options.target_macro_steps, Some(15.0));
        assert_eq!(limit, Some(1000));
    }
}
