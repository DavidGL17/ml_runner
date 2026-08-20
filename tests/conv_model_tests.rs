mod test_utils;
use approx::assert_abs_diff_eq;
use ml_runner::model::Model;
use test_utils::FixtureModelInput;
use test_utils::FloatVec;

#[test]
fn run_conv_simple_model() {
    let fixture_input = FixtureModelInput::load_json("conv_simple_model.json");

    // Parse the JSON into a Model
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();

    // Run forward pass
    let model_output = model.forward(&fixture_input.test_input).unwrap();

    // Verify the result
    assert_abs_diff_eq!(
        FloatVec(model_output.to_vec()),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_conv_flatten_model() {
    let fixture_input = FixtureModelInput::load_json("conv_flatten_model.json");

    // Parse the JSON into a Model
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();

    // Run forward pass
    let model_output = model.forward(&fixture_input.test_input).unwrap();

    // Verify the result
    assert_abs_diff_eq!(
        FloatVec(model_output.to_vec()),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_conv_flatten_dense_activation_model() {
    let fixture_input = FixtureModelInput::load_json("conv_flatten_dense_activation_model.json");

    // Parse the JSON into a Model
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();

    // Run forward pass
    let model_output = model.forward(&fixture_input.test_input).unwrap();

    // Verify the result
    assert_abs_diff_eq!(
        FloatVec(model_output.to_vec()),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}
