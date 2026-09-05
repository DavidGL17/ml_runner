mod test_utils;
use approx::assert_abs_diff_eq;
use ml_runner::model::Model;
use test_utils::FixtureModelInput;
use test_utils::FloatVec;

#[test]
fn run_activation_all_types_model() {
    let fixture_input = FixtureModelInput::load_json("activation_all_types_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}