mod test_utils;
use approx::assert_abs_diff_eq;
use ml_runner::model::Model;
use test_utils::FixtureModelInput;
use test_utils::FloatVec;

#[test]
fn run_add_model() {
    let fixture_input = FixtureModelInput::load_json("add_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_add_constant_model() {
    let fixture_input = FixtureModelInput::load_json("add_constant_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_add_broadcast_model() {
    let fixture_input = FixtureModelInput::load_json("add_broadcast_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_concat_model() {
    let fixture_input = FixtureModelInput::load_json("concat_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_expand_model() {
    let fixture_input = FixtureModelInput::load_json("expand_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_gather_model() {
    let fixture_input = FixtureModelInput::load_json("gather_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_gather_scalar_index_model() {
    let fixture_input = FixtureModelInput::load_json("gather_scalar_index_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_gather_negative_index_model() {
    let fixture_input = FixtureModelInput::load_json("gather_negative_index_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_squeeze_model() {
    let fixture_input = FixtureModelInput::load_json("squeeze_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_unsqueeze_model() {
    let fixture_input = FixtureModelInput::load_json("unsqueeze_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}

#[test]
fn run_transpose_model() {
    let fixture_input = FixtureModelInput::load_json("transpose_model.json");
    let model = Model::from_json(fixture_input.model_json.as_str()).unwrap();
    let model_output = fixture_input.run(&model);

    assert_abs_diff_eq!(
        FloatVec(model_output),
        FloatVec(fixture_input.test_output),
        epsilon = 1e-4
    );
}
