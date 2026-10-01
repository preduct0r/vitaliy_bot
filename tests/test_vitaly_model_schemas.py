"""Codex structured output rejects object schemas with optional properties."""

from tof_reliability.vitaly_bot import learning_runtime, service
from tof_reliability.vitaly_bot.profile import VitalyProfile, default_profile_path


def _assert_all_object_properties_required(schema, path="root"):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert set(schema["properties"]) == set(schema["required"]), path
        for key, child in schema.items():
            _assert_all_object_properties_required(child, path + "/" + key)
    elif isinstance(schema, list):
        for index, child in enumerate(schema):
            _assert_all_object_properties_required(child, path + "/" + str(index))


def test_all_static_model_schemas_are_strict():
    profile = VitalyProfile.load(default_profile_path())
    schemas = [service.review_schema(profile), service.MATERIAL_MAP_SCHEMA,
               service.FEEDBACK_SCHEMA, service.FINISH_SCHEMA]
    schemas.extend(getattr(learning_runtime, key) for key in dir(learning_runtime)
                   if key.endswith("_SCHEMA") and isinstance(getattr(learning_runtime, key), dict))
    for schema in schemas:
        _assert_all_object_properties_required(schema)
