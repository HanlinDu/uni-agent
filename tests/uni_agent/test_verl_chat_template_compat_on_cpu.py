from __future__ import annotations


def test_legacy_verl_chat_template_import_path_is_available() -> None:
    from verl.utils.chat_template import apply_chat_template, initialize_system_prompt

    assert callable(apply_chat_template)
    assert callable(initialize_system_prompt)