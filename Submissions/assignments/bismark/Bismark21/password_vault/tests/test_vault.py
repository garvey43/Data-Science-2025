import os
import tempfile
import pytest
from vault import Vault


def fresh_paths(tmpdir):
    vault_path = os.path.join(tmpdir, "vault.json")
    key_path = os.path.join(tmpdir, "vault.key")
    return vault_path, key_path


def test_add_get_delete_cycle():
    with tempfile.TemporaryDirectory() as d:
        vault_path, key_path = fresh_paths(d)
        v = Vault(vault_path, key_path)
        v.generate_key()
        v.load_key()
        v.save()

        v.add("github", "ray", "TopSecret123")
        v.save()

        v2 = Vault(vault_path, key_path)
        v2.load_key()
        v2.load()
        matches = v2.search("git")
        assert len(matches) == 1
        assert matches[0].password == "TopSecret123"

        assert v2.delete("github") is True
        v2.save()
        v2.load()
        assert v2.search("git") == []


def test_input_validation_and_duplicates():
    with tempfile.TemporaryDirectory() as d:
        vault_path, key_path = fresh_paths(d)
        v = Vault(vault_path, key_path)
        v.generate_key()
        v.load_key()
        v.save()

        with pytest.raises(Exception):
            v.add("", "u", "p")
        with pytest.raises(Exception):
            v.add("n", "", "p")
        with pytest.raises(Exception):
            v.add("n", "u", "")

        v.add("n", "u", "p")
        with pytest.raises(Exception):
            v.add("n", "u2", "p2")