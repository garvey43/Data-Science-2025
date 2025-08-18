"""
vault.py — Core password vault logic: encryption, decryption, file IO.

Design:
- Vault stores entries in-memory as dicts: {name, username, password}
- On disk, only the password is encrypted (Fernet). Name/username remain plaintext for simple search.
- Key management: a single symmetric key stored in `vault.key`.

Exceptions are raised for invalid usage and caught in CLI for nice messages.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import os
from cryptography.fernet import Fernet, InvalidToken

KEY_FILE = "vault.key"

@dataclass
class Entry:
    name: str
    username: str
    password: str  # plaintext in memory

class VaultError(Exception):
    pass

class Vault:
    """A simple local password vault with Fernet encryption."""

    def __init__(self, vault_path: str, key_path: str = KEY_FILE):
        if not vault_path:
            raise VaultError("vault_path must be provided")
        self.vault_path = vault_path
        self.key_path = key_path
        self._entries: List[Entry] = []
        self._fernet: Fernet | None = None

    # ---------- Key Management ----------
    def has_key(self) -> bool:
        return os.path.exists(self.key_path)

    def generate_key(self) -> None:
        if self.has_key():
            return  # DRY/YAGNI: don't overwrite
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)

    def load_key(self) -> None:
        if not self.has_key():
            raise VaultError(f"Key file not found: {self.key_path}. Run 'init' first.")
        with open(self.key_path, "rb") as f:
            key = f.read()
        self._fernet = Fernet(key)

    def _require_fernet(self) -> Fernet:
        if self._fernet is None:
            self.load_key()
        assert self._fernet is not None
        return self._fernet

    # ---------- Disk IO ----------
    def load(self) -> None:
        """Load entries from JSON vault. Missing file -> start empty."""
        self._entries.clear()
        if not os.path.exists(self.vault_path):
            return
        try:
            with open(self.vault_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise VaultError(f"Corrupted vault file: {e}")

        fernet = self._require_fernet()
        for item in data.get("entries", []):
            try:
                token = item["password_encrypted"].encode("utf-8")
                password = fernet.decrypt(token).decode("utf-8")
            except (KeyError, InvalidToken) as e:
                raise VaultError(f"Failed to decrypt an entry: {e}")
            self._entries.append(Entry(
                name=item.get("name", ""),
                username=item.get("username", ""),
                password=password,
            ))

    def save(self) -> None:
        """Persist entries to JSON with encrypted passwords."""
        fernet = self._require_fernet()
        out: Dict[str, List[Dict[str, str]]] = {"entries": []}
        for e in self._entries:
            token = fernet.encrypt(e.password.encode("utf-8")).decode("utf-8")
            out["entries"].append({
                "name": e.name,
                "username": e.username,
                "password_encrypted": token,
            })
        tmp_path = self.vault_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp_path, self.vault_path)

    # ---------- CRUD ----------
    def add(self, name: str, username: str, password: str) -> None:
        name = name.strip()
        username = username.strip()
        if not name or not username or not password:
            raise VaultError("name, username, and password are required")
        if any(e.name == name for e in self._entries):
            raise VaultError(f"Entry with name '{name}' already exists")
        self._entries.append(Entry(name, username, password))

    def delete(self, name: str) -> bool:
        name = name.strip()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.name != name]
        return len(self._entries) < before

    def search(self, keyword: str) -> List[Entry]:
        k = keyword.strip().lower()
        return [e for e in self._entries if k in e.name.lower() or k in e.username.lower()]

    def list_all(self) -> List[Entry]:
        return list(self._entries)