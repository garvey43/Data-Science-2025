"""
cli.py — Command-line interface for the password vault.

Subcommands:
  init    -> create key + optional empty vault file
  add     -> add a new entry
  get     -> search entries by keyword (names/usernames)
  list    -> list all entry names and usernames
  delete  -> delete entry by exact name

Examples:
  python cli.py init --vault myvault.json
  python cli.py add --vault myvault.json --name gmail --username you@x.com --password 'P@ssw0rd!'
  python cli.py get --vault myvault.json --keyword gmail --reveal
"""
import argparse
import getpass
from vault import Vault, VaultError


def cmd_init(args: argparse.Namespace) -> None:
    v = Vault(args.vault, key_path=args.key)
    v.generate_key()
    # Touch an empty vault file if not exists
    try:
        v.load_key()
        v.save()  # saves empty entries list
        print(f"Initialized vault at {args.vault} and key at {args.key}")
    except VaultError as e:
        print(f"Error: {e}")


def _load_vault(args: argparse.Namespace) -> Vault:
    v = Vault(args.vault, key_path=args.key)
    v.load_key()
    v.load()
    return v


def cmd_add(args: argparse.Namespace) -> None:
    v = _load_vault(args)
    password = args.password or getpass.getpass("Password: ")
    try:
        v.add(args.name, args.username, password)
        v.save()
        print(f"Added '{args.name}'")
    except VaultError as e:
        print(f"Error: {e}")


def cmd_get(args: argparse.Namespace) -> None:
    v = _load_vault(args)
    matches = v.search(args.keyword)
    if not matches:
        print("No matches.")
        return
    for e in matches:
        # Never print plaintext unless user opted in
        pwd = e.password if args.reveal else "********"
        print(f"- {e.name} | {e.username} | {pwd}")


def cmd_list(args: argparse.Namespace) -> None:
    v = _load_vault(args)
    for e in v.list_all():
        print(f"- {e.name} | {e.username}")


def cmd_delete(args: argparse.Namespace) -> None:
    v = _load_vault(args)
    if v.delete(args.name):
        v.save()
        print(f"Deleted '{args.name}'")
    else:
        print(f"No entry named '{args.name}'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local password vault (Fernet-encrypted)")
    p.add_argument("--vault", required=True, help="Path to JSON vault file")
    p.add_argument("--key", default="vault.key", help="Path to key file (default: vault.key)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Initialize vault and key")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add", help="Add a new entry")
    sp.add_argument("--name", required=True)
    sp.add_argument("--username", required=True)
    sp.add_argument("--password", help="If omitted, will prompt securely")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("get", help="Retrieve entries by keyword")
    sp.add_argument("--keyword", required=True)
    sp.add_argument("--reveal", action="store_true", help="Show plaintext passwords")
    sp.set_defaults(func=cmd_get)

    sp = sub.add_parser("list", help="List all entries (no passwords)")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("delete", help="Delete entry by exact name")
    sp.add_argument("--name", required=True)
    sp.set_defaults(func=cmd_delete)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except VaultError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()