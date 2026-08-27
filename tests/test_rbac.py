"""RBAC negative test (Phase 4 exit criterion).

Proves a restricted (public) user cannot retrieve a forbidden (admin-only) passage,
while an admin user can — across BOTH dense and hybrid retrieval. Also confirms the
generator refuses for the public user (the answer isn't in the context it's allowed
to see).

Runs against EVERY config the system actually depends on — the Phase 1 BASELINE and
the Phase 5 PRODUCTION winner that the API serves. Access control is a non-negotiable
requirement, so proving it on a table the API no longer reads would not discharge it:
each config gets its own physical table, and a table is only as safe as it's tested.

Self-contained: for each config, ingests an admin-only confidential doc alongside the
public paper, runs the assertions, then removes the confidential doc so the eval
tables are left paper-only. Run:  python -m tests.test_rbac
"""

from dataclasses import replace

from psycopg import sql

from app.db import config_table, connect
from app.generator import generate_answer
from app.pipeline import BASELINE, PRODUCTION, RunConfig
from app.rbac import ADMIN, PUBLIC
from app.retriever import retrieve
from app.store import ingest_document

PAPER = "data/mota-origenes.pdf"
CONFIDENTIAL = "data/confidential.md"
# A question whose answer lives ONLY in the confidential doc.
SECRET_QUERY = "What is the internal codename for the Q3 product launch?"


def _delete_source(config: RunConfig, source: str) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("DELETE FROM {} WHERE metadata->>'source' = %s").format(
                config_table(config.config_id)
            ),
            (source,),
        )
        conn.commit()


def _has_confidential(chunks) -> bool:
    return any(c.source == CONFIDENTIAL for c in chunks)


def _check_config(name: str, config: RunConfig, failures: list[str]) -> None:
    """Run the full dense/hybrid/generation battery against one config's table."""
    print(f"\n--- {name}: {config.label} ({config.config_id}) ---")

    # Tag the corpus: paper public, confidential admin-only (same table).
    ingest_document(PAPER, config, allowed_roles=["public"])
    ingest_document(CONFIDENTIAL, config, allowed_roles=["admin"])

    try:
        for mode in ("dense", "hybrid"):
            cfg = replace(config, retrieval_mode=mode)

            public_hits = retrieve(SECRET_QUERY, cfg, k=5, user=PUBLIC)
            admin_hits = retrieve(SECRET_QUERY, cfg, k=5, user=ADMIN)

            # NEGATIVE: public must NOT see the confidential passage.
            if _has_confidential(public_hits):
                failures.append(f"[{name}/{mode}] public user RETRIEVED the confidential doc")
            else:
                print(f"[{name}/{mode}] public user correctly DENIED the confidential doc "
                      f"({len(public_hits)} public hit(s))")

            # POSITIVE: admin SHOULD see it (proves the doc is retrievable, so the
            # negative result is due to access control, not a broken query).
            if _has_confidential(admin_hits):
                print(f"[{name}/{mode}] admin user correctly retrieved the confidential doc")
            else:
                failures.append(f"[{name}/{mode}] admin user did NOT retrieve the confidential doc")

        # Generation: the public user's answer must not leak the secret; it should refuse.
        public_ctx = retrieve(SECRET_QUERY, config, k=5, user=PUBLIC)
        answer = generate_answer(SECRET_QUERY, public_ctx)
        if "bluefin" in answer.lower():
            failures.append(f"[{name}] generator LEAKED the secret to the public user")
        else:
            print(f"[{name}] generator did not leak to public user; answer: {answer!r}")
    finally:
        # Clean up so the eval tables are left paper-only.
        _delete_source(config, CONFIDENTIAL)
        print(f"[{name}] cleanup: removed {CONFIDENTIAL} from {config.config_id}")


def main() -> int:
    # Every config the system depends on, not just the one that happens to be default.
    configs = {"baseline": BASELINE, "production": PRODUCTION}

    failures: list[str] = []
    for name, config in configs.items():
        _check_config(name, config, failures)

    if failures:
        print("\nRBAC NEGATIVE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nRBAC NEGATIVE TEST PASSED ({len(configs)} configs)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
