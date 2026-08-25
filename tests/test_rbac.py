"""RBAC negative test (Phase 4 exit criterion).

Proves a restricted (public) user cannot retrieve a forbidden (admin-only) passage,
while an admin user can — across BOTH dense and hybrid retrieval. Also confirms the
generator refuses for the public user (the answer isn't in the context it's allowed
to see).

Self-contained: ingests an admin-only confidential doc into the baseline table
alongside the public paper, runs the assertions, then removes the confidential doc
so the eval table is left paper-only. Run:  python -m tests.test_rbac
"""

from psycopg import sql

from app.db import config_table, connect
from app.generator import generate_answer
from app.pipeline import BASELINE, RunConfig
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


def main() -> int:
    # Tag the corpus: paper public, confidential admin-only (same baseline table).
    ingest_document(PAPER, BASELINE, allowed_roles=["public"])
    ingest_document(CONFIDENTIAL, BASELINE, allowed_roles=["admin"])

    failures: list[str] = []
    try:
        for mode in ("dense", "hybrid"):
            cfg = RunConfig(retrieval_mode=mode)

            public_hits = retrieve(SECRET_QUERY, cfg, k=5, user=PUBLIC)
            admin_hits = retrieve(SECRET_QUERY, cfg, k=5, user=ADMIN)

            # NEGATIVE: public must NOT see the confidential passage.
            if _has_confidential(public_hits):
                failures.append(f"[{mode}] public user RETRIEVED the confidential doc")
            else:
                print(f"[{mode}] public user correctly DENIED the confidential doc "
                      f"({len(public_hits)} public hit(s))")

            # POSITIVE: admin SHOULD see it (proves the doc is retrievable, so the
            # negative result is due to access control, not a broken query).
            if _has_confidential(admin_hits):
                print(f"[{mode}] admin user correctly retrieved the confidential doc")
            else:
                failures.append(f"[{mode}] admin user did NOT retrieve the confidential doc")

        # Generation: the public user's answer must not leak the secret; it should refuse.
        public_ctx = retrieve(SECRET_QUERY, RunConfig(), k=5, user=PUBLIC)
        answer = generate_answer(SECRET_QUERY, public_ctx)
        if "bluefin" in answer.lower():
            failures.append("generator LEAKED the secret to the public user")
        else:
            print(f"generator did not leak to public user; answer: {answer!r}")
    finally:
        # Clean up so the eval baseline table is left paper-only.
        _delete_source(BASELINE, CONFIDENTIAL)
        print(f"cleanup: removed {CONFIDENTIAL} from the baseline table")

    if failures:
        print("\nRBAC NEGATIVE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRBAC NEGATIVE TEST PASSED")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
