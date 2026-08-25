"""Role-based access control for retrieval (Phase 4).

A query carries a User (a set of roles); retrieval returns only chunks whose
document ACL (metadata.allowed_roles) intersects those roles. The filter is applied
at the DB read on BOTH the dense and lexical paths, so a forbidden chunk never
enters either candidate pool (and never influences BM25 statistics).

Policy:
  - user=None  -> no filter (a trusted/internal call, e.g. the eval harness).
  - a real User -> DEFAULT-DENY: a chunk with no allowed_roles is invisible; a chunk
    with allowed_roles is visible only if a role matches. Tag every document you
    want reachable.
"""

from dataclasses import dataclass, field

from psycopg import sql


@dataclass
class User:
    id: str = "public"
    roles: list[str] = field(default_factory=lambda: ["public"])


PUBLIC = User(id="public", roles=["public"])
ADMIN = User(id="admin", roles=["admin", "public"])


def acl_condition(user: User | None) -> tuple[sql.Composable, list] | None:
    """Return (sql_condition, params) restricting rows to what `user` may see, or
    None for an unrestricted call. Uses jsonb_exists_any (the `?|` operator) so a
    chunk is visible only if its allowed_roles array shares a role with the user;
    chunks lacking allowed_roles match nothing (default-deny)."""
    if user is None:
        return None
    return sql.SQL("jsonb_exists_any(metadata->'allowed_roles', %s::text[])"), [user.roles]


def where_clause(user: User | None) -> tuple[sql.Composable, list]:
    """Convenience: an SQL WHERE fragment (empty when unrestricted) + its params."""
    acl = acl_condition(user)
    if acl is None:
        return sql.SQL(""), []
    return sql.SQL("WHERE {}").format(acl[0]), acl[1]
