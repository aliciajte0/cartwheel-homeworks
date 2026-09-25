"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from typing import Any

from datetime import timedelta

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    docs = load_policy_docs()
    for doc in docs:
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {"ok": False, "error": "not_found", "reason": f"no policy with id {policy_id!r}"}


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    query = query.strip()
    if not query:
        return {"ok": False, "error": "invalid_argument", "reason": "empty query"}
    if max_price_usd is not None and max_price_usd <= 0:
        return {"ok": False, "error": "invalid_argument", "reason": "price ceiling must be positive"}
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    tokens = query.lower().split()

    with db.connection() as conn:
        store_id = None
        if store is not None:
            s = db.get_store_by_name(conn, store)
            if s is None:
                return {"ok": False, "error": "not_found", "reason": f"no store matching {store!r}"}
            store_id = s.id

        products = db.list_products(conn, store_id)

    matches = []
    for p in products:
        text = f"{p.title} {p.description}".lower()
        if all(t in text for t in tokens):
            if max_price_usd is not None and p.price_usd > max_price_usd:
                continue
            matches.append(p)

    matches.sort(key=lambda p: (p.price_usd, p.id))
    matches = matches[:limit]

    return {
        "ok": True,
        "products": [
            {"product_id": p.id, "store_id": p.store_id, "title": p.title, "price_usd": p.price_usd}
            for p in matches
        ],
        "count": len(matches),
    }


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "support staff do not have their own orders; use get_order to look up a specific order",
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, limit=DEFAULT_ORDER_LIMIT)
        else:
            orders = db.list_orders_for_store(conn, ctx.store_id, limit=DEFAULT_ORDER_LIMIT)
    return {
        "ok": True,
        "orders": [o.to_public_dict() for o in orders],
        "count": len(orders),
    }


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": f"order #{order_id} has status '{order.status}'; orders can only be cancelled before shipment",
            }
        db.set_order_status(conn, order_id, "cancelled")
        return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    from rapidfuzz import fuzz

    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_order_search_candidates(conn, user_id=ctx.user_id)
        elif ctx.role == "merchant":
            orders = db.list_order_search_candidates(conn, store_id=ctx.store_id)
        elif ctx.role == "support":
            orders = db.list_order_search_candidates(conn, all_orders=True)
        else:
            return {"ok": False, "error": "unsupported_role"}

        products = {p.id: p.title for p in db.list_products(conn)}

        scored = []
        for order in orders:
            title = products.get(order.product_id)
            if title is None:
                continue
            score = fuzz.partial_ratio(query.lower(), title.lower())
            if score > 70:
                scored.append(order.to_public_dict())

    return {"ok": True, "orders": scored[:5]}


def check_refund_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Check whether an order is eligible for a return or refund, with a reason.

    Uses the same authorization as get_order: shoppers see only their own
    orders, merchants only their store's orders, support any order.

    Args:
        ctx: The caller's auth context.
        order_id: The order to check.

    Returns:
        On success: {"ok": True, "order_id": int, "eligible": bool,
        "reason": str, "return_window_days": int, "window_expires": str}
        where reason explains why the order is or is not eligible, and
        window_expires is the last eligible date (or null if not applicable).
        On failure: not_found or permission_denied errors.
    """
    facts = load_facts()
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        store = db.get_store(conn, order.store_id)
        as_of = db.world_asof(conn)

    override = store.return_window_days_override if store else None
    window = effective_return_window_days(facts["return_window_days"], override)

    if order.status != "delivered":
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": False,
            "reason": f"order status is '{order.status}'; only delivered orders are eligible for refund",
            "return_window_days": window,
            "window_expires": None,
        }
    if order.delivered_at is None:
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": False,
            "reason": "order has no delivery date recorded",
            "return_window_days": window,
            "window_expires": None,
        }
    expires = order.delivered_at + timedelta(days=window)
    age_days = (as_of - order.delivered_at).days
    if age_days < 0:
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": False,
            "reason": f"delivery date {order.delivered_at} is in the future",
            "return_window_days": window,
            "window_expires": expires.isoformat(),
        }
    if age_days > window:
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": False,
            "reason": (
                f"return window expired on {expires.isoformat()}; "
                f"order was delivered {order.delivered_at.isoformat()}, "
                f"{age_days} days ago, which exceeds the {window}-day window"
            ),
            "return_window_days": window,
            "window_expires": expires.isoformat(),
        }
    return {
        "ok": True,
        "order_id": order_id,
        "eligible": True,
        "reason": (
            f"order is eligible; delivered {order.delivered_at.isoformat()}, "
            f"{age_days} days ago, within the {window}-day return window "
            f"(expires {expires.isoformat()})"
        ),
        "return_window_days": window,
        "window_expires": expires.isoformat(),
    }


def get_store_info(ctx: AuthContext, store_name: str) -> dict[str, Any]:
    """Look up a store's public information by name or slug. Risk tier: read.

    Every role may look up any store's public information (store details
    are public help-center content), so this tool needs no permission check
    beyond resolving the store name.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        store_name: Store name or slug (case-insensitive).

    Returns:
        On success: {"ok": True, "store_id": int, "name": str, "slug": str,
        "category": str, "return_window_days": int,
        "has_return_window_override": bool, "restocking_fee_opt_in": bool,
        "policy_id": str | None} where return_window_days is the effective
        window (store override if present, otherwise platform default) and
        policy_id is the store's policy doc id if one exists.
        If no store matches: {"ok": False, "error": "not_found",
        "reason": ...} naming the store string.
    """
    facts = load_facts()
    with db.connection() as conn:
        store = db.get_store_by_name(conn, store_name)
    if store is None:
        return {"ok": False, "error": "not_found", "reason": f"no store matching {store_name!r}"}

    override = store.return_window_days_override
    window = effective_return_window_days(facts["return_window_days"], override)

    policy_id = None
    for doc in load_policy_docs():
        if store.slug in doc.policy_id:
            policy_id = doc.policy_id
            break

    return {
        "ok": True,
        "store_id": store.id,
        "name": store.name,
        "slug": store.slug,
        "category": store.category,
        "return_window_days": window,
        "has_return_window_override": override is not None,
        "restocking_fee_opt_in": store.restocking_fee_opt_in,
        "policy_id": policy_id,
    }


def track_shipment(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Return shipping status and estimated delivery for an order. Risk tier: read.

    Uses the same authorization as get_order: shoppers see only their own
    orders, merchants only their store's orders, support any order.

    Args:
        ctx: The caller's auth context.
        order_id: The order to track.

    Returns:
        On success: {"ok": True, "order_id": int, "status": str,
        "ordered_at": str, "shipped_at": str | None,
        "delivered_at": str | None, "estimated_delivery": str | None,
        "summary": str} where estimated_delivery is computed from
        shipped_at + shipping transit max days (facts.yaml), and summary
        is a human-readable description of the current shipping state.
        On failure: not_found or permission_denied errors.
    """
    facts = load_facts()
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )

    transit_days = facts["shipping_transit_days_max"]
    handling_days = facts["shipping_handling_days_max"]

    if order.status == "placed":
        est_ship = order.ordered_at + timedelta(days=handling_days)
        est_delivery = est_ship + timedelta(days=transit_days)
        summary = (
            f"order placed on {order.ordered_at.isoformat()}; "
            f"expected to ship by {est_ship.isoformat()} "
            f"and arrive by {est_delivery.isoformat()}"
        )
        return {
            "ok": True,
            "order_id": order_id,
            "status": order.status,
            "ordered_at": order.ordered_at.isoformat(),
            "shipped_at": None,
            "delivered_at": None,
            "estimated_delivery": est_delivery.isoformat(),
            "summary": summary,
        }

    shipped_at_str = order.shipped_at.isoformat() if order.shipped_at else None
    delivered_at_str = order.delivered_at.isoformat() if order.delivered_at else None

    if order.status == "shipped" and order.shipped_at is not None:
        est_delivery = order.shipped_at + timedelta(days=transit_days)
        summary = (
            f"shipped on {shipped_at_str}; "
            f"estimated delivery by {est_delivery.isoformat()}"
        )
        return {
            "ok": True,
            "order_id": order_id,
            "status": order.status,
            "ordered_at": order.ordered_at.isoformat(),
            "shipped_at": shipped_at_str,
            "delivered_at": None,
            "estimated_delivery": est_delivery.isoformat(),
            "summary": summary,
        }

    if order.status == "delivered":
        summary = (
            f"delivered on {delivered_at_str}"
        )
        return {
            "ok": True,
            "order_id": order_id,
            "status": order.status,
            "ordered_at": order.ordered_at.isoformat(),
            "shipped_at": shipped_at_str,
            "delivered_at": delivered_at_str,
            "estimated_delivery": None,
            "summary": summary,
        }

    summary = f"order status is '{order.status}'; shipment tracking is not applicable"
    return {
        "ok": True,
        "order_id": order_id,
        "status": order.status,
        "ordered_at": order.ordered_at.isoformat(),
        "shipped_at": shipped_at_str,
        "delivered_at": delivered_at_str,
        "estimated_delivery": None,
        "summary": summary,
    }


def check_cancel_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Check whether an order is eligible for cancellation, with a reason.

    Uses the same authorization as cancel_order: shoppers see only their own
    orders, merchants only their store's orders, support any order.

    Args:
        ctx: The caller's auth context.
        order_id: The order to check.

    Returns:
        On success: {"ok": True, "order_id": int, "eligible": bool,
        "reason": str, "current_status": str}
        On failure: not_found or permission_denied errors.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )

    if order.status == "placed":
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": True,
            "reason": "order has status 'placed' and has not shipped yet; it can be cancelled",
            "current_status": order.status,
        }
    return {
        "ok": True,
        "order_id": order_id,
        "eligible": False,
        "reason": (
            f"order has status '{order.status}'; "
            f"only orders with status 'placed' (not yet shipped) can be cancelled"
        ),
        "current_status": order.status,
    }


def summarize_order_history(ctx: AuthContext) -> dict[str, Any]:
    """Aggregate order statistics for the caller's scope. Risk tier: read.

    Role behavior:
        - shopper: stats for the shopper's own orders.
        - merchant: stats for the merchant's store's orders.
        - support: returns invalid_argument (support should look up specific
          orders rather than aggregating across all orders).

    Returns:
        On success: {"ok": True, "total_orders": int, "total_spent_usd": float,
        "by_status": dict, "oldest_order": str | None, "newest_order": str | None}
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "support staff should look up specific orders; use get_order or find_order instead",
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, limit=1000)
        else:
            orders = db.list_orders_for_store(conn, ctx.store_id, limit=1000)

    if not orders:
        return {
            "ok": True,
            "total_orders": 0,
            "total_spent_usd": 0.0,
            "by_status": {},
            "oldest_order": None,
            "newest_order": None,
        }

    by_status: dict[str, int] = {}
    total_cents = 0
    for o in orders:
        by_status[o.status] = by_status.get(o.status, 0) + 1
        total_cents += o.total_cents

    sorted_orders = sorted(orders, key=lambda o: o.ordered_at)
    return {
        "ok": True,
        "total_orders": len(orders),
        "total_spent_usd": total_cents / 100,
        "by_status": by_status,
        "oldest_order": sorted_orders[0].ordered_at.isoformat(),
        "newest_order": sorted_orders[-1].ordered_at.isoformat(),
    }
