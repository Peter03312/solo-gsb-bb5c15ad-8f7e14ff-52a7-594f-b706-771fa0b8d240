"""投递撤销（POST /deliveries/{id}/revocation、GET /deliveries/{id}/revocation）：

一次撤销在事务内核对该投递编号的全部终点：已送达 / 永久失败的终结终点保留原回执，
其余终点标记撤销并使现有租约失效；全部终点已终结时返回无可撤销项且不留记录；
同编号再次撤销原样返回首次操作的逐终点结果、时间与原因。撤销后领取不再返回
已撤销终点，直接回执的新尝试与令牌完成对其明确拒绝（已记录回执的同序号同结果
重放仍原样返回）。与领取、回执并发时按 BEGIN IMMEDIATE 事务提交顺序裁决。
未知编号 404、原判定失败 409、空原因 422 且不留记录。判定重放、历史报表、
熔断统计与回执查询的既有约定完全不变。
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 必须在导入 app 之前设置，config 在导入时设置
os.environ.setdefault("ROUTER_DB", os.path.join(tempfile.mkdtemp(prefix="router-test-"), "test.db"))

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def client():
    db.init_db()
    conn = db.connect()
    conn.executescript(
        "DELETE FROM endpoint_leases;"
        "DELETE FROM delivery_attempts;"
        "DELETE FROM delivery_revocation_endpoints;"
        "DELETE FROM delivery_revocations;"
        "DELETE FROM domain_claim_limits;"
        "DELETE FROM draft_rules;"
        "DELETE FROM delivery_policy;"
        "DELETE FROM delivery_decisions;"
        "DELETE FROM circuit_breaker_settings;"
        "DELETE FROM circuit_breaks;"
        "DELETE FROM version_rules;"
        "DELETE FROM versions;"
        "DELETE FROM sqlite_sequence WHERE name = 'versions';"
        "DELETE FROM draft_revisions;"
        "INSERT INTO draft_revisions (revision, rules, rule_count, created_at)"
        " VALUES (0, '[]', 0, strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'));"
        "UPDATE draft_meta SET revision = 0 WHERE id = 1;"
    )
    conn.close()
    with TestClient(app) as c:
        yield c


RULES = [
    {
        "match_type": "exact",
        "match_key": "multi@corp.example.com",
        "action": "route",
        "targets": [
            {"kind": "mailbox", "value": "alice@mail.example.com"},
            {"kind": "mailbox", "value": "bob@mail.example.com"},
        ],
    },
    {
        "match_type": "exact",
        "match_key": "single@corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "carol@mail.example.com"}],
    },
    {
        "match_type": "exact",
        "match_key": "blocked@corp.example.com",
        "action": "reject",
        "targets": [],
    },
]


def setup(client):
    assert client.put("/draft", json={"revision": 0, "rules": RULES}).status_code == 200
    assert client.post("/publish", json={"revision": 1}).status_code == 200
    assert client.put("/policy", json={
        "revision": 0, "baseline_version": 1, "candidate_version": 1, "candidate_percent": 100,
    }).status_code == 200


def submit(client, delivery_id, address="multi@corp.example.com"):
    resp = client.post("/deliveries", json={"delivery_id": delivery_id, "address": address})
    assert resp.status_code == 200, resp.text
    return resp.json()


def receipt(client, delivery_id, mailbox, attempt_no, result):
    return client.post(f"/deliveries/{delivery_id}/receipts", json={
        "mailbox": mailbox, "attempt_no": attempt_no, "result": result,
    })


def revoke(client, delivery_id, reason="operator cancel"):
    return client.post(f"/deliveries/{delivery_id}/revocation", json={"reason": reason})


def get_revocation(client, delivery_id):
    return client.get(f"/deliveries/{delivery_id}/revocation")


def claim(client, max_claims=100, lease_seconds=300):
    return client.post("/deliveries/claims", json={
        "max_claims": max_claims, "lease_seconds": lease_seconds,
    })


def complete(client, token, result):
    return client.post(f"/deliveries/claims/{token}/complete", json={"result": result})


# ---------- 基本撤销与逐终点结果 ----------

def test_revoke_all_unfinished_endpoints(client):
    setup(client)
    submit(client, "msg-1")
    resp = revoke(client, "msg-1", "  cancel before send  ")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["delivery_id"] == "msg-1"
    assert data["revoked"] is True
    assert data["reason"] == "cancel before send"  # 去空白后留存
    assert data["revoked_at"] is not None
    assert datetime.fromisoformat(data["revoked_at"]).tzinfo is not None
    outcomes = {ep["mailbox"]: ep for ep in data["endpoints"]}
    assert set(outcomes) == {"alice@mail.example.com", "bob@mail.example.com"}
    for ep in data["endpoints"]:
        assert ep["outcome"] == "revoked"
        assert ep["revoked_at"] == data["revoked_at"]


def test_get_revocation_returns_stored_first_operation(client):
    setup(client)
    submit(client, "msg-1")
    first = revoke(client, "msg-1", "first reason").json()
    got = get_revocation(client, "msg-1")
    assert got.status_code == 200
    assert got.json() == first


def test_revoke_is_idempotent_same_result_time_reason(client):
    setup(client)
    submit(client, "msg-1")
    first = revoke(client, "msg-1", "original reason").json()
    second = revoke(client, "msg-1", "a different later reason")
    assert second.status_code == 200
    assert second.json() == first  # 再次撤销原样返回首次逐终点结果、时间与原因


def test_revoke_keeps_terminal_endpoints_revokes_the_rest(client):
    setup(client)
    submit(client, "msg-1")
    # alice 已送达（终结，保留）；bob 最新回执为暂时失败（未终结，撤销）
    assert receipt(client, "msg-1", "alice@mail.example.com", 1, "delivered").status_code == 200
    assert receipt(client, "msg-1", "bob@mail.example.com", 1, "temporary_failure").status_code == 200
    data = revoke(client, "msg-1", "partial cancel").json()
    assert data["revoked"] is True
    by = {ep["mailbox"]: ep["outcome"] for ep in data["endpoints"]}
    assert by == {"alice@mail.example.com": "delivered", "bob@mail.example.com": "revoked"}
    kept = next(ep for ep in data["endpoints"] if ep["mailbox"] == "alice@mail.example.com")
    assert kept["revoked_at"] is None  # 终结保留项无撤销时刻
    revoked = next(ep for ep in data["endpoints"] if ep["mailbox"] == "bob@mail.example.com")
    assert revoked["revoked_at"] == data["revoked_at"]


def test_permanent_failure_endpoint_preserved(client):
    setup(client)
    submit(client, "msg-1")
    receipt(client, "msg-1", "alice@mail.example.com", 1, "permanent_failure")
    receipt(client, "msg-1", "bob@mail.example.com", 2, "delivered")
    data = revoke(client, "msg-1").json()
    by = {ep["mailbox"]: ep["outcome"] for ep in data["endpoints"]}
    assert by == {
        "alice@mail.example.com": "permanent_failure",
        "bob@mail.example.com": "delivered",
    }


def test_all_terminal_returns_no_revocable_and_leaves_no_record(client):
    setup(client)
    submit(client, "msg-1")
    receipt(client, "msg-1", "alice@mail.example.com", 1, "delivered")
    receipt(client, "msg-1", "bob@mail.example.com", 1, "permanent_failure")
    resp = revoke(client, "msg-1", "nothing to do")
    assert resp.status_code == 200
    data = resp.json()
    assert data["revoked"] is False
    assert data["revoked_at"] is None
    assert data["reason"] == "nothing to do"
    assert {ep["outcome"] for ep in data["endpoints"]} == {"delivered", "permanent_failure"}
    # 不留撤销记录：GET 明确 404；再次撤销仍是无可撤销项
    assert get_revocation(client, "msg-1").status_code == 404
    again = revoke(client, "msg-1", "still nothing").json()
    assert again["revoked"] is False
    assert again["reason"] == "still nothing"  # 无记录则不回放首次原因


# ---------- 错误契约 ----------

def test_unknown_delivery_post_and_get_404(client):
    setup(client)
    post = revoke(client, "nope", "reason")
    assert post.status_code == 404
    assert post.json()["detail"]["error"] == "delivery_not_found"
    got = get_revocation(client, "nope")
    assert got.status_code == 404
    assert got.json()["detail"]["error"] == "delivery_not_found"


def test_failed_decision_rejected_post_and_get(client):
    setup(client)
    submit(client, "msg-bad", address="single@corp.example.com")  # 先成功一个，确保库可用
    resp = client.post("/deliveries", json={
        "delivery_id": "msg-blocked", "address": "blocked@corp.example.com",
    })
    assert resp.status_code == 422
    post = revoke(client, "msg-blocked", "reason")
    assert post.status_code == 409
    assert post.json()["detail"]["error"] == "delivery_decision_failed"
    got = get_revocation(client, "msg-blocked")
    assert got.status_code == 409
    assert got.json()["detail"]["error"] == "delivery_decision_failed"


def test_empty_reason_422_and_leaves_no_record(client):
    setup(client)
    submit(client, "msg-1")
    for raw in ("", "   ", "\t\n "):
        resp = revoke(client, "msg-1", raw)
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["error"] == "invalid_reason"
    assert get_revocation(client, "msg-1").status_code == 404  # 未留记录，之后仍可撤销
    assert revoke(client, "msg-1", "now valid").json()["revoked"] is True


def test_missing_reason_field_422(client):
    setup(client)
    submit(client, "msg-1")
    resp = client.post("/deliveries/msg-1/revocation", json={})
    assert resp.status_code == 422


def test_get_revocation_not_found_for_known_unrevoked_delivery(client):
    setup(client)
    submit(client, "msg-1")
    resp = get_revocation(client, "msg-1")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "revocation_not_found"


# ---------- 撤销与领取 / 租约 ----------

def test_claims_exclude_revoked_endpoints_and_lease_invalidated(client):
    setup(client)
    submit(client, "msg-1")
    submit(client, "msg-2", address="single@corp.example.com")
    claimed = claim(client).json()["tasks"]
    token_msg1 = {
        (t["delivery_id"], t["mailbox"]): t["token"] for t in claimed
    }
    assert ("msg-1", "alice@mail.example.com") in token_msg1

    data = revoke(client, "msg-1", "stop msg-1").json()
    assert data["revoked"] is True

    # 撤销后领取不再返回 msg-1 的任何终点
    after = claim(client).json()["tasks"]
    assert all(t["delivery_id"] != "msg-1" for t in after)

    # 既有租约令牌立即失效：完成被明确拒绝（reason=revoked），不写回执
    token = token_msg1[("msg-1", "alice@mail.example.com")]
    done = complete(client, token, "delivered")
    assert done.status_code == 409
    detail = done.json()["detail"]
    assert detail["error"] == "invalid_lease_token"
    assert detail["reason"] == "revoked"

    # 撤销后再领取，仍不会把 msg-1 的终点重新租出
    assert claim(client).json()["tasks"] == []
    # 撤销后不得写入新回执
    assert receipt(client, "msg-1", "alice@mail.example.com", 1, "delivered").status_code == 409


def test_revoke_releases_domain_in_flight_occupancy(client):
    setup(client)
    # 同一终点域 mail.example.com 上限 1：msg-1 的 alice 被领取后，msg-2 的 carol 被跳过
    assert client.put("/deliveries/claims/domain-limits", json={
        "domain": "mail.example.com", "max_in_flight": 1,
    }).status_code == 200
    submit(client, "msg-1")
    submit(client, "msg-2", address="single@corp.example.com")
    first = claim(client, max_claims=10).json()["tasks"]
    assert [(t["delivery_id"], t["mailbox"]) for t in first] == [
        ("msg-1", "alice@mail.example.com")
    ]
    # 撤销 msg-1：alice 的租约结束、名额释放（bob 也一同被撤销）
    assert revoke(client, "msg-1").json()["revoked"] is True
    second = claim(client, max_claims=10).json()["tasks"]
    assert [(t["delivery_id"], t["mailbox"]) for t in second] == [
        ("msg-2", "carol@mail.example.com")
    ]


# ---------- 撤销与直接回执 ----------

def test_direct_receipt_rejected_after_revoke_but_recorded_replay_returns_original(client):
    setup(client)
    submit(client, "msg-1")
    # 撤销前已记录 attempt 1 暂时失败
    first = receipt(client, "msg-1", "bob@mail.example.com", 1, "temporary_failure")
    assert first.status_code == 200
    first_body = first.json()
    assert revoke(client, "msg-1").json()["revoked"] is True

    # 新尝试明确拒绝
    new_attempt = receipt(client, "msg-1", "bob@mail.example.com", 2, "delivered")
    assert new_attempt.status_code == 409
    assert new_attempt.json()["detail"]["error"] == "endpoint_revoked"

    # 同序号不同结果：撤销后也是 endpoint_revoked（不写入、不按冲突处理）
    diff_result = receipt(client, "msg-1", "bob@mail.example.com", 1, "delivered")
    assert diff_result.status_code == 409
    assert diff_result.json()["detail"]["error"] == "endpoint_revoked"

    # 已记录回执的同序号同结果重试仍原样返回（含首次 created_at）
    replay = receipt(client, "msg-1", "bob@mail.example.com", 1, "temporary_failure")
    assert replay.status_code == 200
    assert replay.json() == first_body


def test_receipts_query_only_reflects_actual_attempts_after_revoke(client):
    setup(client)
    submit(client, "msg-1")
    receipt(client, "msg-1", "alice@mail.example.com", 1, "temporary_failure")
    revoke(client, "msg-1")
    summary = client.get("/deliveries/msg-1/receipts").json()
    by = {ep["mailbox"]: ep for ep in summary["endpoints"]}
    # 回执查询只反映实际尝试：撤销本身不写回执、不改汇总
    assert by["alice@mail.example.com"]["status"] == "temporary_failure"
    assert len(by["alice@mail.example.com"]["attempts"]) == 1
    assert by["bob@mail.example.com"]["status"] == "pending"
    assert by["bob@mail.example.com"]["attempts"] == []
    assert summary["status"] == "pending"


# ---------- 并发：撤销与回执 / 领取按事务提交顺序裁决 ----------

def test_concurrent_revoke_and_token_complete_adjucated_by_commit_order(client):
    setup(client)
    submit(client, "msg-1")
    # 重复构造多轮，每轮只有两种合法裁决之一（由 BEGIN IMMEDIATE 提交顺序决定）：
    # 1) 令牌完成先提交（delivered 终结）→ 撤销看到终结，该终点保留；
    # 2) 撤销先提交 → 令牌完成得到 409 revoked，该终点已撤销。
    saw_delivered_then_kept = 0
    saw_revoked_then_rejected = 0
    for i in range(20):
        did = f"msg-c{i}"
        submit(client, did)
        token = next(
            t["token"] for t in claim(client, lease_seconds=300).json()["tasks"]
            if t["delivery_id"] == did and t["mailbox"] == "alice@mail.example.com"
        )
        barrier = []

        def finish():
            barrier_wait()
            return complete(client, token, "delivered").status_code

        def do_revoke():
            barrier_wait()
            return revoke(client, did, "concurrent").json()

        def barrier_wait():
            barrier.append(1)
            while len(barrier) < 2:
                pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_done = pool.submit(finish)
            f_rev = pool.submit(do_revoke)
            code = f_done.result()
            rev = f_rev.result()

        rev_by = {ep["mailbox"]: ep["outcome"] for ep in rev["endpoints"]}
        if code == 200:
            # 完成先提交：该终点已 delivered，撤销须保留原回执
            assert rev_by["alice@mail.example.com"] == "delivered"
            saw_delivered_then_kept += 1
        else:
            assert code == 409
            assert rev_by["alice@mail.example.com"] == "revoked"
            saw_revoked_then_rejected += 1
        # 裁决后状态封口：再撤销回放首次结果；该终点不会再被领取
        again = revoke(client, did, "again").json()
        assert {ep["mailbox"]: ep["outcome"] for ep in again["endpoints"]} == rev_by
        assert all(t["delivery_id"] != did for t in claim(client).json()["tasks"])

    assert saw_delivered_then_kept + saw_revoked_then_rejected == 20
    assert saw_delivered_then_kept > 0 and saw_revoked_then_rejected > 0


# ---------- 既有功能不变 ----------

def test_revocation_does_not_change_decision_replay_report(client):
    setup(client)
    decision_before = submit(client, "msg-1")
    revoke(client, "msg-1", "cancel")
    # 判定查询与同编号重放内容完全不变（仍含全部终点）
    got = client.get("/deliveries/msg-1").json()
    assert got["status"] == "ok"
    assert {ep["mailbox"] for ep in got["endpoints"]} == {
        "alice@mail.example.com", "bob@mail.example.com",
    }
    replay = client.post("/deliveries", json={
        "delivery_id": "msg-1", "address": "multi@corp.example.com",
    }).json()
    assert replay == decision_before
