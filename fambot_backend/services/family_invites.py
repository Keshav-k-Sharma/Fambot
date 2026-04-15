from __future__ import annotations

import base64
import io
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import segno
from fastapi import HTTPException
from firebase_admin import firestore

from fambot_backend.core.firebase_init import init_firebase
from fambot_backend.schemas import (
    AcceptFamilyInviteOut,
    FamilyGroupOut,
    FamilyInviteCreatedOut,
    FamilyMemberOut,
    FamilyRole,
    RemoveFamilyMemberOut,
)
from fambot_backend.services.family_roles import reciprocal_role
from fambot_backend.services.firestore_users import (
    get_user_family_group_id,
    get_user_profile,
    set_user_family_group_id,
)


class _InviteFlowError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail


def _as_family_role(raw: str) -> FamilyRole:
    allowed: set[str] = {
        "mother",
        "father",
        "son",
        "daughter",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "nephew",
        "niece",
        "husband",
        "wife",
    }
    if raw not in allowed:
        raise HTTPException(status_code=500, detail="Invalid family role in invite.")
    return cast(FamilyRole, raw)


def _db():
    init_firebase()
    return firestore.client()


def _skip_firestore() -> bool:
    return os.environ.get("FAMBOT_SKIP_FIRESTORE") == "1"


def _invite_ttl_seconds() -> int:
    raw = os.environ.get("FAMBOT_FAMILY_INVITE_TTL_SECONDS", "86400").strip()
    try:
        n = int(raw)
    except ValueError:
        return 86400
    return max(60, min(n, 60 * 60 * 24 * 30))


def build_invite_url(token: str) -> str:
    base = os.environ.get("FAMBOT_INVITE_BASE_URL", "").strip()
    if not base:
        return f"fambot://family-invite?token={token}"
    sep = "&" if "?" in base else "?"
    return f"{base.rstrip('/')}{sep}token={token}"


def qr_png_base64_for_url(url: str) -> str:
    q = segno.make(url, error="m")
    buf = io.BytesIO()
    q.save(buf, kind="png", scale=6, border=2)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- In-memory state when FAMBOT_SKIP_FIRESTORE=1 ---

_skip_groups: dict[str, dict[str, Any]] = {}
_skip_user_to_group: dict[str, str] = {}
_skip_invites: dict[str, dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_owner_group_id(uid: str) -> str:
    """Return group id where uid is owner; create group if needed."""
    if _skip_firestore():
        gid = _skip_user_to_group.get(uid)
        if gid and _skip_groups.get(gid, {}).get("ownerUid") == uid:
            return gid
        gid = str(uuid.uuid4())
        _skip_groups[gid] = {
            "ownerUid": uid,
            "members": {uid},
            "rels": {},
            "createdAt": _now(),
        }
        _skip_user_to_group[uid] = gid
        return gid

    gid = get_user_family_group_id(uid)
    if gid:
        ref = _db().collection("familyGroups").document(gid)
        snap = ref.get()
        if not snap.exists:
            raise HTTPException(status_code=409, detail="User family group reference is invalid.")
        data = snap.to_dict() or {}
        if data.get("ownerUid") != uid:
            raise HTTPException(
                status_code=403,
                detail="Only the group owner can invite members.",
            )
        return gid

    gid = str(uuid.uuid4())
    now = _now()
    ref = _db().collection("familyGroups").document(gid)
    ref.set(
        {
            "ownerUid": uid,
            "createdAt": now,
            "updatedAt": now,
        }
    )
    ref.collection("members").document(uid).set({"joinedAt": now})
    set_user_family_group_id(uid, gid)
    return gid


def _assert_owner(uid: str, group_id: str) -> None:
    if _skip_firestore():
        g = _skip_groups.get(group_id)
        if not g or g.get("ownerUid") != uid:
            raise HTTPException(status_code=403, detail="Not the owner of this family group.")
        return
    snap = _db().collection("familyGroups").document(group_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Family group not found.")
    data = snap.to_dict() or {}
    if data.get("ownerUid") != uid:
        raise HTTPException(status_code=403, detail="Not the owner of this family group.")


def _user_owns_any_group(uid: str) -> bool:
    if _skip_firestore():
        for _gid, g in _skip_groups.items():
            if g.get("ownerUid") == uid:
                return True
        return False
    db = _db()
    q = db.collection("familyGroups").where("ownerUid", "==", uid).limit(1)
    return len(list(q.stream())) > 0


def _is_member_of_group(group_id: str, uid: str) -> bool:
    if _skip_firestore():
        g = _skip_groups.get(group_id)
        return bool(g and uid in g.get("members", set()))
    mref = (
        _db()
        .collection("familyGroups")
        .document(group_id)
        .collection("members")
        .document(uid)
    )
    return mref.get().exists


def _delete_relationships_touching(group_id: str, member_uid: str) -> None:
    if _skip_firestore():
        g = _skip_groups[group_id]
        rels: dict[tuple[str, str], str] = g["rels"]
        to_del = [k for k in rels if member_uid in k]
        for k in to_del:
            del rels[k]
        return
    db = _db()
    rel = (
        db.collection("familyGroups")
        .document(group_id)
        .collection("relationships")
    )
    for field, val in (("fromUid", member_uid), ("toUid", member_uid)):
        for doc in rel.where(field, "==", val).stream():
            doc.reference.delete()


def _remove_member_from_group(group_id: str, member_uid: str) -> None:
    if _skip_firestore():
        g = _skip_groups[group_id]
        g["members"].discard(member_uid)
        _delete_relationships_touching(group_id, member_uid)
        if _skip_user_to_group.get(member_uid) == group_id:
            del _skip_user_to_group[member_uid]
        return
    db = _db()
    _delete_relationships_touching(group_id, member_uid)
    db.collection("familyGroups").document(group_id).collection("members").document(
        member_uid
    ).delete()
    set_user_family_group_id(member_uid, None)


def _leave_group_entirely(uid: str, old_group_id: str) -> None:
    """Remove uid from old_group_id (member, edges, user pointer)."""
    if _skip_firestore():
        g = _skip_groups.get(old_group_id)
        if not g:
            return
        if uid in g["members"]:
            _remove_member_from_group(old_group_id, uid)
        return

    db = _db()
    gref = db.collection("familyGroups").document(old_group_id)
    if not gref.get().exists:
        set_user_family_group_id(uid, None)
        return
    owner_uid = (gref.get().to_dict() or {}).get("ownerUid")
    if owner_uid == uid:
        # Cannot remove owner by leaving; caller should avoid this path for owner.
        raise HTTPException(status_code=409, detail="Cannot leave as owner via member removal.")
    _remove_member_from_group(old_group_id, uid)


def create_family_invite(owner_uid: str, target_role: FamilyRole) -> FamilyInviteCreatedOut:
    group_id = _ensure_owner_group_id(owner_uid)
    _assert_owner(owner_uid, group_id)

    token = secrets.token_urlsafe(32)
    expires_at = _now() + timedelta(seconds=_invite_ttl_seconds())
    invite_url = build_invite_url(token)
    qr_png = qr_png_base64_for_url(invite_url)

    if _skip_firestore():
        _skip_invites[token] = {
            "groupId": group_id,
            "ownerUid": owner_uid,
            "targetRole": target_role,
            "expiresAt": expires_at,
            "consumedAt": None,
            "consumedByUid": None,
        }
    else:
        _db().collection("familyInvites").document(token).set(
            {
                "groupId": group_id,
                "ownerUid": owner_uid,
                "targetRole": target_role,
                "expiresAt": expires_at,
                "consumedAt": None,
                "consumedByUid": None,
            }
        )

    return FamilyInviteCreatedOut(
        token=token,
        invite_url=invite_url,
        expires_at=expires_at,
        qr_png_base64=qr_png,
        qr_media_type="image/png",
        target_role=target_role,
    )


def accept_family_invite(invitee_uid: str, token: str) -> AcceptFamilyInviteOut:
    if _user_owns_any_group(invitee_uid):
        raise HTTPException(
            status_code=409,
            detail="You currently own a family group. Remove or transfer ownership before joining another.",
        )

    group_id: str
    owner_uid: str
    target_role_raw: str

    if _skip_firestore():
        inv = _skip_invites.get(token)
        if not inv:
            raise HTTPException(status_code=404, detail="Invite not found.")
        exp: datetime = inv["expiresAt"]
        if _now() > exp:
            raise HTTPException(status_code=410, detail="Invite expired.")
        if inv.get("consumedAt"):
            raise HTTPException(status_code=409, detail="Invite already used.")
        group_id = inv["groupId"]
        owner_uid = inv["ownerUid"]
        target_role_raw = inv["targetRole"]
        if invitee_uid == owner_uid:
            raise HTTPException(status_code=400, detail="Cannot accept your own invite.")
    else:
        ref = _db().collection("familyInvites").document(token)
        snap = ref.get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail="Invite not found.")
        data = snap.to_dict() or {}
        exp = data.get("expiresAt")
        if not isinstance(exp, datetime):
            raise HTTPException(status_code=500, detail="Invalid invite document.")
        if _now() > exp:
            raise HTTPException(status_code=410, detail="Invite expired.")
        if data.get("consumedAt"):
            raise HTTPException(status_code=409, detail="Invite already used.")
        gid = data.get("groupId")
        ou = data.get("ownerUid")
        tr = data.get("targetRole")
        if not isinstance(gid, str) or not isinstance(ou, str):
            raise HTTPException(status_code=500, detail="Invalid invite document.")
        if not isinstance(tr, str):
            raise HTTPException(status_code=500, detail="Invalid invite document.")
        group_id = gid
        owner_uid = ou
        target_role_raw = tr
        if invitee_uid == owner_uid:
            raise HTTPException(status_code=400, detail="Cannot accept your own invite.")

    target_fr = _as_family_role(target_role_raw)

    if _is_member_of_group(group_id, invitee_uid):
        raise HTTPException(status_code=409, detail="Already a member of this family group.")

    old_gid: str | None
    if _skip_firestore():
        old_gid = _skip_user_to_group.get(invitee_uid)
    else:
        old_gid = get_user_family_group_id(invitee_uid)

    if old_gid and old_gid != group_id:
        _leave_group_entirely(invitee_uid, old_gid)

    owner_profile = get_user_profile(owner_uid)
    invitee_profile = get_user_profile(invitee_uid)
    og = owner_profile.gender.value if owner_profile.gender else None
    ig = invitee_profile.gender.value if invitee_profile.gender else None
    inv_to_owner = reciprocal_role(target_fr, owner_gender=og, invitee_gender=ig)

    if _skip_firestore():
        inv = _skip_invites[token]
        g = _skip_groups[group_id]
        g["members"].add(invitee_uid)
        _skip_user_to_group[invitee_uid] = group_id
        g["rels"][(owner_uid, invitee_uid)] = target_fr
        g["rels"][(invitee_uid, owner_uid)] = inv_to_owner
        inv["consumedAt"] = _now()
        inv["consumedByUid"] = invitee_uid
    else:
        db = _db()
        now = _now()
        inv_ref = db.collection("familyInvites").document(token)
        gref = db.collection("familyGroups").document(group_id)
        mref = gref.collection("members").document(invitee_uid)
        rcol = gref.collection("relationships")
        rid1 = f"{owner_uid}__{invitee_uid}"
        rid2 = f"{invitee_uid}__{owner_uid}"

        @firestore.transactional
        def _run_accept(
            transaction: firestore.Transaction,
        ) -> None:
            snap = inv_ref.get(transaction=transaction)
            if not snap.exists:
                raise _InviteFlowError(404, "Invite not found.")
            d = snap.to_dict() or {}
            exp2 = d.get("expiresAt")
            if isinstance(exp2, datetime) and _now() > exp2:
                raise _InviteFlowError(410, "Invite expired.")
            if d.get("consumedAt"):
                raise _InviteFlowError(409, "Invite already used.")
            transaction.update(
                inv_ref,
                {
                    "consumedAt": now,
                    "consumedByUid": invitee_uid,
                },
            )
            transaction.set(mref, {"joinedAt": now})
            transaction.set(
                rcol.document(rid1),
                {
                    "fromUid": owner_uid,
                    "toUid": invitee_uid,
                    "role": target_fr,
                    "createdAt": now,
                },
            )
            transaction.set(
                rcol.document(rid2),
                {
                    "fromUid": invitee_uid,
                    "toUid": owner_uid,
                    "role": inv_to_owner,
                    "createdAt": now,
                },
            )
            transaction.update(gref, {"updatedAt": now})

        try:
            _run_accept(db.transaction())
        except _InviteFlowError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e

        set_user_family_group_id(invitee_uid, group_id)

    family = get_family_group(invitee_uid)
    return AcceptFamilyInviteOut(group_id=family.group_id, family=family)


def _member_display_name(uid: str) -> str | None:
    return get_user_profile(uid).display_name


def get_family_group(uid: str) -> FamilyGroupOut:
    if _skip_firestore():
        gid = _skip_user_to_group.get(uid)
        if not gid:
            raise HTTPException(status_code=404, detail="Not a member of any family group.")
        g = _skip_groups.get(gid)
        if not g or uid not in g["members"]:
            raise HTTPException(status_code=404, detail="Not a member of any family group.")
        owner_uid = g["ownerUid"]
        rels: dict[tuple[str, str], str] = g["rels"]
        members_out: list[FamilyMemberOut] = []
        for m in sorted(g["members"]):
            if m == uid:
                continue
            role = rels.get((uid, m))
            members_out.append(
                FamilyMemberOut(
                    uid=m,
                    display_name=_member_display_name(m),
                    role_relative_to_me=role,  # type: ignore[arg-type]
                )
            )
        return FamilyGroupOut(group_id=gid, owner_uid=owner_uid, members=members_out)

    gid = get_user_family_group_id(uid)
    if not gid:
        raise HTTPException(status_code=404, detail="Not a member of any family group.")
    gref = _db().collection("familyGroups").document(gid)
    gsnap = gref.get()
    if not gsnap.exists:
        raise HTTPException(status_code=404, detail="Family group not found.")
    gdata = gsnap.to_dict() or {}
    owner_uid = gdata.get("ownerUid")
    if not isinstance(owner_uid, str):
        raise HTTPException(status_code=500, detail="Invalid family group.")

    mem_docs = gref.collection("members").stream()
    member_ids = [d.id for d in mem_docs]
    if uid not in member_ids:
        raise HTTPException(status_code=404, detail="Not a member of any family group.")

    rels: dict[tuple[str, str], str] = {}
    for doc in gref.collection("relationships").stream():
        d = doc.to_dict() or {}
        fu, tu, r = d.get("fromUid"), d.get("toUid"), d.get("role")
        if isinstance(fu, str) and isinstance(tu, str) and isinstance(r, str):
            rels[(fu, tu)] = r

    members_out: list[FamilyMemberOut] = []
    for m in sorted(member_ids):
        if m == uid:
            continue
        members_out.append(
            FamilyMemberOut(
                uid=m,
                display_name=_member_display_name(m),
                role_relative_to_me=rels.get((uid, m)),  # type: ignore[arg-type]
            )
        )
    return FamilyGroupOut(group_id=gid, owner_uid=owner_uid, members=members_out)


def remove_family_member(uid: str, member_uid: str) -> RemoveFamilyMemberOut:
    if uid == member_uid:
        raise HTTPException(status_code=400, detail="Cannot remove yourself.")

    if _skip_firestore():
        gid = _skip_user_to_group.get(uid)
    else:
        gid = get_user_family_group_id(uid)

    if not gid:
        raise HTTPException(status_code=404, detail="Not a member of any family group.")

    _assert_owner(uid, gid)

    if _skip_firestore():
        g = _skip_groups[gid]
        if member_uid == g.get("ownerUid"):
            raise HTTPException(status_code=403, detail="Cannot remove the group owner.")
        if member_uid not in g["members"]:
            raise HTTPException(status_code=404, detail="Member not in this group.")
        _remove_member_from_group(gid, member_uid)
        return RemoveFamilyMemberOut(removed_uid=member_uid, group_id=gid)

    snap = _db().collection("familyGroups").document(gid).get()
    data = snap.to_dict() or {}
    if data.get("ownerUid") == member_uid:
        raise HTTPException(status_code=403, detail="Cannot remove the group owner.")
    mref = (
        _db()
        .collection("familyGroups")
        .document(gid)
        .collection("members")
        .document(member_uid)
    )
    if not mref.get().exists:
        raise HTTPException(status_code=404, detail="Member not in this group.")
    _remove_member_from_group(gid, member_uid)
    db = _db()
    db.collection("familyGroups").document(gid).update({"updatedAt": _now()})
    return RemoveFamilyMemberOut(removed_uid=member_uid, group_id=gid)
