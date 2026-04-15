from __future__ import annotations

from fastapi import APIRouter, Depends

from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    AcceptFamilyInviteIn,
    AcceptFamilyInviteOut,
    CreateFamilyInviteIn,
    FamilyGroupOut,
    FamilyInviteCreatedOut,
    RemoveFamilyMemberOut,
)
from fambot_backend.services.family_invites import (
    accept_family_invite,
    create_family_invite,
    get_family_group,
    remove_family_member,
)

router = APIRouter(prefix="/me/family", tags=["me", "family"])


@router.post("/invitations", response_model=FamilyInviteCreatedOut)
def create_invitation(
    body: CreateFamilyInviteIn,
    uid: str = Depends(firebase_uid),
) -> FamilyInviteCreatedOut:
    return create_family_invite(uid, body.target_role)


@router.post("/invitations/accept", response_model=AcceptFamilyInviteOut)
def accept_invitation(
    body: AcceptFamilyInviteIn,
    uid: str = Depends(firebase_uid),
) -> AcceptFamilyInviteOut:
    return accept_family_invite(uid, body.token)


@router.get("", response_model=FamilyGroupOut)
def read_family(uid: str = Depends(firebase_uid)) -> FamilyGroupOut:
    return get_family_group(uid)


@router.delete("/members/{member_uid}", response_model=RemoveFamilyMemberOut)
def delete_family_member(
    member_uid: str,
    uid: str = Depends(firebase_uid),
) -> RemoveFamilyMemberOut:
    return remove_family_member(uid, member_uid)
