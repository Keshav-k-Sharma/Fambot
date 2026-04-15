from __future__ import annotations

"""Perspective-aware reciprocal family roles (v1 fixed vocabulary).

`owner_to_invitee` is how the group owner describes the person joining.
The return value is how that person (invitee) describes the owner — i.e. the
edge invitee → owner.
"""

from fambot_backend.schemas import FamilyRole


def _child_of_parent(owner_gender: str | None) -> FamilyRole:
    g = (owner_gender or "male").lower()
    return "son" if g == "male" else "daughter"


def _parent_of_child(owner_gender: str | None) -> FamilyRole:
    g = (owner_gender or "male").lower()
    return "father" if g == "male" else "mother"


def _sibling_label_for_peer(owner_gender: str | None) -> FamilyRole:
    g = (owner_gender or "male").lower()
    return "brother" if g == "male" else "sister"


def _nephew_niece(owner_gender: str | None) -> FamilyRole:
    g = (owner_gender or "male").lower()
    return "nephew" if g == "male" else "niece"


def reciprocal_role(
    owner_to_invitee: FamilyRole,
    *,
    owner_gender: str | None,
    invitee_gender: str | None,
) -> FamilyRole:
    """Compute invitee→owner role label given owner→invitee and genders.

    `invitee_gender` is reserved for future refinements; v1 uses owner gender
    for parent/child and sibling symmetry.
    """
    _ = invitee_gender
    o = owner_gender.lower() if owner_gender else None

    if owner_to_invitee == "mother":
        return _child_of_parent(o)
    if owner_to_invitee == "father":
        return _child_of_parent(o)
    if owner_to_invitee == "son":
        return _parent_of_child(o)
    if owner_to_invitee == "daughter":
        return _parent_of_child(o)
    if owner_to_invitee == "brother":
        return _sibling_label_for_peer(o)
    if owner_to_invitee == "sister":
        return _sibling_label_for_peer(o)
    if owner_to_invitee == "uncle":
        return _nephew_niece(o)
    if owner_to_invitee == "aunt":
        return _nephew_niece(o)
    if owner_to_invitee == "nephew":
        g = o or "male"
        return "aunt" if g == "female" else "uncle"
    if owner_to_invitee == "niece":
        g = o or "male"
        return "uncle" if g == "male" else "aunt"
    if owner_to_invitee == "husband":
        return "wife"
    if owner_to_invitee == "wife":
        return "husband"
    raise ValueError(f"unknown family role: {owner_to_invitee!r}")
