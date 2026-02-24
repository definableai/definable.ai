"""Tests for element_refs — role-based ref system."""

from __future__ import annotations

import pytest

from definable.browser.element_refs import (
  CONTENT_ROLES,
  INTERACTIVE_ROLES,
  STRUCTURAL_ROLES,
  RefResolver,
  RoleRef,
  RoleSnapshotOptions,
  build_role_snapshot,
  is_ref,
  parse_role_ref,
)


class TestBuildRoleSnapshot:
  def test_basic_assignment(self):
    aria = '- button "Submit"\n- textbox "Email"'
    result = build_role_snapshot(aria)
    assert "[ref=e1]" in result.snapshot
    assert "[ref=e2]" in result.snapshot
    assert "e1" in result.refs
    assert "e2" in result.refs
    assert result.refs["e1"].role == "button"
    assert result.refs["e1"].name == "Submit"
    assert result.refs["e2"].role == "textbox"
    assert result.refs["e2"].name == "Email"

  def test_heading_gets_ref(self):
    aria = '- heading "Welcome" [level=1]'
    result = build_role_snapshot(aria)
    assert "e1" in result.refs
    assert result.refs["e1"].role == "heading"
    assert result.refs["e1"].name == "Welcome"

  def test_unnamed_heading_no_ref(self):
    aria = "- heading [level=1]"
    result = build_role_snapshot(aria)
    # heading without name doesn't get ref (content role needs name)
    assert len(result.refs) == 0

  def test_structural_no_ref(self):
    aria = '- generic\n  - button "Click"'
    result = build_role_snapshot(aria)
    # Only button should get ref
    assert len(result.refs) == 1
    assert result.refs["e1"].role == "button"

  def test_duplicate_role_name_gets_nth(self):
    aria = '- button "Save"\n- button "Save"'
    result = build_role_snapshot(aria)
    assert result.refs["e1"].nth is not None or result.refs["e2"].nth is not None
    # Both should be in refs
    assert len(result.refs) == 2

  def test_unique_role_name_no_nth(self):
    aria = '- button "Submit"\n- button "Cancel"'
    result = build_role_snapshot(aria)
    assert result.refs["e1"].nth is None
    assert result.refs["e2"].nth is None

  def test_interactive_only_mode(self):
    aria = '- heading "Title" [level=1]\n- button "Click"\n- generic\n  - link "Home"'
    opts = RoleSnapshotOptions(interactive=True)
    result = build_role_snapshot(aria, opts)
    # Only interactive roles: button, link
    roles = {r.role for r in result.refs.values()}
    assert roles == {"button", "link"}

  def test_compact_mode(self):
    aria = '- generic\n  - button "Click" [ref=e1]\n- generic\n  - group'
    opts = RoleSnapshotOptions(compact=True)
    result = build_role_snapshot(aria, opts)
    # Compact should keep lines with refs but remove empty structural nodes
    assert "[ref=" in result.snapshot

  def test_max_depth(self):
    aria = '- group\n  - group\n    - button "Deep"'
    opts = RoleSnapshotOptions(max_depth=1)
    result = build_role_snapshot(aria, opts)
    # Button at depth 2 should be excluded
    assert "Deep" not in result.snapshot

  def test_stats(self):
    aria = '- button "A"\n- textbox "B"\n- heading "C" [level=1]'
    result = build_role_snapshot(aria)
    assert result.stats.refs == 3
    assert result.stats.interactive == 2  # button + textbox

  def test_empty_input(self):
    result = build_role_snapshot("")
    assert result.snapshot == "(empty)" or result.snapshot == ""
    assert len(result.refs) == 0

  def test_preserves_suffix(self):
    aria = '- heading "Welcome" [level=1]'
    result = build_role_snapshot(aria)
    assert "[level=1]" in result.snapshot

  def test_all_interactive_roles_get_refs(self):
    """Every INTERACTIVE_ROLE should get a ref when it appears."""
    for role in sorted(INTERACTIVE_ROLES):
      aria = f'- {role} "Test"'
      result = build_role_snapshot(aria)
      assert len(result.refs) >= 1, f"Role {role} should get a ref"


class TestParseRoleRef:
  def test_simple(self):
    assert parse_role_ref("e1") == "e1"

  def test_at_prefix(self):
    assert parse_role_ref("@e42") == "e42"

  def test_ref_equals(self):
    assert parse_role_ref("ref=e3") == "e3"

  def test_not_a_ref(self):
    assert parse_role_ref("button.submit") is None

  def test_empty(self):
    assert parse_role_ref("") is None

  def test_just_e(self):
    assert parse_role_ref("e") is None


class TestIsRef:
  def test_valid(self):
    assert is_ref("e1") is True
    assert is_ref("e42") is True
    assert is_ref("e999") is True

  def test_invalid(self):
    assert is_ref("button") is False
    assert is_ref("e") is False
    assert is_ref("E1") is False
    assert is_ref("e1a") is False
    assert is_ref("#submit") is False


class TestRefResolver:
  def test_resolve_ref(self):
    from tests.unit.browser.conftest import MockPage

    resolver = RefResolver()
    resolver.store({"e1": RoleRef(role="button", name="Submit")})
    page = MockPage()
    locator = resolver.resolve(page, "e1")
    assert locator is not None

  def test_resolve_css_selector(self):
    from tests.unit.browser.conftest import MockPage

    resolver = RefResolver()
    page = MockPage()
    locator = resolver.resolve(page, "button.submit")
    assert locator is not None

  def test_unknown_ref_raises(self):
    from tests.unit.browser.conftest import MockPage

    resolver = RefResolver()
    resolver.store({})
    page = MockPage()
    with pytest.raises(ValueError, match="Unknown ref"):
      resolver.resolve(page, "e99")

  def test_at_prefix_stripped(self):
    from tests.unit.browser.conftest import MockPage

    resolver = RefResolver()
    resolver.store({"e1": RoleRef(role="button", name="OK")})
    page = MockPage()
    locator = resolver.resolve(page, "@e1")
    assert locator is not None

  def test_ref_equals_stripped(self):
    from tests.unit.browser.conftest import MockPage

    resolver = RefResolver()
    resolver.store({"e5": RoleRef(role="link", name="Home")})
    page = MockPage()
    locator = resolver.resolve(page, "ref=e5")
    assert locator is not None

  def test_store_replaces(self):
    resolver = RefResolver()
    resolver.store({"e1": RoleRef(role="button", name="A")})
    resolver.store({"e1": RoleRef(role="link", name="B")})
    assert resolver.refs["e1"].role == "link"


class TestRoleClassification:
  def test_no_overlap(self):
    """Role sets should not overlap."""
    assert not (INTERACTIVE_ROLES & CONTENT_ROLES)
    assert not (INTERACTIVE_ROLES & STRUCTURAL_ROLES)
    assert not (CONTENT_ROLES & STRUCTURAL_ROLES)

  def test_button_is_interactive(self):
    assert "button" in INTERACTIVE_ROLES

  def test_heading_is_content(self):
    assert "heading" in CONTENT_ROLES

  def test_generic_is_structural(self):
    assert "generic" in STRUCTURAL_ROLES
