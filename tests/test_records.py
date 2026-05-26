"""Tests for PartnershipRecord."""

import pytest

from partnersim_dynet.generator import PartnershipRecord


class TestPartnershipRecordConstruction:
    def test_full_record_construction(self):
        """A normal dissolved partnership has all fields populated."""
        rec = PartnershipRecord(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=2,
            partner_sex="Female",
            partner_orientation="Opposite-sex",
            partner_age=28,
            start_time=100,
            end_time=500,
            duration=400,
            relationship_type="M-F",
            censored=False,
            external_partner=False,
        )
        assert rec.agent_id == 1
        assert rec.partner_id == 2
        assert rec.duration == 400
        assert rec.relationship_type == "M-F"
        assert rec.censored is False

    def test_single_record_construction(self):
        """An agent who never partnered: all partner fields are None."""
        rec = PartnershipRecord(
            agent_id=7,
            agent_sex="Female",
            agent_orientation="Same-sex",
            agent_age=22,
            partner_id=None,
            partner_sex=None,
            partner_orientation=None,
            partner_age=None,
            start_time=None,
            end_time=None,
            duration=None,
            relationship_type="None",
            censored=False,
            external_partner=False,
        )
        assert rec.partner_id is None
        assert rec.relationship_type == "None"

    def test_censored_record(self):
        """A partnership still active at end-of-simulation."""
        rec = PartnershipRecord(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=2,
            partner_sex="Female",
            partner_orientation="Opposite-sex",
            partner_age=28,
            start_time=1500,
            end_time=1875,
            duration=375,
            relationship_type="M-F",
            censored=True,
            external_partner=False,
        )
        assert rec.censored is True


class TestSlotsBehaviour:
    """The slots=True optimisation should be active."""

    def test_no_dict_attribute(self):
        """Slotted dataclasses don't have __dict__ — that's the whole point."""
        rec = PartnershipRecord(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=None,
            partner_sex=None,
            partner_orientation=None,
            partner_age=None,
            start_time=None,
            end_time=None,
            duration=None,
            relationship_type="None",
            censored=False,
            external_partner=False,
        )
        assert not hasattr(rec, "__dict__")

    def test_cannot_add_arbitrary_attribute(self):
        """Slots prevents attribute injection — catches typos that
        would silently fail on regular classes."""
        rec = PartnershipRecord(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=None,
            partner_sex=None,
            partner_orientation=None,
            partner_age=None,
            start_time=None,
            end_time=None,
            duration=None,
            relationship_type="None",
            censored=False,
            external_partner=False,
        )
        with pytest.raises(AttributeError):
            rec.nonexistent_field = 42  # type: ignore[misc]


class TestDataclassFeatures:
    def test_equality(self):
        """Two records with the same field values should compare equal."""
        kwargs = dict(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=2,
            partner_sex="Female",
            partner_orientation="Opposite-sex",
            partner_age=28,
            start_time=100,
            end_time=500,
            duration=400,
            relationship_type="M-F",
            censored=False,
            external_partner=False,
        )
        a = PartnershipRecord(**kwargs)
        b = PartnershipRecord(**kwargs)
        assert a == b

    def test_repr_contains_field_names(self):
        rec = PartnershipRecord(
            agent_id=1,
            agent_sex="Male",
            agent_orientation="Opposite-sex",
            agent_age=30,
            partner_id=None,
            partner_sex=None,
            partner_orientation=None,
            partner_age=None,
            start_time=None,
            end_time=None,
            duration=None,
            relationship_type="None",
            censored=False,
            external_partner=False,
        )
        r = repr(rec)
        assert "agent_id=1" in r
        assert "relationship_type='None'" in r
