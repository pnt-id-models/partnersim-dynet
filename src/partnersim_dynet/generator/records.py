"""Partnership record dataclass.

A single `PartnershipRecord` captures one observation of a partnership at
the moment it either dissolved or was censored at end-of-simulation.
The simulation accumulates these records during the run and generates them
as the final partnership-level output table.

Uses ``slots=True`` because the simulation creates one record per
dissolved or censored partnership — for a 15k-agent, 1875-timestep run
that's typically several hundred thousand instances. Slots remove the
per-instance ``__dict__`` overhead, cutting memory and speeding attribute
access.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PartnershipRecord:
    """One row of partnership-level output.

    Parameters
    ----------
    agent_id : int
        ID of the focal agent.
    agent_sex, agent_orientation : str
        Demographic labels of the focal agent.
    agent_age : int
        Age of the focal agent at the moment the record was created.
    partner_id : int | None
        Partner ID. None if the agent is recorded as never having had a
        partnership during the run (the "single" record case).
    partner_sex, partner_orientation : str | None
        Partner demographic labels. None for single records.
    partner_age : int | None
        Partner's age at the moment the record was created. None for
        single records.
    start_time, end_time : int | None
        Simulation timesteps at which the partnership started and ended.
        Both None for single records. ``end_time`` equals the
        simulation's final timestep for censored partnerships.
    duration : int | None
        ``end_time - start_time``. None for single records.
    relationship_type : str
        One of "M-M", "F-F", "M-F", or "None" for singles.
    censored : bool
        True if the partnership was still active at end-of-simulation
        (its dissolution would have happened later, but the simulation
        ended before observing it).
    external_partner : bool
        True if the partner was removed from the simulation (e.g. aged out
        at MAX_AGE) before this record was created, so the partnership's
        end was triggered by the partner's departure rather than by a
        normal dissolution draw.
    """

    agent_id: int
    agent_sex: str
    agent_orientation: str
    agent_age: int
    partner_id: int | None
    partner_sex: str | None
    partner_orientation: str | None
    partner_age: int | None
    start_time: int | None
    end_time: int | None
    duration: int | None
    relationship_type: str
    censored: bool
    external_partner: bool
