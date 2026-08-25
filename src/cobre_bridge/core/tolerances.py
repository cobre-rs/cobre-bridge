"""The shared relative-tolerance idiom for comparing a float against a
declared/reference value, mirroring cobre-io's semantic-validation
``ENVELOPE_TOLERANCE``.
"""

from __future__ import annotations

#: Mirrors cobre-io's semantic-validation ``ENVELOPE_TOLERANCE``. A relative
#: tolerance, not an absolute one — an absolute epsilon would false-fire on a
#: plant declared at, say, 1e6 m^3/s and float-noise-pass a plant near zero.
RELATIVE_TOLERANCE: float = 1e-9


def relative_tolerance(reference: float) -> float:
    """The absolute magnitude ``RELATIVE_TOLERANCE * max(|reference|, 1.0)``
    for a comparison against *reference*.
    """
    return RELATIVE_TOLERANCE * max(abs(reference), 1.0)


def floats_differ(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* past relative float noise."""
    return abs(value - reference) > relative_tolerance(reference)
