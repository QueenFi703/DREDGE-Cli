"""Atomic owner-scoped exact-input reservations for existing live Studio jobs."""
from copy import deepcopy
import hashlib
import json

from .studio_provider import INSTRUCTIONS

DEDUP_WINDOW_SECONDS = 86400


def execution_fingerprint(payload):
    # Preserve all proposal fields, including model/options, declared data scope,
    # consent, source order/content/versions and any future context boundaries.
    # Only attachment bookkeeping time is omitted; it is not sent to the model.
    substantive = deepcopy(payload)
    for source in substantive.get('evidence', []):
        source.pop('attached_at', None)
    identity = {'version': 1, 'payload': substantive, 'instructions': INSTRUCTIONS,
                'provider_options': {'store': False, 'service_tier': 'default',
                                     'reasoning': {'effort': 'low'}, 'tools': []}}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def reserve_live_input(db, owner, run_id, payload, now):
    """Caller must hold BEGIN IMMEDIATE until reservation and running commit.

    Return the same owner's previous run ID on a duplicate. No token/cost savings
    are inferred; a blocked request is not evidence it would have completed.
    Unknown provider outcomes retain the reservation for the entire window.
    """
    if not db.in_transaction:
        raise RuntimeError('Live dedup requires an active write transaction.')
    fingerprint = execution_fingerprint(payload)
    prior = db.execute('SELECT run_id,reserved_at FROM studio_live_dedup '
                       'WHERE owner=? AND fingerprint=?', (owner, fingerprint)).fetchone()
    if prior and prior['reserved_at'] >= now - DEDUP_WINDOW_SECONDS:
        return prior['run_id']
    db.execute('INSERT INTO studio_live_dedup(owner,fingerprint,run_id,reserved_at) '
               'VALUES(?,?,?,?) ON CONFLICT(owner,fingerprint) DO UPDATE SET '
               'run_id=excluded.run_id,reserved_at=excluded.reserved_at',
               (owner, fingerprint, run_id, now))
    return None
