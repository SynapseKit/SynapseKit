# Seeded user simulation

The simulation harness exercises an async agent and simulated user against the
same mutable environment. It does not invoke a model or network service; any
such calls in actor callbacks are the caller's responsibility.

From the repository root, run the offline example and tests (with SynapseKit
and pytest installed):

```sh
PYTHONPATH=src python examples/user_simulation.py
PYTHONPATH=src python -m pytest -p no:cacheprovider tests/evaluation/test_simulation.py
```

The example imports `synapsekit.evaluation.simulation` directly. An episode
starts with the user, then alternates actors; `max_turns` counts individual
user **and** agent actions. `run(seed=19)` resets environment state first and
returns a `SimulationResult` containing the seed, termination reason, ordered
pre-action observation/action pairs, reward and any error. If `apply` raises
`InvalidActionError`, the default `error_policy="raise"` propagates it. With
`error_policy="terminate"`, the rejected action and error are recorded, the
reason is `invalid_action`, and the environment supplies the reward. Exceptions
from actors, observations, resets, rewards, or other errors propagate in both
modes; cancellation is not swallowed.

The transcript deep-copies each observation before the actor receives it and
each action before `apply` receives it, including rejected actions. Callbacks
still receive the original objects; copying is only for recording. Copy errors
propagate in both error policies before the next callback runs. Transcript
values are not recursively immutable: callers can still mutate their own copies.

## Booking domain

Each `BookingObservation` is an immutable snapshot. The user may `request`
an available slot, `consent`, `cancel` the request or existing booking, or
`wait`. Changing a request clears consent. The agent may `message`, `book`
the requested slot after consent, `finish` after booking, or `wait`. An invalid
action is rejected. Booking without consent, booking the wrong slot, and
finishing without a booking additionally record policy violations, even if the
caller later continues an environment manually. The policy awards `1.0` only
when the episode finishes with a matching consented booking and no violations;
otherwise it awards `0.0`. Text such as “success” does not affect the score.
This is a toy in-memory calendar, not a production scheduler or a security
boundary.

## Custom environments and actors

Implement the generic `SimulationEnvironment[Observation, Action]` protocol:
async `reset(seed)`, `observe(actor)`, `apply(actor, action) -> bool` (true
when finished), and `reward() -> Reward`. Return independent snapshots rather
than references to mutable state; raise `InvalidActionError` for invalid actions.
Observations and actions must support `copy.deepcopy` with independent copies
of mutable contents; custom `__deepcopy__` implementations must honor that
contract. Non-copyable resources should not be included in transcript values.
Provide two async callables `(observation, rng) -> action` to
`SimulationHarness`. Each receives a separate `random.Random` stream derived
from the episode seed. A custom environment should use its reset seed for its
own randomness. Reproducibility requires deterministic callbacks and environment
logic that use those streams, with no unseeded randomness, external state, or
nondeterministic service responses. Reusing a harness sequentially resets the
environment but does **not** reset state retained inside your callbacks. Use a
separate environment and actors per concurrent episode; the harness does not
serialize concurrent runs.
