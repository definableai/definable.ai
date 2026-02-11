"""
MemoryStore protocol walkthrough using InMemoryStore.

Exercises all 19 protocol methods with assertions and printed output.
No external dependencies or API keys required.

Usage:
    python definable/examples/memory/02_store_protocol.py
"""

import asyncio
import time

from definable.memory import InMemoryStore
from definable.memory.types import Episode, KnowledgeAtom, Procedure


async def main():
  store = InMemoryStore()
  await store.initialize()
  print("InMemoryStore initialized\n")

  # ------------------------------------------------------------------
  # 1. Episodes
  # ------------------------------------------------------------------
  print("=" * 60)
  print("EPISODES")
  print("=" * 60)

  now = time.time()
  episodes = [
    Episode(
      id="ep-1",
      user_id="alice",
      session_id="sess-1",
      role="user",
      content="Hi, I'm Alice!",
      embedding=[1.0, 0.0, 0.0],
      topics=["greeting"],
      compression_stage=0,
      created_at=now - 300,
    ),
    Episode(
      id="ep-2",
      user_id="alice",
      session_id="sess-1",
      role="assistant",
      content="Hello Alice! How can I help?",
      embedding=[0.9, 0.1, 0.0],
      topics=["greeting"],
      compression_stage=0,
      created_at=now - 200,
    ),
    Episode(
      id="ep-3",
      user_id="alice",
      session_id="sess-2",
      role="user",
      content="Tell me about Python decorators.",
      embedding=[0.0, 1.0, 0.0],
      topics=["python", "decorators"],
      compression_stage=1,
      created_at=now - 100,
    ),
    Episode(
      id="ep-4",
      user_id="bob",
      session_id="sess-3",
      role="user",
      content="What is Rust?",
      embedding=[0.0, 0.0, 1.0],
      topics=["rust"],
      compression_stage=0,
      created_at=now - 50,
    ),
  ]

  for ep in episodes:
    eid = await store.store_episode(ep)
    print(f"  stored episode {eid!r}")

  # get_episodes — filter by user
  alice_eps = await store.get_episodes(user_id="alice")
  assert len(alice_eps) == 3, f"Expected 3 alice episodes, got {len(alice_eps)}"
  print(f"\n  alice episodes: {len(alice_eps)}")

  # filter by session
  sess1_eps = await store.get_episodes(session_id="sess-1")
  assert len(sess1_eps) == 2
  print(f"  sess-1 episodes: {len(sess1_eps)}")

  # filter by compression stage
  stage0 = await store.get_episodes(user_id="alice", max_stage=0)
  assert len(stage0) == 2
  print(f"  alice stage-0 episodes: {len(stage0)}")

  # limit
  limited = await store.get_episodes(user_id="alice", limit=1)
  assert len(limited) == 1
  print(f"  alice episodes (limit=1): {len(limited)}")

  # update_episode
  await store.update_episode("ep-1", content="Hi, I'm Alice (updated)!")
  updated = await store.get_episodes(session_id="sess-1", limit=50)
  ep1 = next(e for e in updated if e.id == "ep-1")
  assert "updated" in ep1.content
  print(f"  updated ep-1 content: {ep1.content!r}")

  # get_episodes_for_distillation — stage 0, older than (now - 60s)
  distillable = await store.get_episodes_for_distillation(stage=0, older_than=now - 60)
  assert len(distillable) == 2, f"Expected 2 distillable, got {len(distillable)}"
  print(f"  distillable (stage=0, older than 60s): {len(distillable)}")

  print()

  # ------------------------------------------------------------------
  # 2. Knowledge Atoms
  # ------------------------------------------------------------------
  print("=" * 60)
  print("KNOWLEDGE ATOMS")
  print("=" * 60)

  atoms = [
    KnowledgeAtom(
      id="atom-1",
      user_id="alice",
      subject="Alice",
      predicate="lives in",
      object="San Francisco",
      content="Alice lives in San Francisco",
      embedding=[1.0, 0.0, 0.0],
      confidence=0.9,
    ),
    KnowledgeAtom(
      id="atom-2",
      user_id="alice",
      subject="Alice",
      predicate="works as",
      object="Python developer",
      content="Alice works as a Python developer",
      embedding=[0.8, 0.2, 0.0],
      confidence=0.7,
    ),
    KnowledgeAtom(
      id="atom-3",
      user_id="alice",
      subject="Alice",
      predicate="likes",
      object="hiking",
      content="Alice likes hiking",
      embedding=[0.0, 1.0, 0.0],
      confidence=0.05,  # very low — will be pruned
    ),
  ]

  for atom in atoms:
    aid = await store.store_atom(atom)
    print(f"  stored atom {aid!r} (confidence={atom.confidence})")

  # get_atoms — default min_confidence=0.1
  high_conf = await store.get_atoms(user_id="alice")
  assert len(high_conf) == 2, f"Expected 2 high-confidence atoms, got {len(high_conf)}"
  print(f"\n  atoms with confidence >= 0.1: {len(high_conf)}")

  # get_atoms — lower threshold
  all_atoms = await store.get_atoms(user_id="alice", min_confidence=0.0)
  assert len(all_atoms) == 3
  print(f"  atoms with confidence >= 0.0: {len(all_atoms)}")

  # find_similar_atom — hit
  found = await store.find_similar_atom("Alice", "lives in", user_id="alice")
  assert found is not None and found.object == "San Francisco"
  print(f"  find_similar_atom('Alice', 'lives in'): {found.content!r}")

  # find_similar_atom — miss
  missing = await store.find_similar_atom("Bob", "lives in", user_id="alice")
  assert missing is None
  print(f"  find_similar_atom('Bob', 'lives in'): {missing}")

  # update_atom
  await store.update_atom("atom-2", confidence=0.95)
  updated_atom = await store.find_similar_atom("Alice", "works as", user_id="alice")
  assert updated_atom is not None and updated_atom.confidence == 0.95
  print(f"  updated atom-2 confidence: {updated_atom.confidence}")

  # prune_atoms — removes atoms below threshold
  pruned = await store.prune_atoms(min_confidence=0.1)
  assert pruned == 1, f"Expected 1 pruned, got {pruned}"
  remaining = await store.get_atoms(user_id="alice", min_confidence=0.0)
  assert len(remaining) == 2
  print(f"  pruned {pruned} atom(s), remaining: {len(remaining)}")

  print()

  # ------------------------------------------------------------------
  # 3. Procedures
  # ------------------------------------------------------------------
  print("=" * 60)
  print("PROCEDURES")
  print("=" * 60)

  proc = Procedure(
    id="proc-1",
    user_id="alice",
    trigger="user asks for code",
    action="use Python with type hints",
    confidence=0.8,
    observation_count=5,
  )
  pid = await store.store_procedure(proc)
  print(f"  stored procedure {pid!r}")

  # get_procedures
  procs = await store.get_procedures(user_id="alice")
  assert len(procs) == 1
  print(f"  procedures for alice: {len(procs)}")

  # find_similar_procedure — hit (shares words "code" and "asks")
  similar = await store.find_similar_procedure("user asks about code examples", user_id="alice")
  assert similar is not None
  print(f"  find_similar_procedure('user asks about code examples'): {similar.action!r}")

  # find_similar_procedure — miss
  no_match = await store.find_similar_procedure("weather forecast", user_id="alice")
  assert no_match is None
  print(f"  find_similar_procedure('weather forecast'): {no_match}")

  # update_procedure
  await store.update_procedure("proc-1", confidence=0.95, observation_count=10)
  procs = await store.get_procedures(user_id="alice")
  assert procs[0].confidence == 0.95 and procs[0].observation_count == 10
  print(f"  updated proc-1: confidence={procs[0].confidence}, observations={procs[0].observation_count}")

  print()

  # ------------------------------------------------------------------
  # 4. Topic Transitions
  # ------------------------------------------------------------------
  print("=" * 60)
  print("TOPIC TRANSITIONS")
  print("=" * 60)

  # Store several transitions so counts exceed min_count=3
  for _ in range(5):
    await store.store_topic_transition("greeting", "python", user_id="alice")
  for _ in range(3):
    await store.store_topic_transition("greeting", "rust", user_id="alice")
  # Only 2 times — should NOT appear with default min_count=3
  for _ in range(2):
    await store.store_topic_transition("greeting", "java", user_id="alice")

  transitions = await store.get_topic_transitions("greeting", user_id="alice")
  assert len(transitions) == 2, f"Expected 2 transitions (>=3 count), got {len(transitions)}"
  print(f"  transitions from 'greeting' (min_count=3): {len(transitions)}")
  for t in transitions:
    print(f"    -> {t.to_topic}: count={t.count}, probability={t.probability:.2f}")

  # Verify probabilities sum to 1.0
  total_prob = sum(t.probability for t in transitions)
  assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, expected ~1.0"
  print(f"  probability sum: {total_prob:.2f}")

  print()

  # ------------------------------------------------------------------
  # 5. Vector Search
  # ------------------------------------------------------------------
  print("=" * 60)
  print("VECTOR SEARCH")
  print("=" * 60)

  # search_episodes_by_embedding — query close to ep-1 [1,0,0]
  results = await store.search_episodes_by_embedding([0.95, 0.05, 0.0], user_id="alice")
  assert len(results) > 0
  assert results[0].id == "ep-1", f"Expected ep-1 first, got {results[0].id}"
  print(f"  episode search near [0.95, 0.05, 0]: top result = {results[0].id} ({results[0].content!r})")

  # search_atoms_by_embedding — query close to atom-1 [1,0,0]
  atom_results = await store.search_atoms_by_embedding([0.9, 0.1, 0.0], user_id="alice")
  assert len(atom_results) > 0
  assert atom_results[0].id == "atom-1"
  print(f"  atom search near [0.9, 0.1, 0]: top result = {atom_results[0].id} ({atom_results[0].content!r})")

  print()

  # ------------------------------------------------------------------
  # 6. Deletion
  # ------------------------------------------------------------------
  print("=" * 60)
  print("DELETION")
  print("=" * 60)

  # delete_session_data — removes sess-1 episodes
  await store.delete_session_data("sess-1")
  sess1_after = await store.get_episodes(session_id="sess-1")
  assert len(sess1_after) == 0
  print(f"  after delete_session_data('sess-1'): {len(sess1_after)} episodes")

  # delete_user_data — removes all alice data
  await store.delete_user_data("alice")
  alice_after = await store.get_episodes(user_id="alice")
  alice_atoms_after = await store.get_atoms(user_id="alice", min_confidence=0.0)
  alice_procs_after = await store.get_procedures(user_id="alice", min_confidence=0.0)
  assert len(alice_after) == 0
  assert len(alice_atoms_after) == 0
  assert len(alice_procs_after) == 0
  print(f"  after delete_user_data('alice'): episodes={len(alice_after)}, atoms={len(alice_atoms_after)}, procedures={len(alice_procs_after)}")

  # Bob's data should still exist
  bob_eps = await store.get_episodes(user_id="bob")
  assert len(bob_eps) == 1
  print(f"  bob's data intact: {len(bob_eps)} episode(s)")

  await store.close()
  print("\nAll 19 protocol methods verified successfully!")


if __name__ == "__main__":
  asyncio.run(main())
