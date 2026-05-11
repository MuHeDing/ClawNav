# SpatialMemory Protocol

Protocol version: `phase2.spatial_memory.v1`

This protocol defines the local HTTP-compatible contract used by the ClawNav
harness when a SpatialMemory service is available. The harness remains runnable
without that service by using `FakeSpatialMemoryClient`.

## Endpoints

GET /health

POST /query/object

POST /query/place

POST /query/position

POST /query/semantic/text

POST /query/unified

POST /memory/semantic/ingest

## Query Request

Every query request should include:

```json
{
  "protocol_version": "phase2.spatial_memory.v1",
  "query_type": "semantic",
  "text": "kitchen entrance",
  "n_results": 3,
  "memory_source": "episode-local"
}
```

Position queries use `pose` instead of `text`:

```json
{
  "protocol_version": "phase2.spatial_memory.v1",
  "query_type": "position",
  "pose": {"x": 1.0, "y": 2.0},
  "n_results": 5,
  "memory_source": "scene-prior"
}
```

## Required Result Fields

The service response should contain a `results` array. Each result is normalized
by the harness into these fields:

```text
memory_id, memory_type, name, confidence, target_pose, evidence_text,
image_path, memory_source, metadata
```

Example response:

```json
{
  "results": [
    {
      "id": "place-001",
      "memory_type": "place",
      "name": "kitchen",
      "confidence": 0.82,
      "target_pose": {"x": 1.0, "y": 2.0},
      "evidence": {
        "text": "kitchen entrance near hallway",
        "image_path": "/memory/keyframes/kitchen.jpg"
      },
      "metadata": {
        "floor": "1"
      }
    }
  ]
}
```

## Memory Sources

`memory_source` must describe where the memory came from:

```text
episode-local
scene-prior
train-scene-only
```

Evaluation reports should state which source was used. Online evaluation must
not silently mix these sources.

## No Oracle Fields

The service must not return `distance_to_goal`, `success`, `SPL`, oracle path,
oracle shortest path action, or future observations in query results used by
online controller logic.

If such fields are present in metadata, the harness protocol normalization
strips them before constructing `MemoryHit` records.

## Ingest Request

`POST /memory/semantic/ingest` accepts episode-local or approved prior memory
records:

```json
{
  "write_type": "episodic_keyframe",
  "step_id": 12,
  "scene_id": "scene1",
  "episode_id": "episode1",
  "image_path": "/tmp/keyframe-12.jpg",
  "note": "turning point near sofa",
  "memory_source": "episode-local",
  "metadata": {
    "active_subgoal": "turn left at sofa"
  }
}
```
