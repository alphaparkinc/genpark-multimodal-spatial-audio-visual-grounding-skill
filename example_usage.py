"""Example usage for SpatialAudioVisualGroundingEngine."""
import json
from client import SpatialAudioVisualGroundingEngine

def main():
    print("=== Multimodal Spatial Audio-Visual Grounding Engine Demo ===")
    engine = SpatialAudioVisualGroundingEngine(horizontal_fov_deg=70.0)

    # 1. Multi-modal frame simulation (Meta Ray-Ban / Muse smart glasses)
    detections = [
        {"id": "v1", "label": "espresso_machine", "bbox": [0.48, 0.45, 0.15, 0.25], "confidence": 0.92},
        {"id": "v2", "label": "colleague_alice", "bbox": [0.15, 0.30, 0.20, 0.50], "confidence": 0.88},
        {"id": "v3", "label": "laptop_screen", "bbox": [0.75, 0.50, 0.22, 0.30], "confidence": 0.96}
    ]
    audio_streams = [
        {"label": "hissing_steam", "azimuth_deg": 1.5, "energy_db": -14.0},
        {"label": "alice_speech", "azimuth_deg": -22.0, "energy_db": -18.5}
    ]
    head_pose = {"yaw": 0.0, "pitch": -2.0, "roll": 0.0}

    print("\n--- 1. Fusing Visual & Spatial Audio Sensors ---")
    frame = engine.fuse_sensory_input(detections, audio_streams, head_pose)
    print(f"Top Focal Salience Target: {frame['focal_entity']['label']} (Score: {frame['focal_entity']['salience_score']})")
    print(json.dumps(frame["focal_entity"], indent=2))

    # 2. Resolve deictic reference
    print("\n--- 2. Deictic Reference Resolution ('What is that hissing noise?') ---")
    res1 = engine.resolve_deictic_reference("What is that hissing noise right in front of me?")
    print(f"Resolved Entity: {res1['grounded_entity']['label']}, Guidance: {res1['spatial_guidance']}")

    print("\n--- 3. Deictic Reference Resolution ('Who is speaking on my left?') ---")
    res2 = engine.resolve_deictic_reference("Who is speaking on my left?")
    print(f"Resolved Entity: {res2['grounded_entity']['label']}, Guidance: {res2['spatial_guidance']}")

if __name__ == "__main__":
    main()
