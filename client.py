"""
Multimodal Spatial Audio-Visual Grounding Engine (Zero External Dependencies)
Fuses 2D/3D visual bounding boxes, user head-pose gaze vectors, and acoustic beamforming azimuth angles.
"""
import math
import time
import json
from typing import List, Dict, Any, Optional

class SpatialAudioVisualGroundingEngine:
    def __init__(self, horizontal_fov_deg: float = 70.0, vertical_fov_deg: float = 55.0):
        self.horizontal_fov = horizontal_fov_deg
        self.vertical_fov = vertical_fov_deg
        self.history = []
        self.max_history = 100
        self.current_focus = None

    def _bbox_to_angles(self, bbox: List[float], head_pose: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized bbox [x_min, y_min, x_max, y_max] or [x, y, w, h] to visual azimuth/elevation."""
        if len(bbox) == 4:
            if bbox[2] <= 1.0 and bbox[3] <= 1.0 and bbox[0] + bbox[2] <= 1.5:
                # [x, y, w, h]
                cx = bbox[0] + bbox[2] / 2.0
                cy = bbox[1] + bbox[3] / 2.0
            else:
                # [x1, y1, x2, y2]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
        else:
            cx, cy = 0.5, 0.5

        # Map 0.0 -> -fov/2, 0.5 -> 0.0, 1.0 -> +fov/2
        cam_azimuth = (cx - 0.5) * self.horizontal_fov
        cam_elevation = (0.5 - cy) * self.vertical_fov

        yaw_offset = head_pose.get("yaw", 0.0)
        pitch_offset = head_pose.get("pitch", 0.0)

        world_azimuth = cam_azimuth + yaw_offset
        world_elevation = cam_elevation + pitch_offset
        return {
            "center_x": round(cx, 4),
            "center_y": round(cy, 4),
            "azimuth_deg": round(world_azimuth, 2),
            "elevation_deg": round(world_elevation, 2)
        }

    def _angular_distance(self, az1: float, az2: float) -> float:
        """Compute shortest angular distance between two azimuth angles in [-180, 180]."""
        diff = abs(az1 - az2) % 360.0
        return 360.0 - diff if diff > 180.0 else diff

    def fuse_sensory_input(
        self,
        visual_detections: Optional[List[Dict[str, Any]]] = None,
        audio_events: Optional[List[Dict[str, Any]]] = None,
        head_pose: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Fuses visual detections with spatial acoustic beamforming.
        Computes cross-modal focal salience score (0.0 to 1.0) for each entity.
        """
        visual_detections = visual_detections or []
        audio_events = audio_events or []
        head_pose = head_pose or {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        fused_entities = []

        # 1. Process visual candidates
        for idx, det in enumerate(visual_detections):
            label = det.get("label", f"visual_obj_{idx}")
            bbox = det.get("bbox", [0.4, 0.4, 0.2, 0.2])
            conf = float(det.get("confidence", 0.8))
            spatial_coords = self._bbox_to_angles(bbox, head_pose)

            # Center gaze penalty: distance from user center line of sight (yaw=0)
            gaze_angular_dist = self._angular_distance(spatial_coords["azimuth_deg"], head_pose.get("yaw", 0.0))
            gaze_proximity_score = max(0.0, 1.0 - (gaze_angular_dist / (self.horizontal_fov / 2.0 + 1e-5)))

            # Acoustic alignment
            best_audio_match = None
            min_audio_dist = 999.0
            for a_ev in audio_events:
                a_az = float(a_ev.get("azimuth_deg", 0.0))
                dist = self._angular_distance(spatial_coords["azimuth_deg"], a_az)
                if dist < min_audio_dist:
                    min_audio_dist = dist
                    best_audio_match = a_ev

            audio_alignment_score = 0.0
            energy_boost = 0.0
            matched_audio_label = None

            if best_audio_match and min_audio_dist <= 25.0: # within 25 degree cone
                audio_alignment_score = max(0.0, 1.0 - (min_audio_dist / 25.0))
                # Normalize energy_db (-60 to 0 dB typical)
                energy_db = float(best_audio_match.get("energy_db", -30.0))
                energy_boost = min(1.0, max(0.0, (energy_db + 60.0) / 60.0))
                matched_audio_label = best_audio_match.get("label")

            # Salience composite calculation:
            # 40% gaze proximity + 30% visual confidence + 20% audio alignment + 10% acoustic energy
            salience = (
                0.40 * gaze_proximity_score +
                0.30 * conf +
                0.20 * audio_alignment_score +
                0.10 * energy_boost
            )

            fused_entities.append({
                "entity_id": det.get("id", f"ent_{idx}"),
                "label": label,
                "modality": "multimodal_audio_visual" if audio_alignment_score > 0 else "visual_only",
                "salience_score": round(salience, 4),
                "spatial": spatial_coords,
                "gaze_proximity": round(gaze_proximity_score, 4),
                "audio_match": {
                    "matched": audio_alignment_score > 0,
                    "label": matched_audio_label,
                    "angular_delta_deg": round(min_audio_dist, 2) if audio_alignment_score > 0 else None,
                    "energy_boost": round(energy_boost, 3)
                }
            })

        # 2. Check for unmatched auditory-only events (e.g. sound behind the user)
        for a_idx, a_ev in enumerate(audio_events):
            a_az = float(a_ev.get("azimuth_deg", 0.0))
            # see if matched
            matched = any(
                e["audio_match"]["matched"] and e["audio_match"]["label"] == a_ev.get("label")
                for e in fused_entities
            )
            if not matched:
                energy_db = float(a_ev.get("energy_db", -30.0))
                norm_energy = min(1.0, max(0.0, (energy_db + 60.0) / 60.0))
                # audio salience based purely on energy and proximity to front
                az_dist_to_front = self._angular_distance(a_az, head_pose.get("yaw", 0.0))
                front_bias = max(0.2, 1.0 - (az_dist_to_front / 180.0))
                audio_salience = 0.6 * norm_energy + 0.4 * front_bias

                fused_entities.append({
                    "entity_id": f"audio_ent_{a_idx}",
                    "label": a_ev.get("label", "unidentified_acoustic_source"),
                    "modality": "audio_only",
                    "salience_score": round(audio_salience, 4),
                    "spatial": {
                        "azimuth_deg": a_az,
                        "elevation_deg": 0.0,
                        "relative_direction": "left" if a_az < -15 else ("right" if a_az > 15 else "ahead")
                    },
                    "audio_match": {
                        "matched": True,
                        "label": a_ev.get("label"),
                        "energy_db": energy_db
                    }
                })

        # Sort by salience descending
        fused_entities.sort(key=lambda x: x["salience_score"], reverse=True)
        top_focus = fused_entities[0] if fused_entities else None
        self.current_focus = top_focus

        record = {
            "timestamp": time.time(),
            "focal_entity": top_focus,
            "total_entities_grounded": len(fused_entities),
            "head_pose": head_pose,
            "entities": fused_entities
        }
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return record

    def resolve_deictic_reference(
        self,
        utterance: str,
        sensory_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolves natural language deictic phrases:
        'look at that', 'what is this cup', 'the buzzing noise on the left', 'person speaking'.
        """
        state = sensory_state or (self.history[-1] if self.history else self.fuse_sensory_input())
        entities = state.get("entities", [])
        if not entities:
            return {"resolved": False, "reason": "No spatial entities detected in sensory field"}

        utt_lower = utterance.lower()
        candidates = []

        # Directional keywords
        wants_left = any(w in utt_lower for w in ["left", "portside", "on my left"])
        wants_right = any(w in utt_lower for w in ["right", "starboard", "on my right"])
        wants_front = any(w in utt_lower for w in ["front", "center", "ahead", "straight"])

        wants_sound = any(w in utt_lower for w in ["noise", "sound", "humming", "buzzing", "speaking", "voice", "music"])

        for ent in entities:
            score = ent["salience_score"]
            az = ent["spatial"].get("azimuth_deg", 0.0)

            # Spatial filter check
            if wants_left and az >= -5:
                continue
            if wants_right and az <= 5:
                continue
            if wants_front and abs(az) > 30:
                continue

            # Modality preference check
            if wants_sound and ent["modality"] == "visual_only":
                score *= 0.3
            elif wants_sound and ent["modality"] in ("multimodal_audio_visual", "audio_only"):
                score *= 1.5

            # Semantic match
            label = ent["label"].lower()
            if label in utt_lower or any(token in utt_lower for token in label.split("_")):
                score += 1.0

            candidates.append({"entity": ent, "match_weight": round(score, 4)})

        if not candidates:
            # Fallback to top salience entity
            candidates = [{"entity": entities[0], "match_weight": entities[0]["salience_score"]}]

        candidates.sort(key=lambda x: x["match_weight"], reverse=True)
        winner = candidates[0]["entity"]

        return {
            "resolved": True,
            "utterance": utterance,
            "grounded_entity": winner,
            "confidence": min(1.0, candidates[0]["match_weight"]),
            "spatial_guidance": {
                "azimuth_deg": winner["spatial"].get("azimuth_deg"),
                "turn_direction": "turn_left" if winner["spatial"].get("azimuth_deg", 0) < -15 else ("turn_right" if winner["spatial"].get("azimuth_deg", 0) > 15 else "centered")
            }
        }

    def track_focal_salience(self, decay_rate: float = 0.85) -> Dict[str, Any]:
        """Calculates smooth temporal focal tracking over recent sensor frames."""
        if not self.history:
            return {"active_focus": None, "smoothed_entities": []}

        entity_weights = {}
        for idx, frame in enumerate(reversed(self.history[-10:])):
            weight = math.pow(decay_rate, idx)
            for ent in frame.get("entities", []):
                eid = ent["entity_id"]
                if eid not in entity_weights:
                    entity_weights[eid] = {
                        "entity_id": eid,
                        "label": ent["label"],
                        "modality": ent["modality"],
                        "cumulative_score": 0.0,
                        "last_azimuth": ent["spatial"].get("azimuth_deg", 0.0)
                    }
                entity_weights[eid]["cumulative_score"] += ent["salience_score"] * weight

        sorted_focal = sorted(entity_weights.values(), key=lambda x: x["cumulative_score"], reverse=True)
        for s in sorted_focal:
            s["cumulative_score"] = round(s["cumulative_score"], 4)

        return {
            "window_size": min(len(self.history), 10),
            "dominant_focus": sorted_focal[0] if sorted_focal else None,
            "tracked_entities": sorted_focal
        }

    def get_grounding_state(self) -> Dict[str, Any]:
        return {
            "engine": "SpatialAudioVisualGroundingEngine",
            "fov": {"horizontal": self.horizontal_fov, "vertical": self.vertical_fov},
            "history_frames": len(self.history),
            "current_focus": self.current_focus
        }
