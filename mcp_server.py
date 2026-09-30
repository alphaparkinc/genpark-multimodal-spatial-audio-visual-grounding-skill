"""MCP Server for Spatial Audio-Visual Grounding Engine."""
import sys
import json
from client import SpatialAudioVisualGroundingEngine

engine = SpatialAudioVisualGroundingEngine()

def handle_call_tool(params):
    name = params.get("name")
    args = params.get("arguments", {})
    if name != "ground_spatial_sensory_stream":
        raise ValueError(f"Unknown tool: {name}")

    action = args.get("action", "fuse_sensory_input")
    if action == "fuse_sensory_input":
        return engine.fuse_sensory_input(
            visual_detections=args.get("visual_detections"),
            audio_events=args.get("audio_events"),
            head_pose=args.get("head_pose")
        )
    elif action == "resolve_deictic_reference":
        return engine.resolve_deictic_reference(
            utterance=args.get("deictic_utterance", "look at this"),
            sensory_state=None
        )
    elif action == "track_focal_salience":
        return engine.track_focal_salience()
    elif action == "get_grounding_state":
        return engine.get_grounding_state()
    else:
        raise ValueError(f"Invalid action: {action}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running self-test...")
        det = [{"id": "obj1", "label": "coffee_mug", "bbox": [0.45, 0.45, 0.1, 0.1], "confidence": 0.95}]
        audio = [{"label": "speaker_voice", "azimuth_deg": -25.0, "energy_db": -12.0}]
        res = engine.fuse_sensory_input(det, audio, {"yaw": 0.0, "pitch": 0.0})
        deictic = engine.resolve_deictic_reference("what is that voice on the left?")
        assert res["total_entities_grounded"] >= 2
        assert deictic["resolved"] is True
        print("Self-test PASSED!")
        sys.exit(0)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            msg_id = req.get("id")
            method = req.get("method")
            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "SpatialAudioVisualGroundingEngine", "version": "1.0.0"},
                        "capabilities": {"tools": {}}
                    }
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [{
                            "name": "ground_spatial_sensory_stream",
                            "description": "Fuse visual detections with spatial acoustic beamforming, compute cross-modal focal salience, and resolve deictic spatial references.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["fuse_sensory_input", "resolve_deictic_reference", "track_focal_salience", "get_grounding_state"]},
                                    "visual_detections": {"type": "array"},
                                    "audio_events": {"type": "array"},
                                    "head_pose": {"type": "object"},
                                    "deictic_utterance": {"type": "string"}
                                },
                                "required": ["action"]
                            }
                        }]
                    }
                }
            elif method == "tools/call":
                res = handle_call_tool(req.get("params", {}))
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
                }
            else:
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            print(json.dumps(resp), flush=True)
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}
            print(json.dumps(err_resp), flush=True)

if __name__ == "__main__":
    main()
