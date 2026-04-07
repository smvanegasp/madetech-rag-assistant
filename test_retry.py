"""Quick smoke test for the streaming chat endpoint."""
import httpx
import json

body = {"query": "Compare parental leave and sick leave policies", "history": []}

with httpx.stream("POST", "http://localhost:9481/api/chat/stream",
                   json=body, timeout=120) as r:
    buffer = ""
    events = []
    for chunk in r.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            lines = block.strip().split("\n")
            event_type = ""
            data = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
            if event_type and data:
                events.append((event_type, data))
                if event_type == "tool_step":
                    print(f"TOOL: {data['tool_name']} - {json.dumps(data.get('arguments', {}))[:80]}")

for et, d in events:
    if et == "done":
        print(f"\nSUCCESS: {not d.get('isError', False)}")
        print(f"ANSWER: {d['content'][:200]}...")
        print(f"SOURCES: {len(d.get('sources', []))}")
        print(f"TOOL_STEPS: {len(d.get('tool_steps', []))}")
        if d.get("isError"):
            print("ERROR FLAG SET")
