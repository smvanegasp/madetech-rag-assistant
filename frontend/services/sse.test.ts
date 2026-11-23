import { describe, expect, it } from 'vitest';
import { createSseParser } from './sse';

const stream =
  'event: tool_step\ndata: {"tool_name":"search_handbook","order":1}\n\n' +
  ': ping\n\n' +
  'event: done\ndata: {"content":"Hi","sources":[]}\n\n';

describe('createSseParser', () => {
  it('parses whole events and skips keep-alive comments', () => {
    const events = createSseParser()(stream);
    expect(events.map(e => e.event)).toEqual(['tool_step', 'done']);
    expect(JSON.parse(events[1].data)).toEqual({ content: 'Hi', sources: [] });
  });

  it('gives the same result however the stream is split into chunks', () => {
    for (let size = 1; size <= stream.length; size++) {
      const feed = createSseParser();
      const events = [];
      for (let i = 0; i < stream.length; i += size) {
        events.push(...feed(stream.slice(i, i + size)));
      }
      expect(events.map(e => e.event), `chunk size ${size}`).toEqual(['tool_step', 'done']);
    }
  });

  it('handles CRLF line endings', () => {
    const events = createSseParser()('event: done\r\ndata: {}\r\n\r\n');
    expect(events).toEqual([{ event: 'done', data: '{}' }]);
  });
});
